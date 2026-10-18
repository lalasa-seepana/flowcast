"""Randomly initialized GRU and historical bidirectional LSTM benchmarks."""
import copy
import logging
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.features import supervised, feature_columns, split_masks
from src.evaluate import regression_metrics, classification_metrics
from src.utils import ROOT, config, save_json, digest


class SequenceDataset(Dataset):
    """Slice past-only sequences lazily to keep workstation memory bounded."""
    def __init__(self, values, indices, volume, congestion, length):
        self.values, self.indices = values, np.asarray(indices)
        self.volume, self.congestion, self.length = volume, congestion, length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        end = self.indices[i]
        return (torch.from_numpy(self.values[end-self.length+1:end+1]),
                torch.tensor(self.volume[end], dtype=torch.float32),
                torch.tensor(self.congestion[end], dtype=torch.long))


class SequenceForecaster(nn.Module):
    """Train a recurrent volume head and four-class congestion head from scratch."""
    def __init__(self, features, hidden_size, layers, dropout, kind='GRU'):
        super().__init__()
        self.kind = kind
        recurrent = nn.GRU if kind == 'GRU' else nn.LSTM
        self.core = recurrent(features, hidden_size, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0,
                              bidirectional=kind == 'BiLSTM')
        width = hidden_size * (2 if kind == 'BiLSTM' else 1)
        self.dropout = nn.Dropout(dropout)
        self.volume = nn.Linear(width, 1)
        self.congestion = nn.Linear(width, 4)

    def forward(self, x):
        """Map a past-only sequence to volume and congestion logits."""
        _, state = self.core(x)
        hidden = state[0] if self.kind == 'BiLSTM' else state
        final = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.kind == 'BiLSTM' else hidden[-1]
        final = self.dropout(final)
        return self.volume(final).squeeze(1), self.congestion(final)


def predict(model, loader, mean, scale):
    """Return genuine model outputs in original vehicle units."""
    model.eval()
    volumes, classes = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            volume, logits = model(x)
            volumes.append(volume.numpy()*scale + mean)
            classes.append(torch.softmax(logits, dim=1).numpy())
    return np.maximum(0, np.concatenate(volumes)), np.concatenate(classes)


def train(df, cfg=None):
    """Tune two GRU configurations and train the required BiLSTM comparison."""
    cfg = cfg or config()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    frame = supervised(df, horizons=[1], require_targets=False).sort_values(['road_id', 'timestamp']).reset_index(drop=True)
    features = feature_columns(frame)
    masks = split_masks(frame, cfg)
    observed = frame[[f'y_{c}' for c in ['traffic_volume', 'avg_speed', 'travel_time', 'congestion', 'accident']]].notna().all(axis=1)
    raw = frame[features].to_numpy(np.float32)
    bounded = [features.index(c) for c in ['occupancy', 'visibility', 'share_2w', 'share_car', 'share_lcv', 'share_hcv']]
    unbounded = [i for i in range(len(features)) if i not in bounded]
    standard, minmax = StandardScaler(), MinMaxScaler()
    X = raw.copy()
    X[:, unbounded] = standard.fit(raw[masks['train']][:, unbounded]).transform(raw[:, unbounded])
    X[:, bounded] = minmax.fit(raw[masks['train']][:, bounded]).transform(raw[:, bounded])
    # No clipping: a held-out value outside the training range remains visible.
    mean = float(frame.loc[masks['train'] & observed, 'y_traffic_volume'].mean())
    scale = float(frame.loc[masks['train'] & observed, 'y_traffic_volume'].std())
    y = ((frame.y_traffic_volume.to_numpy()-mean)/scale).astype(np.float32)
    c = frame.y_congestion.fillna(0).to_numpy(dtype=np.int64)
    val_times = sorted(frame.loc[masks['validation'], 'target_timestamp'].unique())
    mid = val_times[len(val_times)//2]
    tuning = masks['validation'] & (frame.target_timestamp < mid)
    calibration = masks['validation'] & (frame.timestamp >= mid)
    histories, trials, score_rows = [], [], []
    best_gru = None
    for kind in ['GRU', 'BiLSTM']:
        choices = cfg['dl']['candidates'] if kind == 'GRU' else [best_gru['parameters']]
        best = None
        for trial, params in enumerate(choices):
            torch.manual_seed(cfg['seed'])
            np.random.seed(cfg['seed'])
            length = params['sequence_length']
            # Ensure consecutive timestamps and no sequence crosses a road boundary.
            valid = observed & frame.road_id.eq(frame.road_id.shift(length-1))
            valid &= frame.timestamp.sub(frame.timestamp.shift(length-1)).eq(pd.Timedelta(minutes=30*(length-1)))
            indices = {name: np.flatnonzero(mask & valid) for name, mask in
                       [('train', masks['train']), ('validation', tuning), ('calibration', calibration), ('test', masks['test'])]}
            loaders = {}
            for name, ids in indices.items():
                ds = SequenceDataset(X, ids, y, c, length)
                loaders[name] = DataLoader(ds, batch_size=params['batch_size'], shuffle=False, num_workers=0)
            if len(indices['test']) != int((masks['test'] & observed).sum()):
                raise ValueError('Deep and classical test windows do not match.')
            model = SequenceForecaster(len(features), params['hidden_size'], params['layers'], params['dropout'], kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5)
            best_loss, stale, state = float('inf'), 0, None
            for epoch in range(1, cfg['dl']['epochs']+1):
                train_loss = train_mse = val_loss = val_mse = 0.0
                for phase in ['train', 'validation']:
                    model.train(phase == 'train')
                    total, total_mse, count = 0., 0., 0
                    with torch.set_grad_enabled(phase == 'train'):
                        for x, volume, congestion in loaders[phase]:
                            pred, logits = model(x)
                            mse = nn.functional.mse_loss(pred, volume)
                            loss = mse + .25*nn.functional.cross_entropy(logits, congestion)
                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                                optimizer.step()
                            total += loss.item()*len(x)
                            total_mse += mse.item()*len(x)
                            count += len(x)
                    if phase == 'train':
                        train_loss, train_mse = total/count, total_mse/count
                    else:
                        val_loss, val_mse = total/count, total_mse/count
                histories.append({'model': kind, 'trial': trial, 'epoch': epoch,
                                  'train_loss': train_loss, 'validation_loss': val_loss,
                                  'train_rmse': np.sqrt(train_mse)*scale,
                                  'validation_rmse': np.sqrt(val_mse)*scale,
                                  'learning_rate': optimizer.param_groups[0]['lr']})
                logging.info('%s trial %d epoch %d: train %.4f validation %.4f', kind, trial, epoch, train_loss, val_loss)
                scheduler.step(val_loss)
                if val_loss < best_loss - 1e-5:
                    best_loss, stale, state = val_loss, 0, copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= cfg['dl']['patience']:
                    break
            model.load_state_dict(state)
            vp, cp = predict(model, loaders['validation'], mean, scale)
            val_rmse = regression_metrics(frame.iloc[indices['validation']].y_traffic_volume, vp)['rmse']
            trials.append({'model': kind, 'trial': trial, **params, 'best_validation_loss': best_loss,
                           'validation_rmse': val_rmse, 'epochs_run': epoch})
            if best is None or val_rmse < best['validation_rmse']:
                best = {'model': model, 'parameters': params, 'validation_rmse': val_rmse,
                        'indices': indices, 'loaders': loaders, 'trial': trial, 'state': state}
        if kind == 'GRU':
            best_gru = best
        model, ids, loaders = best['model'], best['indices'], best['loaders']
        calib_pred, _ = predict(model, loaders['calibration'], mean, scale)
        errors = np.abs(frame.iloc[ids['calibration']].y_traffic_volume.to_numpy()-calib_pred)
        q = min(1, np.ceil((len(errors)+1)*cfg['interval_coverage'])/len(errors))
        radius = float(np.quantile(errors, q, method='higher'))
        start = time.perf_counter()
        volume, probabilities = predict(model, loaders['test'], mean, scale)
        elapsed = time.perf_counter()-start
        test = frame.iloc[ids['test']]
        out = test[['road_id', 'timestamp', 'target_timestamp', 'horizon', 'y_traffic_volume', 'y_congestion']].copy()
        out['issued_at'] = out.timestamp + pd.Timedelta(minutes=30)
        out['target_window_end'] = out.target_timestamp + pd.Timedelta(minutes=30)
        out['model'] = kind
        out['model_version'] = f'{kind.lower()}_{cfg["version"]}'
        out['pred_traffic_volume'] = volume
        out['lower_traffic_volume'] = np.maximum(0, volume-radius)
        out['upper_traffic_volume'] = volume+radius
        for j in range(4):
            out[f'p_{j}'] = probabilities[:, j]
        out.to_csv(ROOT / f'data/processed/{kind.lower()}_{cfg["version"]}_test.csv.gz', index=False)
        metrics = [{'model': kind, 'task': 'traffic_volume', 'horizon': 1, 'n': len(test),
                    **regression_metrics(test.y_traffic_volume, volume)},
                   {'model': kind, 'task': 'congestion', 'horizon': 1, 'n': len(test),
                    **classification_metrics(test.y_congestion, probabilities, 'congestion')}]
        score_rows.extend(metrics)
        checkpoint = {'state_dict': best['state'], 'features': features, 'parameters': best['parameters'],
                      'kind': kind, 'target_mean': mean, 'target_scale': scale, 'interval_radius': radius}
        torch.save(checkpoint, ROOT / f'models/{kind.lower()}_{cfg["version"]}.pt')
        joblib.dump({'standard': standard, 'minmax': minmax, 'bounded': bounded, 'unbounded': unbounded},
                    ROOT / f'models/{kind.lower()}_{cfg["version"]}_scalers.joblib')
        card = {'name': kind, 'version': cfg['version'],
                'artifact': f'{kind.lower()}_{cfg["version"]}.pt',
                'scaler_artifact': f'{kind.lower()}_{cfg["version"]}_scalers.joblib', 'initialization': 'Seeded random weights; no pretrained model',
                'training_origin_start': str(frame.iloc[ids['train']].timestamp.min()),
                'training_target_end': str(frame.iloc[ids['train']].target_timestamp.max()),
                'validation_target_end': str(frame.iloc[ids['validation']].target_timestamp.max()),
                'test_origin_start': str(test.timestamp.min()), 'test_target_end': str(test.target_timestamp.max()),
                'features': features, 'hyperparameters': best['parameters'], 'selected_trial': best['trial'],
                'test_metrics': metrics, 'seed': cfg['seed'], 'test_inference_seconds': elapsed,
                'interval_radius': radius, 'interval_coverage': cfg['interval_coverage'],
                'training_rows': len(ids['train']), 'test_rows': len(test),
                'source_manifest_sha256': digest(ROOT / 'data/processed/source_manifest.json'),
                'limitations': 'Next-window only. Class softmax probabilities are uncalibrated. BiLSTM is a historical benchmark; both directions see only the past input window.'}
        save_json(ROOT / f'models/{kind.lower()}_{cfg["version"]}_card.json', card)
        pd.DataFrame(histories).to_csv(ROOT / 'reports/dl_training_curves.csv', index=False)
        pd.DataFrame(trials).to_csv(ROOT / 'reports/dl_search.csv', index=False)
        pd.DataFrame(score_rows).to_csv(ROOT / 'reports/dl_scoreboard.csv', index=False)
    logging.info('Deep-learning training and exact-window benchmark complete.')
