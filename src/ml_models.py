"""Time-aware classical baselines, model persistence and four-horizon inference."""
import logging
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from src.features import supervised, feature_columns, split_masks, time_folds
from src.utils import ROOT, config, save_json, digest, save_model, save_csv


class NumpyLinearRegression(RegressorMixin, BaseEstimator):
    """Batch gradient descent: L=mean((XW+b-y)^2), dW=2 X.T error / n."""
    def __init__(self, learning_rate=.015, epochs=180):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        """Fit feature/target scaling on training data and descend the MSE."""
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.one_target_ = y.ndim == 1
        if self.one_target_:
            y = y[:, None]
        self.x_mean_, self.x_scale_ = X.mean(0), X.std(0)
        self.x_scale_[self.x_scale_ < 1e-8] = 1
        self.y_mean_, self.y_scale_ = y.mean(0), y.std(0)
        self.y_scale_[self.y_scale_ < 1e-8] = 1
        Z = (X - self.x_mean_) / self.x_scale_
        Y = (y - self.y_mean_) / self.y_scale_
        Z = np.column_stack([np.ones(len(Z)), Z])
        self.weights_ = np.zeros((Z.shape[1], Y.shape[1]))
        self.loss_curve_ = []
        # X.T X and X.T y are sufficient statistics for the full-batch gradient.
        gram, cross = Z.T @ Z / len(Z), Z.T @ Y / len(Z)
        target_square = np.mean(Y**2)
        for _ in range(self.epochs):
            grad = 2 * (gram @ self.weights_ - cross)
            self.weights_ -= self.learning_rate * grad
            loss = (np.trace(self.weights_.T @ gram @ self.weights_) -
                    2*np.sum(self.weights_*cross))/Y.shape[1] + target_square
            self.loss_curve_.append(float(loss))
        return self

    def predict(self, X):
        """Predict through an explicit matrix product and inverse target scaling."""
        Z = (np.asarray(X) - self.x_mean_) / self.x_scale_
        pred = np.column_stack([np.ones(len(Z)), Z]) @ self.weights_
        pred = pred * self.y_scale_ + self.y_mean_
        return pred.ravel() if self.one_target_ else pred


def candidates(task, cfg):
    """Small, explicit search spaces keep the required comparisons practical."""
    s, n = cfg['seed'], cfg['threads']
    if task == 'flow':
        return {
            'Numpy GD': [NumpyLinearRegression(learning_rate=r) for r in [.01, .02]],
            'Linear Regression': [make_pipeline(StandardScaler(), LinearRegression())],
            'Decision Tree': [DecisionTreeRegressor(max_depth=d, min_samples_leaf=40, random_state=s) for d in [8, 12]],
            'Random Forest': [RandomForestRegressor(n_estimators=32, max_depth=d, min_samples_leaf=20,
                                                    max_features=.8, max_samples=.35, n_jobs=n, random_state=s) for d in [9, 13]],
            'XGBoost': [XGBRegressor(n_estimators=160, max_depth=d, learning_rate=.07,
                                      subsample=.85, colsample_bytree=.9, tree_method='hist',
                                      n_jobs=n, random_state=s) for d in [3, 5]],
        }
    constructor = DecisionTreeClassifier
    return {
        'Decision Tree': [constructor(max_depth=d, min_samples_leaf=50, class_weight='balanced', random_state=s) for d in [7, 11]],
        'Random Forest': [RandomForestClassifier(n_estimators=32, max_depth=d, min_samples_leaf=30,
                                                 class_weight='balanced', max_samples=.35, n_jobs=n, random_state=s) for d in [9, 13]],
        'XGBoost': [XGBClassifier(n_estimators=160, max_depth=d, learning_rate=.07,
                                  subsample=.85, colsample_bytree=.9, tree_method='hist',
                                  n_jobs=n, random_state=s) for d in [3, 5]],
        'SVM': [make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight='balanced',
                                                         dual=False, max_iter=4000, random_state=s)) for c in [.1, 1]],
    }


def probabilities(model, X):
    """Return probabilities, or provisional margins before held-out calibration."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    margin = model.decision_function(X)
    return np.column_stack([1-expit(margin), expit(margin)]) if margin.ndim == 1 else softmax(margin, axis=1)


def travel_inputs(flow, frame, lengths):
    """Use forecast volume/speed, capacity and training-derived segment length."""
    length = frame.road_id.map(lengths).to_numpy()
    speed = np.clip(flow[:, 1], 1, 200)
    return np.column_stack([flow[:, 0], speed, frame.road_capacity.to_numpy(), length, 60*length/speed])


def predict_bundle(bundle, frame):
    """Run a persisted model without any fit or future outcome access."""
    X = frame[bundle['features']].to_numpy(dtype=np.float32)
    if bundle['task'] == 'flow':
        flow = bundle['model'].predict(X)
        flow[:, 0] = np.maximum(0, flow[:, 0])
        flow[:, 1] = np.clip(flow[:, 1], 1, 200)
        travel = np.maximum(.01, bundle['travel_model'].predict(travel_inputs(flow, frame, bundle['lengths'])))
        return np.column_stack([flow, travel])
    return probabilities(bundle['model'], X)


def train(df, cfg=None):
    """Tune within training folds; select on validation; score the untouched test."""
    from src.evaluate import regression_metrics, classification_metrics
    cfg = cfg or config()
    frame = supervised(df)
    masks = split_masks(frame, cfg)
    train = frame.loc[masks['train']].reset_index(drop=True)
    validation = frame.loc[masks['validation']].reset_index(drop=True)
    test = frame.loc[masks['test']].reset_index(drop=True)
    mid = sorted(validation.target_timestamp.unique())[len(validation.target_timestamp.unique())//2]
    tune = validation.loc[validation.target_timestamp < mid].reset_index(drop=True)
    calibration = validation.loc[validation.timestamp >= mid].reset_index(drop=True)
    features = feature_columns(frame)
    split_counts = {k: int(v.sum()) for k, v in masks.items()} | {
        'boundary_excluded': int((~(masks['train'] | masks['validation'] | masks['test'])).sum()),
        'feature_or_target_excluded': int(len(df)*len(cfg['horizons']) - len(frame))}
    del frame, validation, masks
    import gc
    gc.collect()
    X = train[features].to_numpy(np.float32)
    Xt = tune[features].to_numpy(np.float32)
    folds = list(time_folds(train, cfg['cv_folds']))
    lengths = (df.loc[(df.timestamp <= pd.Timestamp(cfg['train_end'], tz=cfg['timezone'])) & df.avg_speed_observed]
               .assign(length=lambda x: x.avg_speed*x.travel_time/60).groupby('road_id').length.median().to_dict())
    rows, cv_rows, val_rows, pred_frames, cards = [], [], [], [], []
    selected = {}
    regression_best = {t: float('inf') for t in ['flow', 'speed', 'travel_time']}
    for task in ['flow', 'congestion', 'accident']:
        y = train[['y_traffic_volume', 'y_avg_speed']].to_numpy() if task == 'flow' else train[f'y_{task}'].to_numpy(dtype=int)
        best_value = np.inf if task == 'flow' else -np.inf
        for name, choices in candidates(task, cfg).items():
            logging.info('Tuning %s / %s on %d examples', task, name, len(train))
            trial_scores = []
            for ci, candidate in enumerate(choices):
                scores = []
                for fi, (tr, va) in enumerate(folds):
                    model = clone(candidate).fit(X[tr], y[tr])
                    if task == 'flow':
                        from sklearn.metrics import root_mean_squared_error
                        score = root_mean_squared_error(y[va, 0], model.predict(X[va])[:, 0])
                    else:
                        from sklearn.metrics import f1_score, roc_auc_score
                        p = probabilities(model, X[va])
                        score = f1_score(y[va], p.argmax(1), average='macro') if task == 'congestion' else roc_auc_score(y[va], p[:, 1])
                    scores.append(float(score))
                    cv_rows.append({'task': task, 'model': name, 'candidate': ci, 'fold': fi+1,
                                    'score': float(score), 'train_end': str(train.target_timestamp.iloc[tr].max()),
                                    'validation_start': str(train.timestamp.iloc[va].min()),
                                    'parameters': str(candidate.get_params())})
                trial_scores.append(float(np.mean(scores)))
            chosen = int(np.argmin(trial_scores) if task == 'flow' else np.argmax(trial_scores))
            model = clone(choices[chosen]).fit(X, y)
            bundle = {'task': task, 'name': name, 'version': cfg['version'], 'features': features,
                      'model': model, 'lengths': lengths}
            if task == 'flow':
                # Segment travel time is learned from physical flow/speed/capacity inputs.
                # Training uses measured inputs; validation/test always use forecast inputs.
                travel_model = clone(choices[chosen])
                travel_model.fit(travel_inputs(y, train, lengths), train.y_travel_time.to_numpy())
                bundle['travel_model'] = travel_model
                pred = predict_bundle(bundle, tune)
                value = regression_metrics(tune.y_traffic_volume, pred[:, 0])['rmse']
                target_scores = {}
                for j, (key, target) in enumerate(zip(['flow', 'speed', 'travel_time'], ['traffic_volume', 'avg_speed', 'travel_time'])):
                    target_scores[key] = regression_metrics(tune[f'y_{target}'], pred[:, j])['rmse']
            else:
                p = probabilities(model, Xt)
                m = classification_metrics(tune[f'y_{task}'], p, task)
                value = m['macro_f1'] if task == 'congestion' else m['roc_auc']
                # Calibration is strictly later than selection, and earlier than test.
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
                calibrated.fit(calibration[features].to_numpy(np.float32), calibration[f'y_{task}'].to_numpy(dtype=int))
                bundle['model'] = calibrated
            slug = f"{task}_{name.lower().replace(' ', '_')}_{cfg['version']}"
            path = ROOT / f'models/{slug}.joblib'
            save_model(bundle, path)
            val_rows.append({'task': task, 'model': name, 'selection_score': value,
                             'cv_mean': trial_scores[chosen], 'candidate': chosen,
                             'speed_validation_rmse': target_scores['speed'] if task == 'flow' else None,
                             'travel_validation_rmse': target_scores['travel_time'] if task == 'flow' else None})
            better = value < best_value if task == 'flow' else value > best_value
            if task == 'flow':
                for key, score in target_scores.items():
                    if score < regression_best[key]:
                        regression_best[key] = score
                        selected[key] = path.name
            elif better:
                best_value = value
                selected[task] = path.name
            start = time.perf_counter()
            prediction = predict_bundle(bundle, test)
            duration = time.perf_counter() - start
            out = test[['road_id', 'timestamp', 'target_timestamp', 'horizon'] + [f'y_{t}' for t in ['traffic_volume', 'avg_speed', 'travel_time', 'congestion', 'accident']]].copy()
            out['issued_at'] = out.timestamp + pd.Timedelta(minutes=30)
            out['target_window_end'] = out.target_timestamp + pd.Timedelta(minutes=30)
            out['model'] = name
            out['task'] = task
            out['model_version'] = slug
            if task == 'flow':
                residual = np.abs(calibration[['y_traffic_volume', 'y_avg_speed', 'y_travel_time']].to_numpy() - predict_bundle(bundle, calibration))
                intervals = {}
                for h in cfg['horizons']:
                    errors = residual[calibration.horizon.eq(h)]
                    q = min(1, np.ceil((len(errors)+1)*cfg['interval_coverage'])/len(errors))
                    intervals[str(h)] = np.quantile(errors, q, axis=0, method='higher').tolist()
                bundle['intervals'] = intervals
                for j, target in enumerate(['traffic_volume', 'avg_speed', 'travel_time']):
                    out[f'pred_{target}'] = prediction[:, j]
                    radius = out.horizon.map(lambda h: intervals[str(int(h))][j]).to_numpy()
                    out[f'lower_{target}'] = np.maximum(0, prediction[:, j]-radius)
                    out[f'upper_{target}'] = prediction[:, j]+radius
                    for h in cfg['horizons']:
                        mask = test.horizon.eq(h)
                        m = regression_metrics(test.loc[mask, f'y_{target}'], prediction[mask, j])
                        rows.append({'model': name, 'task': target, 'horizon': h, 'n': int(mask.sum()), **m})
                if name == 'Numpy GD':
                    pd.DataFrame({'epoch': np.arange(1, len(model.loss_curve_)+1), 'mse_scaled': model.loss_curve_}).to_csv(ROOT / 'reports/numpy_gd_loss.csv', index=False)
            else:
                for j in range(prediction.shape[1]):
                    out[f'p_{j}'] = prediction[:, j]
                for h in cfg['horizons']:
                    mask = test.horizon.eq(h)
                    m = classification_metrics(test.loc[mask, f'y_{task}'], prediction[mask], task)
                    rows.append({'model': name, 'task': task, 'horizon': h, 'n': int(mask.sum()), **m})
            save_csv(out, ROOT / f'data/processed/{slug}_test.csv.gz')
            save_model(bundle, path)
            card = {'name': name, 'task': task, 'artifact': path.name, 'model_version': slug,
                    'training_start': str(train.timestamp.min()), 'training_target_end': str(train.target_timestamp.max()),
                    'validation_selection_end': str(tune.target_timestamp.max()),
                    'calibration_start': str(calibration.timestamp.min()), 'calibration_end': str(calibration.target_timestamp.max()),
                    'test_origin_start': str(test.timestamp.min()), 'test_target_end': str(test.target_timestamp.max()),
                    'training_rows': len(train), 'test_rows': len(test), 'seed': cfg['seed'],
                    'features': features, 'hyperparameters': choices[chosen].get_params(),
                    'travel_features': ['forecast_volume', 'forecast_speed', 'road_capacity', 'training_segment_length_km', 'physical_travel_minutes'] if task == 'flow' else None,
                    'travel_hyperparameters': bundle['travel_model'].get_params() if task == 'flow' else None,
                    'cv_scores': trial_scores, 'selected_candidate': chosen,
                    'test_inference_seconds': duration, 'source_manifest_sha256': digest(ROOT / 'data/processed/source_manifest.json'),
                    'test_metrics': [r for r in rows if r['model'] == name and r['task'] in (['traffic_volume', 'avg_speed', 'travel_time'] if task == 'flow' else [task])],
                    'limitations': 'Offline rolling-origin forecasts. Future weather is unavailable. Travel model uses measured flow/speed during training and predicted inputs during evaluation.'}
            save_json(ROOT / f'models/{slug}_card.json', card)
            cards.append(card)
            logging.info('Finished %s / %s; validation score %.4f', task, name, value)
    pd.DataFrame(rows).to_csv(ROOT / 'reports/ml_scoreboard.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(ROOT / 'reports/cv_results.csv', index=False)
    pd.DataFrame(val_rows).to_csv(ROOT / 'reports/model_selection.csv', index=False)
    save_json(ROOT / 'models/selected_models.json', selected)
    save_json(ROOT / 'reports/split_counts.json', split_counts)
    forecast = forecast_latest(df)
    save_csv(forecast, ROOT / 'data/processed/latest_forecasts.csv')
    logging.info('Classical training complete; selected %s', selected)


def forecast_latest(df, origin=None, roads=None):
    """Forecast within supplied calendar coverage, never inventing future flags."""
    import json
    cfg = config()
    frame = supervised(df, require_targets=False)
    if origin is None:
        # Use latest origin with every requested horizon and all roads represented.
        expected = len(cfg['horizons'])*df.road_id.nunique()
        counts = frame.groupby('timestamp').size()
        valid = counts[counts == expected]
        if valid.empty:
            raise ValueError('Not enough valid history/calendar coverage for four horizons.')
        origin = valid.index.max()
    frame = frame.loc[frame.timestamp.eq(origin)].copy()
    if roads is not None:
        frame = frame.loc[frame.road_id.isin(roads)]
    expected_roads = list(roads) if roads is not None else list(df.road_id.unique())
    expected_rows = len(expected_roads) * len(cfg['horizons'])
    if len(frame) != expected_rows:
        raise ValueError(f'Incomplete forecast inputs: {len(frame)} of {expected_rows} road/horizon rows are available. Check past history and calendar coverage.')
    selected = json.loads((ROOT / 'models/selected_models.json').read_text())
    out = frame[['road_id', 'timestamp', 'target_timestamp', 'horizon']].copy()
    for task, artifact in selected.items():
        bundle = joblib.load(ROOT / 'models' / artifact)
        pred = predict_bundle(bundle, frame)
        out[f'{task}_model'] = artifact
        if task in ['flow', 'speed', 'travel_time']:
            target = {'flow': 'traffic_volume', 'speed': 'avg_speed', 'travel_time': 'travel_time'}[task]
            j = ['traffic_volume', 'avg_speed', 'travel_time'].index(target)
            for j, target in [(j, target)]:
                out[target] = pred[:, j]
                radius = frame.horizon.map(lambda h: bundle['intervals'][str(int(h))][j]).to_numpy()
                out[f'{target}_lower'] = np.maximum(0, pred[:, j]-radius)
                out[f'{target}_upper'] = pred[:, j]+radius
        elif task == 'congestion':
            from src.utils import CLASSES
            out['congestion_level'] = np.asarray(CLASSES)[pred.argmax(1)]
            out['congestion_confidence'] = pred.max(1)
            for j, name in enumerate(CLASSES):
                out[f'probability_{name}'] = pred[:, j]
        else:
            out['accident_risk'] = pred[:, 1]
            out['risk_confidence'] = np.maximum(pred[:, 1], 1-pred[:, 1])
    out['issued_at'] = out.timestamp + pd.Timedelta(minutes=30)
    out['target_window_end'] = out.target_timestamp + pd.Timedelta(minutes=30)
    out['interval_coverage'] = cfg['interval_coverage']
    out['source_manifest_sha256'] = digest(ROOT / 'data/processed/source_manifest.json')
    out['dataset_version'] = cfg['version']
    return out
