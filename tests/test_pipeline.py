"""Verify real artifacts, causal timing and model reloads without fabricated outputs."""
import hashlib
import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.utils import ROOT, config, CLASSES
from src.features import load_dataset, supervised, feature_columns, split_masks, time_folds
from src.clean import congestion
from src.ml_models import predict_bundle, NumpyLinearRegression


class DataAndTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        # Real source rows for two roads are enough to exercise timing logic.
        cls.small = cls.df.loc[cls.df.road_id.isin(['NL-001', 'NL-002'])].copy()
        cls.frame = supervised(cls.small)

    def test_raw_files_match_manifest(self):
        manifest = json.loads((ROOT / 'data/processed/source_manifest.json').read_text())
        source = Path('/home/lalasa/Downloads/Internmo/e66b00359860d20d922be58a65e87993')
        for name, entry in manifest.items():
            raw = (ROOT / 'data/raw' / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sha256'])
            if source.exists():
                self.assertEqual(raw, (source / name).read_bytes())

    def test_missing_required_column_is_rejected(self):
        from src.ingest import validate
        source = pd.read_csv(ROOT / 'data/raw/traffic_sensor_log.csv', nrows=10)
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            validate(source.drop(columns='road_id'), 'traffic_sensor_log', [], persist_rejected=False)

    def test_grid_reconciliation_and_flags(self):
        quality = json.loads((ROOT / 'reports/data_quality.json').read_text())
        self.assertEqual(len(self.df), 25*151*48)
        self.assertFalse(self.df.duplicated(['road_id', 'timestamp']).any())
        self.assertEqual(quality['source_rows']['traffic_sensor_log']-quality['duplicates']+
                         quality['absent_windows'], len(self.df))
        self.assertEqual(int((~self.df.window_observed).sum()), quality['absent_windows'])
        self.assertFalse(self.df.loc[~self.df.window_observed, 'traffic_volume_observed'].any())

    def test_dictionary_congestion_rule_on_real_rows(self):
        source = pd.read_csv(ROOT / 'data/raw/traffic_sensor_log.csv')
        source = source.drop_duplicates(['road_id', 'date', 'time'])
        blank = source.loc[source.congestion_level.isna() & source.traffic_volume.ge(0)]
        expected = congestion(blank.traffic_volume, blank.road_capacity)
        actual = self.df.set_index(['road_id', 'date', 'time']).congestion_level
        values = actual.reindex(pd.MultiIndex.from_frame(blank[['road_id', 'date', 'time']]))
        self.assertEqual(list(values), list(expected))

    def test_features_and_targets_are_chronological(self):
        frame = self.frame
        self.assertTrue(np.isfinite(frame[feature_columns(frame)].to_numpy()).all())
        self.assertTrue((frame.target_timestamp-frame.timestamp == pd.to_timedelta(frame.horizon*30, unit='min')).all())
        masks = split_masks(frame)
        self.assertLess(frame.loc[masks['train'], 'target_timestamp'].max(), frame.loc[masks['validation'], 'timestamp'].min())
        self.assertLess(frame.loc[masks['validation'], 'target_timestamp'].max(), frame.loc[masks['test'], 'timestamp'].min())
        train = frame.loc[masks['train']].reset_index(drop=True)
        for tr, va in time_folds(train):
            self.assertLess(train.target_timestamp.iloc[tr].max(), train.timestamp.iloc[va].min())

    def test_past_features_do_not_change_when_future_is_removed(self):
        cut = pd.Timestamp('2025-03-15 12:00', tz=config()['timezone'])
        past = self.small.loc[self.small.timestamp <= cut]
        short = supervised(past, require_targets=False)
        whole = supervised(self.small, require_targets=False)
        keys = ['road_id', 'timestamp', 'horizon']
        joined = short.merge(whole, on=keys, suffixes=('_short', '_whole'), validate='one_to_one')
        for col in feature_columns(short):
            if col not in keys:
                np.testing.assert_allclose(joined[f'{col}_short'], joined[f'{col}_whole'], rtol=0, atol=0)

    def test_lag_48_is_previous_day_for_same_road(self):
        first = self.frame.iloc[100]
        previous = self.small.loc[(self.small.road_id == first.road_id) &
                                  (self.small.timestamp == first.timestamp-pd.Timedelta(days=1))]
        self.assertAlmostEqual(first.traffic_volume_lag_48, previous.causal_traffic_volume.iloc[0], places=3)

    def test_cleaning_causal_values_match_truncated_real_sources(self):
        from src.clean import prepare
        from src.ingest import SCHEMAS
        raw = {name: pd.read_csv(ROOT / f'data/raw/{name}.csv') for name in SCHEMAS}
        traffic = raw['traffic_sensor_log']
        cutoff = '2025-01-03 12:00'
        times = traffic.date + ' ' + traffic.time
        raw['traffic_sensor_log'] = traffic.loc[traffic.road_id.isin(['NL-001', 'NL-002']) & (times <= cutoff)]
        repaired, _ = prepare(raw_tables=raw, persist=False)
        reference = self.small.loc[self.small.timestamp <= pd.Timestamp(cutoff, tz=config()['timezone'])]
        cols = [c for c in reference if c.startswith('causal_')]
        joined = repaired.merge(reference, on=['road_id', 'timestamp'], suffixes=('_short', '_full'))
        for col in cols:
            np.testing.assert_allclose(joined[f'{col}_short'], joined[f'{col}_full'], equal_nan=True)

    def test_numpy_gradient_descent_reduces_real_data_loss(self):
        part = self.frame.head(1000)
        model = NumpyLinearRegression(epochs=60).fit(part[feature_columns(part)], part.y_traffic_volume)
        self.assertLess(model.loss_curve_[-1], model.loss_curve_[0])
        self.assertTrue(np.isfinite(model.predict(part[feature_columns(part)])).all())


class ArtifactTests(unittest.TestCase):
    def test_reload_predictions_match_saved_outputs(self):
        df = load_dataset()
        frame = supervised(df.loc[df.road_id.eq('NL-001')])
        selected = json.loads((ROOT / 'models/selected_models.json').read_text())
        for artifact in set(selected.values()):
            bundle = joblib.load(ROOT / 'models' / artifact)
            saved = pd.read_csv(ROOT / 'data/processed' / artifact.replace('.joblib', '_test.csv.gz'))
            saved = saved.loc[saved.road_id.eq('NL-001')].head(12)
            saved['timestamp'] = pd.to_datetime(saved.timestamp, utc=True).dt.tz_convert(config()['timezone'])
            joined = saved.merge(frame, on=['road_id', 'timestamp', 'horizon'], suffixes=('_saved', ''), validate='one_to_one')
            actual = predict_bundle(bundle, joined)
            cols = ['pred_traffic_volume', 'pred_avg_speed', 'pred_travel_time'] if bundle['task'] == 'flow' else [f'p_{j}' for j in range(actual.shape[1])]
            np.testing.assert_allclose(actual, joined[cols].to_numpy(), rtol=2e-5, atol=2e-4)

    def test_deep_weights_reload_and_test_keys_match(self):
        import torch
        from src.dl_model import SequenceForecaster
        cfg = config()
        reference = pd.read_csv(ROOT / f'data/processed/flow_xgboost_{cfg["version"]}_test.csv.gz')
        keys = ['road_id', 'timestamp', 'target_timestamp']
        reference = set(map(tuple, reference.loc[reference.horizon.eq(1), keys].to_numpy()))
        for kind in ['gru', 'bilstm']:
            checkpoint = torch.load(ROOT / f'models/{kind}_{cfg["version"]}.pt', map_location='cpu', weights_only=False)
            params = checkpoint['parameters']
            model = SequenceForecaster(len(checkpoint['features']), params['hidden_size'], params['layers'], params['dropout'], checkpoint['kind'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            output = pd.read_csv(ROOT / f'data/processed/{kind}_{cfg["version"]}_test.csv.gz')
            self.assertEqual(reference, set(map(tuple, output[keys].to_numpy())))
            self.assertTrue((output.lower_traffic_volume <= output.pred_traffic_volume).all())
            self.assertTrue((output.upper_traffic_volume >= output.pred_traffic_volume).all())

    def test_forecast_issue_time_and_missing_history(self):
        from src.ml_models import forecast_latest
        df = load_dataset()
        with self.assertRaises(ValueError):
            forecast_latest(df, origin=df.timestamp.min())
        latest = pd.read_csv(ROOT / 'data/processed/latest_forecasts.csv')
        self.assertEqual(len(latest), 100)
        issue = pd.to_datetime(latest.issued_at, utc=True)
        end = pd.to_datetime(latest.target_window_end, utc=True)
        self.assertTrue((end-issue == pd.to_timedelta(latest.horizon*30, unit='min')).all())

    def test_scoreboard_metrics_match_saved_predictions(self):
        score = pd.read_csv(ROOT / 'reports/ml_scoreboard.csv')
        out = pd.read_csv(ROOT / f'data/processed/flow_xgboost_{config()["version"]}_test.csv.gz')
        for h in [1, 2, 3, 4]:
            block = out.loc[out.horizon.eq(h)]
            expected = np.sqrt(np.mean((block.y_traffic_volume-block.pred_traffic_volume)**2))
            row = score.loc[(score.model == 'XGBoost') & (score.task == 'traffic_volume') & (score.horizon == h)]
            self.assertAlmostEqual(expected, row.rmse.iloc[0], places=5)


if __name__ == '__main__':
    unittest.main()
