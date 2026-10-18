"""Features available at an origin and explicit future targets, with no look-ahead."""
import numpy as np
import pandas as pd
from src.utils import ROOT, CLASSES, config

TARGETS = ['traffic_volume', 'avg_speed', 'travel_time', 'congestion', 'accident']


def load_dataset():
    """Load the analysis-ready data with its confirmed timezone."""
    cfg = config()
    df = pd.read_csv(ROOT / f"data/processed/flowcast_{cfg['version']}.csv.gz", low_memory=False)
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(cfg['timezone'])
    return df.sort_values(['road_id', 'timestamp']).reset_index(drop=True)


def origin_features(df):
    """Construct causal features; current sensor values are known at the origin."""
    df = df.sort_values(['road_id', 'timestamp']).reset_index(drop=True)
    f = df[['road_id', 'timestamp']].copy()
    for col in ['traffic_volume', 'avg_speed', 'occupancy', 'vehicle_count', 'signal_timing',
                'share_2w', 'share_car', 'share_lcv', 'share_hcv']:
        f[col] = df[f'causal_{col}']
    for col in ['latitude', 'longitude', 'road_capacity', 'temperature', 'rainfall', 'visibility']:
        f[col] = df[col]
    f['road_number'] = df.road_id.str[-3:].astype(int)
    for col in ['traffic_volume', 'avg_speed']:
        group = df.groupby('road_id')[f'causal_{col}']
        for lag in [1, 2, 48]:
            f[f'{col}_lag_{lag}'] = group.shift(lag)
        for n in [4, 8]:
            f[f'{col}_mean_{n}'] = group.transform(lambda x: x.shift().rolling(n).mean())
            f[f'{col}_std_{n}'] = group.transform(lambda x: x.shift().rolling(n).std())
    f['vc_ratio'] = f.traffic_volume / (f.road_capacity / 2)
    f['capacity_headroom'] = f.road_capacity / 2 - f.traffic_volume
    f['rain_flag'] = (df.rainfall > 0).astype(int)
    f['low_visibility'] = (df.visibility < 1000).astype(int)
    f['temperature_band'] = pd.cut(df.temperature, [-np.inf, 15, 25, 35, np.inf], labels=False)
    for name in ['Clear', 'Cloudy', 'Rain', 'Fog']:
        f[f'weather_{name.lower()}'] = (df.weather_condition == name).astype(int)
    return f


def supervised(df, horizons=None, require_targets=True):
    """Join each origin to its exact target timestamp and known calendar flags."""
    cfg = config()
    f = origin_features(df)
    df = df.sort_values(['road_id', 'timestamp']).reset_index(drop=True)
    calendar = df[['timestamp', 'public_holiday', 'event_flag', 'roadwork_flag']].copy()
    calendar['date_key'] = calendar.timestamp.dt.normalize()
    calendar = calendar.drop(columns='timestamp').drop_duplicates('date_key')
    labels = df[['road_id', 'timestamp', 'traffic_volume', 'avg_speed', 'travel_time',
                 'accident_count', 'congestion_level', 'traffic_volume_observed',
                 'avg_speed_observed', 'congestion_observed', 'window_observed']].copy()
    labels = labels.rename(columns={'timestamp': 'target_timestamp'})
    labels['congestion'] = labels.congestion_level.map(dict(zip(CLASSES, range(4))))
    labels['accident'] = (labels.accident_count > 0).astype(float).mask(labels.accident_count.isna())
    for col in ['traffic_volume', 'avg_speed']:
        labels[col] = labels[col].where(labels[f'{col}_observed'])
    labels['congestion'] = labels.congestion.where(labels.congestion_observed)
    labels = labels[['road_id', 'target_timestamp'] + TARGETS].rename(columns={c: f'y_{c}' for c in TARGETS})
    frames = []
    for h in horizons or cfg['horizons']:
        block = f.copy()
        block['horizon'] = h
        block['target_timestamp'] = block.timestamp + pd.Timedelta(minutes=30*h)
        block['date_key'] = block.target_timestamp.dt.normalize()
        block = block.merge(calendar, on='date_key', how='left', validate='many_to_one').drop(columns='date_key')
        hours = block.target_timestamp.dt.hour + block.target_timestamp.dt.minute / 60
        dow = block.target_timestamp.dt.dayofweek
        block['hour_sin'] = np.sin(2*np.pi*hours/24)
        block['hour_cos'] = np.cos(2*np.pi*hours/24)
        block['dow_sin'] = np.sin(2*np.pi*dow/7)
        block['dow_cos'] = np.cos(2*np.pi*dow/7)
        block['weekend'] = (dow >= 5).astype(int)
        block['peak'] = ((hours >= 7) & (hours < 10) | (hours >= 17) & (hours < 20)).astype(int)
        block['holiday_peak'] = block.public_holiday * block.peak
        # Calendar is date-level. Event proximity means known event day, not invented event hours.
        block['event_peak'] = block.event_flag * block.peak
        block = block.merge(labels, on=['road_id', 'target_timestamp'], how='left', validate='one_to_one')
        frames.append(block)
    result = pd.concat(frames, ignore_index=True)
    cols = feature_columns(result)
    result = result.dropna(subset=cols + ([f'y_{c}' for c in TARGETS] if require_targets else []))
    result = result.sort_values(['target_timestamp', 'road_id', 'horizon']).reset_index(drop=True)
    result[cols] = result[cols].astype(np.float32)
    result['horizon'] = result['horizon'].astype('int8')
    assert np.isfinite(result[cols].to_numpy()).all()
    return result


def feature_columns(frame):
    """Allow only numeric predictors, never labels or future observations."""
    return [c for c in frame if c not in ['road_id', 'timestamp', 'target_timestamp'] and not c.startswith('y_')]


def split_masks(frame, cfg=None):
    """Split by target time, purging forecasts whose origins cross the boundary."""
    cfg = cfg or config()
    train_end = pd.Timestamp(cfg['train_end'], tz=cfg['timezone'])
    val_end = pd.Timestamp(cfg['validation_end'], tz=cfg['timezone'])
    return {'train': frame.target_timestamp <= train_end,
            'validation': (frame.timestamp > train_end) & (frame.target_timestamp <= val_end),
            'test': frame.timestamp > val_end}


def time_folds(frame, n_splits=2):
    """Expanding folds on unique timestamps, with a four-window embargo."""
    from sklearn.model_selection import TimeSeriesSplit
    times = np.sort(frame.target_timestamp.unique())
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=4).split(times):
        train = np.flatnonzero(frame.target_timestamp.isin(times[tr]))
        valid = np.flatnonzero(frame.target_timestamp.isin(times[va]) & (frame.timestamp > times[tr[-1]]))
        yield train, valid
