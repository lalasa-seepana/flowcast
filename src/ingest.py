"""Load the three original tables and quarantine invalid records explicitly."""
import json
import logging
import numpy as np
import pandas as pd
from src.utils import ROOT, CLASSES, config, digest, save_json

SCHEMAS = {
    'calendar_events': 'date public_holiday holiday_name event_flag event_name roadwork_flag'.split(),
    'weather_observations': 'station_id date time weather_condition temperature rainfall visibility'.split(),
    'traffic_sensor_log': ('road_id road_name latitude longitude weather_station_id date time '
                           'traffic_volume vehicle_count vehicle_type_dist avg_speed occupancy '
                           'congestion_level travel_time accident_count signal_timing road_capacity').split(),
}
BOUNDS = {'latitude': (-90, 90), 'longitude': (-180, 180), 'traffic_volume': (0, np.inf),
          'vehicle_count': (0, np.inf), 'avg_speed': (0, 200), 'occupancy': (0, 100),
          'travel_time': (0.001, np.inf), 'accident_count': (0, np.inf),
          'signal_timing': (0, np.inf), 'road_capacity': (1, np.inf),
          'temperature': (-60, 60), 'rainfall': (0, np.inf), 'visibility': (0, np.inf),
          'public_holiday': (0, 1), 'event_flag': (0, 1), 'roadwork_flag': (0, 1)}
INTEGER = {'traffic_volume', 'vehicle_count', 'accident_count', 'signal_timing',
           'road_capacity', 'public_holiday', 'event_flag', 'roadwork_flag'}
WEATHER = {'clear': 'Clear', 'cloudy': 'Cloudy', 'overcast': 'Cloudy',
           'rain': 'Rain', 'rainy': 'Rain', 'fog': 'Fog', 'foggy': 'Fog'}


def validate(frame, name, audit, persist_rejected=True):
    """Reject broken keys; mark repairable bad cells without dropping their rows."""
    missing = set(SCHEMAS[name]) - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')
    df = frame.copy()
    df['source_row'] = np.arange(2, len(df) + 2)
    fmt = '%d/%m/%Y' if name == 'weather_observations' else '%Y-%m-%d'
    text = df['date'].astype(str)
    if 'time' in df:
        text = text + ' ' + df['time'].astype(str)
        fmt += ' %H:%M'
    df['timestamp'] = pd.to_datetime(text, format=fmt, errors='coerce').dt.tz_localize(config()['timezone'])
    invalid = df.timestamp.isna()
    if 'road_id' in df:
        invalid |= ~df.road_id.astype(str).str.fullmatch(r'NL-0(?:0[1-9]|1[0-9]|2[0-5])')
        invalid |= df.road_name.isna()
        invalid |= ~df.weather_station_id.isin(['WS-NORTH', 'WS-CENTRAL', 'WS-SOUTH'])
        invalid |= ~df.timestamp.dt.minute.isin([0, 30])
    if 'station_id' in df:
        invalid |= ~df.station_id.isin(['WS-NORTH', 'WS-CENTRAL', 'WS-SOUTH'])
        invalid |= df.timestamp.dt.minute.ne(0)
    def record(mask, field, reason, action):
        if mask.any():
            audit.append(pd.DataFrame({'table': name, 'source_row': df.loc[mask, 'source_row'],
                                       'field': field, 'original_value': df.loc[mask, field].astype(str),
                                       'reason': reason, 'action': action}))
    record(invalid, 'date', 'Invalid key, required metadata or timestamp', 'quarantine row')
    for col, (low, high) in BOUNDS.items():
        if col not in df:
            continue
        raw = df[col].copy()
        num = pd.to_numeric(raw, errors='coerce')
        bad = num.notna() & (~np.isfinite(num) | (num < low) | (num > high))
        if col in INTEGER:
            bad |= num.notna() & num.mod(1).ne(0)
        bad |= raw.notna() & num.isna()
        record(bad, col, 'Invalid numeric type or physical range', 'set cell missing')
        df[col] = num.mask(bad)
        if col in ['road_capacity', 'latitude', 'longitude', 'signal_timing',
                   'public_holiday', 'event_flag', 'roadwork_flag']:
            invalid |= df[col].isna()
    if 'weather_condition' in df:
        normalized = df.weather_condition.astype(str).str.strip().str.lower().map(WEATHER)
        record(normalized.isna(), 'weather_condition', 'Unknown weather category', 'set cell missing')
        changed = normalized.notna() & df.weather_condition.ne(normalized)
        record(changed, 'weather_condition', 'Noncanonical valid weather label', 'normalize category')
        df['weather_condition'] = normalized
    if 'congestion_level' in df:
        bad = df.congestion_level.notna() & ~df.congestion_level.isin(CLASSES)
        record(bad, 'congestion_level', 'Unknown congestion category', 'derive from V/C')
        df.loc[bad, 'congestion_level'] = np.nan
        def parse(value):
            try:
                shares = json.loads(value)
                vals = [float(shares[k]) for k in ['2W', 'Car', 'LCV', 'HCV']]
                if any(x < 0 or x > 1 for x in vals) or abs(sum(vals) - 1) > .025:
                    return [np.nan] * 4
                return vals
            except (TypeError, ValueError, KeyError):
                return [np.nan] * 4
        shares = pd.DataFrame(df.vehicle_type_dist.map(parse).tolist(), index=df.index,
                              columns=['share_2w', 'share_car', 'share_lcv', 'share_hcv'])
        record(shares.isna().any(axis=1), 'vehicle_type_dist', 'Invalid JSON shares', 'quarantine row')
        invalid |= shares.isna().any(axis=1)
        df = pd.concat([df, shares], axis=1)
    if invalid.any():
        # Keep the complete rejected record as well as the cell audit.
        record(invalid, 'date', 'Unrecoverable required field', 'quarantine row')
        rejected = frame.loc[invalid].copy()
        rejected['source_row'] = df.loc[invalid, 'source_row']
        if persist_rejected:
            rejected.to_csv(ROOT / f'data/interim/{name}_rejected.csv', index=False)
    logging.info('%s: %s loaded, %s quarantined', name, len(df), invalid.sum())
    return df.loc[~invalid].copy()


def load(audit):
    """Load in dictionary order and record byte-level source lineage."""
    tables, manifest = {}, {}
    for name in SCHEMAS:
        path = ROOT / f'data/raw/{name}.csv'
        manifest[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        tables[name] = validate(pd.read_csv(path), name, audit)
    save_json(ROOT / 'data/processed/source_manifest.json', manifest)
    return tables
