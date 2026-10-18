"""Deterministic repair with separate historical and causal modelling values."""
import logging
import numpy as np
import pandas as pd
from src.ingest import load
from src.utils import ROOT, CLASSES, config, markdown_table, save_json, save_csv

SENSORS = ['traffic_volume', 'avg_speed', 'occupancy']


def deduplicate(df, keys, name, audit):
    """Keep the most complete record, then the earliest source row on ties."""
    df = df.assign(completeness=df.notna().sum(axis=1)).sort_values(
        keys + ['completeness', 'source_row'], ascending=[True]*len(keys) + [False, True])
    dup = df.duplicated(keys, keep='first')
    if dup.any():
        audit.append(pd.DataFrame({'table': name, 'source_row': df.loc[dup, 'source_row'],
                                   'field': '+'.join(keys), 'original_value': 'duplicate',
                                   'reason': 'Duplicate key; most complete record retained',
                                   'action': 'remove duplicate'}))
    logging.info('%s: %d duplicate records removed', name, dup.sum())
    return df.loc[~dup].drop(columns='completeness'), int(dup.sum())


def causal_fill(series, slots):
    """Fill short gaps from past readings and long gaps from past slot medians."""
    past_slot = series.groupby(slots).transform(lambda x: x.shift().expanding().median())
    past_road = series.shift().expanding().median()
    return series.ffill(limit=2).fillna(past_slot).fillna(past_road)


def congestion(volume, capacity):
    """Apply the dictionary's exact half-hour capacity thresholds."""
    ratio = volume / (capacity / 2)
    return pd.Series(np.select([ratio < .5, ratio < .8, ratio < 1],
                              CLASSES[:3], default=CLASSES[3]), index=volume.index).mask(ratio.isna())


def prepare(raw_tables=None, persist=True):
    """Preserve raw data and build the versioned, fully labelled analysis table."""
    cfg = config()
    audit = []
    if raw_tables is None:
        tables = load(audit)
    else:
        from src.ingest import validate
        tables = {name: validate(table, name, audit, persist_rejected=False) for name, table in raw_tables.items()}
    counts = {name: len(table) for name, table in tables.items()}
    weather_missing = tables['weather_observations'][['temperature', 'visibility', 'rainfall']].isna().sum().to_dict()
    blank_congestion = int(tables['traffic_sensor_log'].congestion_level.isna().sum())
    traffic, nd = deduplicate(tables['traffic_sensor_log'], ['road_id', 'timestamp'],
                             'traffic_sensor_log', audit)
    weather, nw = deduplicate(tables['weather_observations'], ['station_id', 'timestamp'],
                             'weather_observations', audit)
    calendar, nc = deduplicate(tables['calendar_events'], ['timestamp'], 'calendar_events', audit)
    # Statistical extremes are flagged, not clipped: rush-hour peaks may be real.
    train_end = pd.Timestamp(cfg['train_end'], tz=cfg['timezone'])
    extreme_counts = {}
    for col in SENSORS:
        train = traffic.loc[traffic.timestamp <= train_end, col]
        q1, q3 = train.quantile([.25, .75])
        spread = q3 - q1
        extreme = ((traffic[col] < q1 - 3*spread) | (traffic[col] > q3 + 3*spread)
                   | ((traffic[col] - train.mean()).abs() > 4*train.std()))
        traffic[f'{col}_extreme'] = extreme
        extreme_counts[col] = int(extreme.sum())
    all_times = pd.date_range(traffic.timestamp.min(), traffic.timestamp.max(), freq='30min')
    blocks = []
    missing_counts = {}
    static = ['road_name', 'latitude', 'longitude', 'weather_station_id', 'road_capacity']
    for road, block in traffic.groupby('road_id', sort=True):
        block = block.set_index('timestamp').reindex(all_times)
        block.index.name = 'timestamp'
        block['window_observed'] = block.source_row.notna()
        block['road_id'] = road
        for col in static:
            observed = block[col].dropna()
            if observed.nunique() != 1:
                raise ValueError(f'{road}: inconsistent static metadata in {col}')
            block[col] = observed.iloc[0]
        slots = pd.Series(block.index.hour*2 + block.index.minute//30, index=block.index)
        for col in SENSORS:
            block[f'{col}_observed'] = block[col].notna()
            missing_counts[col] = missing_counts.get(col, 0) + int(block[col].isna().sum())
            block[f'causal_{col}'] = causal_fill(block[col], slots)
            # Retrospective interpolation is for EDA only. It never enters features/labels.
            missing = block[col].isna()
            gap_size = missing.groupby((~missing).cumsum()).transform('sum')
            interpolated = block[col].interpolate(method='time', limit_area='inside')
            history = block[col].where(~(missing & (gap_size <= 2)), interpolated)
            block[col] = history.fillna(block[f'causal_{col}'])
        for col in ['vehicle_count', 'signal_timing', 'share_2w', 'share_car', 'share_lcv', 'share_hcv']:
            block[f'causal_{col}'] = causal_fill(block[col], slots)
        block['congestion_observed'] = block.congestion_level.notna() | block.traffic_volume_observed
        missing = block.congestion_level.isna()
        block.loc[missing, 'congestion_level'] = congestion(block.traffic_volume, block.road_capacity)[missing]
        # Missing incidents and travel-time labels remain missing: no fabricated outcomes.
        blocks.append(block.reset_index())
    df = pd.concat(blocks, ignore_index=True)
    absent = int((~df.window_observed).sum())
    logging.info('Regular grid: %d missing windows explicitly added and flagged', absent)
    traffic = df.copy()
    weather_blocks = []
    for station, block in weather.groupby('station_id', sort=True):
        block = block.sort_values('timestamp').copy()
        for col in ['temperature', 'visibility', 'rainfall']:
            block[f'{col}_missing'] = block[col].isna()
            block[col] = block[col].ffill(limit=2).fillna(block[col].shift().expanding().median())
        block['weather_condition'] = block.weather_condition.ffill()
        weather_blocks.append(block)
    weather = pd.concat(weather_blocks, ignore_index=True)
    for name, frame in [('traffic_clean', traffic), ('weather_clean', weather), ('calendar_clean', calendar)]:
        if persist:
            frame.to_csv(ROOT / f'data/interim/{name}.csv.gz', index=False)
    df['weather_hour'] = df.timestamp.dt.floor('h')
    cols = ['station_id', 'timestamp', 'weather_condition', 'temperature', 'rainfall', 'visibility']
    df = df.merge(weather[cols].rename(columns={'timestamp': 'weather_hour'}),
                  left_on=['weather_station_id', 'weather_hour'], right_on=['station_id', 'weather_hour'],
                  how='left', validate='many_to_one')
    df['date_key'] = df.timestamp.dt.normalize()
    df = df.merge(calendar.drop(columns=['date', 'source_row']).rename(columns={'timestamp': 'date_key'}),
                  on='date_key', how='left', validate='many_to_one')
    df['date'] = df.timestamp.dt.strftime('%Y-%m-%d')
    df['time'] = df.timestamp.dt.strftime('%H:%M')
    df['day_of_week'] = df.timestamp.dt.day_name().str[:3]
    df = df.drop(columns=['weather_station_id', 'station_id', 'weather_hour', 'date_key'])
    assert not df.duplicated(['road_id', 'timestamp']).any()
    assert len(df) == len(all_times)*df.road_id.nunique()
    for col in ['road_capacity', 'temperature', 'visibility', 'rainfall', 'public_holiday']:
        if df[col].isna().any():
            raise ValueError(f'Unresolved merged values: {col}')
    if not persist:
        return df, (pd.concat(audit, ignore_index=True) if audit else pd.DataFrame())
    save_csv(df, ROOT / f"data/processed/flowcast_{cfg['version']}.csv.gz")
    # Every repair and removal has an attributable source row or grid key.
    issues = pd.concat(audit, ignore_index=True) if audit else pd.DataFrame(
        columns=['table', 'source_row', 'field', 'original_value', 'reason', 'action'])
    issues.to_csv(ROOT / 'data/interim/validation_audit.csv', index=False)
    gaps = df.loc[~df.window_observed, ['road_id', 'timestamp']]
    gaps.to_csv(ROOT / 'data/interim/missing_windows.csv', index=False)
    stats = {'source_rows': counts, 'duplicates': nd, 'weather_duplicates': nw,
             'calendar_duplicates': nc, 'absent_windows': absent, 'processed_rows': len(df),
             'missing_sensor_cells_after_validation': missing_counts,
             'statistical_extremes_retained': extreme_counts, 'audit_records': len(issues),
             'weather_missing_cells': weather_missing, 'blank_congestion_source_rows': blank_congestion}
    save_json(ROOT / 'reports/data_quality.json', stats)
    summary = pd.DataFrame({'Measure': ['Traffic input rows', 'Traffic duplicates', 'Absent windows',
                                       'Processed rows', 'Roads', 'Validation audit entries'],
                            'Count': [counts['traffic_sensor_log'], nd, absent, len(df), 25, len(issues)]})
    text = '# FlowCast data quality report\n\n' + markdown_table(summary) + '''

## Source preservation and validation

The three raw CSV files are byte-for-byte copies of the supplied files. Their SHA-256 hashes
are in `data/processed/source_manifest.json`. The original Word files remain in their source
folder and are unchanged. Dates use Asia/Kolkata, confirmed by the user. Traffic and calendar
dates use YYYY-MM-DD. Weather dates use DD/MM/YYYY.

The pipeline checks columns, keys, JSON vehicle shares, categories, numeric types and physical
ranges. Invalid cells become missing and are listed by original CSV row in
`data/interim/validation_audit.csv`. Unrecoverable rows are quarantined with their full contents.
Duplicate keys keep the most complete record, with original row order as the tie breaker.

## Repairs and their limits

The regular road-by-time grid makes absent windows visible. These are listed in
`data/interim/missing_windows.csv`. Historical sensor values use time interpolation for up to
two missing positions, followed by past segment-and-time-slot medians and past segment medians.
Remaining leading values are left missing. These rows cannot enter a model until its features
are available. Imputed historical values are clearly flagged and are not scored as real targets.

Modelling uses separate `causal_` columns. Short gaps carry the latest reading for at most two
windows. Longer gaps use only earlier readings from the same road and time slot, then the
road's earlier median. Future interpolation values never enter prediction features. Accident
and travel-time outcomes are never imputed. Training and evaluation require observed targets.

Statistical outliers use training-period IQR and z-score rules (3 IQR or 4 standard deviations).
They are flagged and retained because valid traffic peaks should not be removed. Physical
outliers are marked missing before imputation. Vehicle shares are parsed from the original JSON.

Weather labels map to Clear, Cloudy, Rain and Fog. Overcast maps to Cloudy. Missing weather
values use only earlier observations at that station. Each hourly observation joins both
half-hour windows, exactly as the dictionary requires. Rainfall stays an hourly measurement;
it is not divided or described as an observed half-hour total. This assumes the supplied
hourly weather observation is available at that hour. No future weather observations are used.
Calendar flags join by local date. Missing calendar coverage blocks a forecast rather than
inventing holidays or events. Station IDs are dropped after merging.

Missing congestion labels use volume / (capacity / 2), with thresholds 0.50, 0.80 and 1.00.
Provided valid congestion labels are retained. Rows whose labels depend on imputed volume are
excluded from classification scoring. Descriptive history may include flagged repaired values.

## Counts by defect

'''
    text += markdown_table(pd.DataFrame(missing_counts.items(), columns=['Sensor field', 'Missing cells including grid gaps']))
    text += '\n\n' + markdown_table(pd.DataFrame(extreme_counts.items(), columns=['Sensor field', 'Statistical extremes retained']))
    text += '\n\n' + markdown_table(pd.DataFrame(weather_missing.items(), columns=['Weather field', 'Missing source cells']))
    text += f'\n\nBlank congestion labels in the source before deduplication: {blank_congestion:,}.\n'
    text += '\n\n' + markdown_table(issues.groupby(['field', 'reason', 'action']).size().reset_index(name='Rows')) + '\n'
    (ROOT / 'reports/data_quality.md').write_text(text)
    logging.info('Cleaning and merge complete: %d rows', len(df))
    return df
