# FlowCast data quality report

| Measure | Count |
| --- | --- |
| Traffic input rows | 178468 |
| Traffic duplicates | 1767 |
| Absent windows | 4499 |
| Processed rows | 181200 |
| Roads | 25 |
| Validation audit entries | 5834 |

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

| Sensor field | Missing cells including grid gaps |
| --- | --- |
| traffic_volume | 9086 |
| avg_speed | 9078 |
| occupancy | 9072 |

| Sensor field | Statistical extremes retained |
| --- | --- |
| traffic_volume | 124 |
| avg_speed | 0 |
| occupancy | 0 |

| Weather field | Missing source cells |
| --- | --- |
| temperature | 167 |
| visibility | 111 |
| rainfall | 0 |

Blank congestion labels in the source before deduplication: 26,883.


| field | reason | action | Rows |
| --- | --- | --- | --- |
| avg_speed | Invalid numeric type or physical range | set cell missing | 237 |
| occupancy | Invalid numeric type or physical range | set cell missing | 234 |
| road_id+timestamp | Duplicate key; most complete record retained | remove duplicate | 1767 |
| traffic_volume | Invalid numeric type or physical range | set cell missing | 241 |
| weather_condition | Noncanonical valid weather label | normalize category | 3355 |
