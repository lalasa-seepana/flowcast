"""FlowCast: nine operational views backed only by saved real model outputs."""
from pathlib import Path
import sys
import json
import subprocess
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils import CLASSES, COLORS, config, markdown_table
from src.features import load_dataset

st.set_page_config(page_title='FlowCast | Northline Corridor', page_icon='🚦', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.03em;}
[data-testid="stMetric"] {background: #f1f6f8; padding: 16px; border-radius: 8px; color: #17394c;}
</style>''', unsafe_allow_html=True)

VIEWS = ['Live prediction', 'Historical trends', 'Congestion heatmap', 'Road comparison',
         'Model performance', 'Feature importance', 'Forecast visualisation',
         'Prediction confidence', 'Weather vs traffic']


@st.cache_data(show_spinner=False)
def data(stamp):
    """Invalidate cached data when the persisted dataset changes."""
    return load_dataset()


@st.cache_data(show_spinner=False)
def csv(path, stamp):
    """Read a changed artifact only once per dashboard session."""
    return pd.read_csv(path)


def read_csv(path):
    """Read a CSV with modification-aware caching."""
    return csv(str(path), path.stat().st_mtime_ns)


def chart(figure):
    """Apply the common legible chart style."""
    figure.update_layout(template='plotly_white', font=dict(family='Arial', color='#23394d'),
                         margin=dict(l=20, r=20, t=55, b=30), legend_title_text='')
    st.plotly_chart(figure, use_container_width=True)


def selected_outputs(task='flow'):
    """Load predictions from the validation-selected artifact."""
    names = json.loads((ROOT / 'models/selected_models.json').read_text())
    path = ROOT / 'data/processed' / names[task].replace('.joblib', '_test.csv.gz')
    out = read_csv(path)
    out['target_timestamp'] = pd.to_datetime(out.target_timestamp, utc=True).dt.tz_convert(config()['timezone'])
    return out


def confidence_chart(out, target, title):
    """Display observed targets, forecasts and empirical prediction intervals."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out.target_timestamp, y=out[f'upper_{target}'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=out.target_timestamp, y=out[f'lower_{target}'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(22,107,140,.17)', name='Nominal 90% interval'))
    fig.add_trace(go.Scatter(x=out.target_timestamp, y=out[f'pred_{target}'], name='Forecast', line=dict(color='#166b8c')))
    if f'y_{target}' in out:
        fig.add_trace(go.Scatter(x=out.target_timestamp, y=out[f'y_{target}'], name='Observed', line=dict(color='#de8957')))
    unit = {'traffic_volume': 'Vehicles / 30 min', 'avg_speed': 'km/h', 'travel_time': 'Minutes'}[target]
    fig.update_layout(title=title, xaxis_title='Target time (IST)', yaxis_title=unit)
    chart(fig)


cfg = config()
path = ROOT / f'data/processed/flowcast_{cfg["version"]}.csv.gz'
if not path.exists() or not (ROOT / 'models/selected_models.json').exists():
    st.error('Run the pipeline first: python -m src.utils')
    st.stop()
df = data(path.stat().st_mtime_ns)
st.sidebar.title('FlowCast')
st.sidebar.caption('Northline Corridor · 25 road segments')
role = st.sidebar.selectbox('Your role', ['Traffic Operations Analyst', 'Transport Planner', 'Incident Response Coordinator', 'System Owner / Reviewer'])
defaults = {'Traffic Operations Analyst': 'Congestion heatmap', 'Transport Planner': 'Road comparison',
            'Incident Response Coordinator': 'Live prediction', 'System Owner / Reviewer': 'Model performance'}
if st.session_state.get('previous_role') != role:
    st.session_state['view'] = defaults[role]
    st.session_state['previous_role'] = role
view = st.sidebar.radio('View', VIEWS, key='view')
road_names = df[['road_id', 'road_name']].drop_duplicates().set_index('road_id').road_name.to_dict()
road = st.sidebar.selectbox('Road segment', sorted(road_names), format_func=lambda r: f'{r} · {road_names[r]}')
horizon = st.sidebar.select_slider('Forecast horizon (minutes)', options=[30, 60, 90, 120], value=30)
dates = st.sidebar.date_input('Historical date range', value=(df.timestamp.max().date()-pd.Timedelta(days=6), df.timestamp.max().date()),
                              min_value=df.timestamp.min().date(), max_value=df.timestamp.max().date())
if len(dates) != 2:
    st.info('Select a start and end date.')
    st.stop()
start, end = dates
history = df.loc[(df.timestamp.dt.date >= start) & (df.timestamp.dt.date <= end)]
st.title('FlowCast')
st.caption('Traffic forecasts and corridor analytics')
st.info('Historical replay · Data: 1 January–31 May 2025 · Times: India Standard Time. No live sensor feed is connected.')
st.subheader(view)

if view == 'Live prediction':
    forecast = st.session_state.get('requested_forecast', read_csv(ROOT / 'data/processed/latest_forecasts.csv'))
    shown = forecast.loc[forecast.horizon.eq(horizon//30)].copy()
    st.caption(f"Last completed window starts: {shown.timestamp.iloc[0]} · Horizon: {horizon} minutes")
    issued = pd.Timestamp(shown.timestamp.iloc[0]) + pd.Timedelta(minutes=30)
    st.caption(f'Issued after the observed window ends: {issued}. Horizon is measured to the end of the forecast window.')
    metrics = st.columns(3)
    metrics[0].metric('Roads forecast', len(shown))
    metrics[1].metric('Corridor traversal estimate', f'{shown.travel_time.sum():.1f} min')
    metrics[2].metric('Highest accident probability', f'{shown.accident_risk.max():.1%}')
    st.caption(f'Sum of segment bounds: {shown.travel_time_lower.sum():.1f}–{shown.travel_time_upper.sum():.1f} min. Assumes one traversal of each segment. This is not a calibrated corridor interval.')
    columns = ['road_id', 'traffic_volume', 'traffic_volume_lower', 'traffic_volume_upper', 'avg_speed',
               'avg_speed_lower', 'avg_speed_upper', 'congestion_level', 'congestion_confidence',
               'travel_time', 'travel_time_lower', 'travel_time_upper', 'accident_risk', 'risk_confidence']
    display_table = shown[columns].sort_values('accident_risk', ascending=False).copy()
    for probability in ['congestion_confidence', 'accident_risk', 'risk_confidence']:
        display_table[probability] *= 100
    labels = {
        'road_id': 'Road', 'traffic_volume': 'Volume (veh/30 min)',
        'traffic_volume_lower': 'Volume lower', 'traffic_volume_upper': 'Volume upper',
        'avg_speed': 'Speed (km/h)', 'avg_speed_lower': 'Speed lower',
        'avg_speed_upper': 'Speed upper', 'congestion_level': 'Congestion',
        'congestion_confidence': 'Congestion confidence (%)',
        'travel_time': 'Travel time (min)', 'travel_time_lower': 'Travel lower',
        'travel_time_upper': 'Travel upper', 'accident_risk': 'Accident risk (%)',
        'risk_confidence': 'Risk class confidence (%)'}
    display_table = display_table.rename(columns=labels).round(2)
    def severity_style(value):
        color = dict(zip(CLASSES, COLORS)).get(value, '#ffffff')
        text_color = '#ffffff' if value in ['Free-flow', 'Severe'] else '#202020'
        return f'background-color: {color}; color: {text_color}'
    st.dataframe(display_table.style.map(severity_style, subset=['Congestion']),
                 hide_index=True, use_container_width=True)
    st.caption('Volume: vehicles per 30 minutes. Speed: km/h. Travel time: minutes. Regression bounds: nominal 90% prediction intervals. Class confidence: model probability; risk confidence: max(p, 1−p).')
    risk_artifact = shown.accident_model.iloc[0]
    risk_card = json.loads((ROOT / 'models' / risk_artifact.replace('.joblib', '_card.json')).read_text())
    risk_metrics = next(row for row in risk_card['test_metrics'] if row['horizon'] == horizon//30)
    st.caption(f"Held-out risk-model ROC-AUC: {risk_metrics['roc_auc']:.3f}. Incident recall at a 0.5 threshold: {risk_metrics['recall']:.1%}. Use these measured limits when reading the risk ranking.")
    st.download_button('Download forecasts with model lineage', forecast.to_csv(index=False), 'flowcast_forecasts.csv', 'text/csv')

elif view == 'Historical trends':
    sample = history.loc[history.road_id.eq(road)]
    for target, unit in [('traffic_volume', 'Vehicles / 30 min'), ('avg_speed', 'km/h')]:
        observed = sample.loc[sample[f'{target}_observed']]
        chart(px.line(observed, x='timestamp', y=target, title=f'{road} observed {target.replace("_", " ")}', labels={target: unit, 'timestamp': 'Window start (IST)'}))
    hours = sample.loc[sample.traffic_volume_observed].assign(hour=lambda x: x.timestamp.dt.hour).groupby('hour', as_index=False).traffic_volume.mean()
    chart(px.bar(hours, x='hour', y='traffic_volume', title='Mean volume by hour', labels={'hour': 'Hour of day (IST)', 'traffic_volume': 'Mean vehicles / 30 min'}))

elif view == 'Congestion heatmap':
    day = st.selectbox('Day', sorted(history.timestamp.dt.date.unique()), index=len(history.timestamp.dt.date.unique())-1)
    grid = history.loc[history.timestamp.dt.date.eq(day)].copy()
    grid['clock'] = grid.timestamp.dt.strftime('%H:%M')
    grid['severity'] = grid.congestion_level.map(dict(zip(CLASSES, range(4))))
    pivot = grid.pivot(index='road_id', columns='clock', values='severity')
    scale = [[0, COLORS[0]], [.2499, COLORS[0]], [.25, COLORS[1]], [.4999, COLORS[1]], [.5, COLORS[2]], [.7499, COLORS[2]], [.75, COLORS[3]], [1, COLORS[3]]]
    chart(go.Figure(go.Heatmap(z=pivot.to_numpy(), x=pivot.columns, y=pivot.index, zmin=-.5, zmax=3.5,
                              colorscale=scale, colorbar=dict(tickvals=list(range(4)), ticktext=CLASSES))).update_layout(height=650, xaxis_title='Window start (IST)', yaxis_title='Road segment'))
    st.caption('Includes dictionary-derived labels and clearly flagged repaired historical windows. The source observation flags are available in the processed dataset.')

elif view == 'Road comparison':
    roads = st.multiselect('Compare roads', sorted(road_names), default=sorted(road_names)[:5])
    compare = history.loc[history.road_id.isin(roads)].copy()
    for col in ['traffic_volume', 'avg_speed']:
        compare.loc[~compare[f'{col}_observed'], col] = np.nan
    stats = compare.groupby('road_id').agg(volume=('traffic_volume', 'mean'), speed=('avg_speed', 'mean'),
                                           mean_time=('travel_time', 'mean'), p95_time=('travel_time', lambda x: x.quantile(.95))).reset_index()
    stats['reliability_ratio'] = stats.p95_time/stats.mean_time
    for col, label in [('volume', 'Mean vehicles / 30 min'), ('speed', 'Mean speed (km/h)'), ('reliability_ratio', 'P95 / mean travel time (lower is steadier)')]:
        chart(px.bar(stats, x='road_id', y=col, labels={col: label, 'road_id': 'Road segment'}, title=label))

elif view == 'Model performance':
    score = read_csv(ROOT / 'reports/model_comparison.csv')
    st.caption('Fixed held-out test period. The sidebar historical dates do not change model scores. Deep models cover the 30-minute horizon only.')
    st.dataframe(score.loc[score.horizon.eq(horizon//30)].round(4), hide_index=True, use_container_width=True)
    st.image(str(ROOT / 'reports/figures/model_benchmark.png'))
    st.image(str(ROOT / 'reports/figures/congestion_confusion_matrix.png'))
    st.image(str(ROOT / 'reports/figures/accident_roc.png'))
    st.image(str(ROOT / 'reports/figures/dl_learning_curves.png'))
    st.dataframe(read_csv(ROOT / 'reports/model_selection.csv'), hide_index=True)

elif view == 'Feature importance':
    importance = read_csv(ROOT / 'reports/feature_importance.csv').head(20).sort_values('importance')
    chart(px.bar(importance, x='importance', y='feature', orientation='h', title='XGBoost volume and speed feature importance'))
    st.caption('Normalized tree importance describes this fitted model. It does not measure causal effects.')

elif view in ['Forecast visualisation', 'Prediction confidence']:
    target = st.selectbox('Forecast target', ['traffic_volume', 'avg_speed', 'travel_time'])
    task = {'traffic_volume': 'flow', 'avg_speed': 'speed', 'travel_time': 'travel_time'}[target]
    out = selected_outputs(task)
    out = out.loc[out.road_id.eq(road) & out.horizon.eq(horizon//30)]
    selected = out.loc[(out.target_timestamp.dt.date >= start) & (out.target_timestamp.dt.date <= end)]
    if selected.empty:
        st.warning('The selected dates have no held-out forecasts. Choose 9–31 May 2025.')
    else:
        confidence_chart(selected, target, f'{road} · {horizon}-minute {target.replace("_", " ")} forecast')
    if view == 'Prediction confidence':
        st.caption('Nominal 90% empirical prediction intervals, calibrated on validation residuals. Coverage can change over time.')
        st.dataframe(read_csv(ROOT / 'reports/interval_coverage.csv'), hide_index=True)
        st.dataframe(read_csv(ROOT / 'reports/classification_calibration.csv'), hide_index=True)
        st.caption('Congestion calibration compares top-class probability with accuracy. Accident calibration compares risk probability with incident frequency.')
    else:
        benchmark = st.selectbox('Historical sequence comparison', ['GRU', 'BiLSTM'])
        deep = read_csv(ROOT / f'data/processed/{benchmark.lower()}_{cfg["version"]}_test.csv.gz')
        deep['target_timestamp'] = pd.to_datetime(deep.target_timestamp, utc=True).dt.tz_convert(cfg['timezone'])
        deep = deep.loc[deep.road_id.eq(road) & (deep.target_timestamp.dt.date >= start) & (deep.target_timestamp.dt.date <= end)]
        if len(deep):
            confidence_chart(deep, 'traffic_volume', f'{benchmark} historical comparison · 30 minutes')
        st.caption('Both recurrent models use past windows only. The bidirectional LSTM is a historical benchmark, not the operational predictor.')

elif view == 'Weather vs traffic':
    observed = history.loc[history.traffic_volume_observed]
    weather = observed.groupby('weather_condition', as_index=False).agg(volume=('traffic_volume', 'mean'), windows=('traffic_volume', 'size'), speed=('avg_speed', 'mean'))
    chart(px.bar(weather, x='weather_condition', y='volume', hover_data=['windows', 'speed'], title='Observed volume by weather', labels={'volume': 'Mean vehicles / 30 min', 'weather_condition': 'Weather'}))
    st.image(str(ROOT / 'reports/figures/weather_vs_traffic.png'))
    st.image(str(ROOT / 'reports/figures/weather_observations.png'))
    st.caption('These are observed associations, not causal weather effects. Forecasts use origin-time weather because future forecasts were not supplied.')

st.divider()
with st.expander('Request a forecast'):
    st.caption('Choose a historical origin with enough past readings. All four horizons must stay within the supplied calendar dates.')
    origin_day = st.date_input('Last observed window date', df.timestamp.max().date(), min_value=df.timestamp.min().date(), max_value=df.timestamp.max().date())
    time_slot = st.selectbox('Last observed window start (IST)', [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30]], index=43)
    if st.button('Run prediction'):
        from src.ml_models import forecast_latest
        origin = pd.Timestamp(f'{origin_day} {time_slot}', tz=cfg['timezone'])
        if origin + pd.Timedelta(hours=2) > df.timestamp.max():
            st.error('All four horizons need calendar coverage. Choose an origin at or before 21:30 on the final date.')
        else:
            with st.spinner('Loading saved models and running inference...'):
                try:
                    forecast = forecast_latest(df, origin=origin)
                    st.session_state['requested_forecast'] = forecast
                    st.dataframe(forecast.loc[forecast.road_id.eq(road)], hide_index=True)
                    st.success('Prediction complete. Open Live prediction for the corridor table.')
                except ValueError as exc:
                    st.error(str(exc))

with st.expander('Upload sensor data'):
    st.caption('Upload a CSV with the traffic_sensor_log schema. Data is validated and cleaned in memory. Original files stay unchanged. Weather and calendar coverage must be available for the uploaded dates.')
    upload = st.file_uploader('Sensor CSV', type=['csv'])
    if upload is not None and st.button('Validate upload and predict'):
        from src.ingest import SCHEMAS, validate
        from src.clean import prepare
        from src.ml_models import forecast_latest
        try:
            incoming = pd.read_csv(upload)
            raw = {name: pd.read_csv(ROOT / f'data/raw/{name}.csv') for name in SCHEMAS}
            required = set(SCHEMAS['traffic_sensor_log']) - set(incoming.columns)
            if required:
                raise ValueError(f'Missing columns: {sorted(required)}')
            upload_audit = []
            valid_upload = validate(incoming, 'traffic_sensor_log', upload_audit, persist_rejected=False)
            if valid_upload.empty:
                raise ValueError('No uploaded rows passed key and required-field validation.')
            parsed_dates = pd.to_datetime(incoming.date, format='%Y-%m-%d', errors='coerce')
            if parsed_dates.dropna().empty or parsed_dates.min().date() < df.timestamp.min().date() or parsed_dates.max().date() > df.timestamp.max().date():
                raise ValueError('Upload dates must be inside the supplied weather and calendar coverage (1 January–31 May 2025).')
            # Uploaded rows take priority on equal-completeness key ties.
            raw['traffic_sensor_log'] = pd.concat([incoming, raw['traffic_sensor_log']], ignore_index=True)
            with st.spinner('Validating, cleaning and forecasting the upload...'):
                cleaned, audit = prepare(raw_tables=raw, persist=False)
                new_origin = min(valid_upload.timestamp.max(), df.timestamp.max()-pd.Timedelta(hours=2))
                if not (valid_upload.timestamp <= new_origin).any():
                    raise ValueError('Uploaded readings fall after the latest origin with complete calendar coverage.')
                result = forecast_latest(cleaned, origin=new_origin)
                result['upload_sha256'] = hashlib.sha256(upload.getvalue()).hexdigest()
            st.session_state['requested_forecast'] = result
            st.success(f'{len(valid_upload):,} of {len(incoming):,} uploaded rows passed required-field validation. Replay origin: {new_origin}.')
            if upload_audit:
                st.dataframe(pd.concat(upload_audit, ignore_index=True), hide_index=True)
            st.caption('The full merge audit below numbers rows in the upload-first combined sensor table.')
            st.dataframe(audit.head(100), hide_index=True)
            st.download_button('Download complete upload validation audit', audit.to_csv(index=False), 'upload_audit.csv', 'text/csv')
            st.download_button('Download upload predictions', result.to_csv(index=False), 'upload_predictions.csv', 'text/csv')
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f'Upload could not be used: {exc}')

with st.expander('Reports and insights'):
    st.caption('Export observed historical summaries for the sidebar date range. Missing sensor values do not enter these aggregates.')
    summary = history.loc[history.traffic_volume_observed].groupby('road_id').agg(observed_windows=('traffic_volume', 'size'), mean_volume=('traffic_volume', 'mean')).reset_index()
    summary_text = f'# FlowCast selected-period insights\n\nPeriod: {start} to {end}, Asia/Kolkata.\n\n' + markdown_table(summary)
    summary_text += '\n\nThese are observed historical aggregates. They are not new forecasts. See the final technical report for model scores and uncertainty.\n'
    st.download_button('Export selected-period report', summary_text, f'flowcast_{start}_{end}.md', 'text/markdown')
    st.download_button('Download final technical report', (ROOT / 'reports/final_report.md').read_text(), 'flowcast_technical_report.md', 'text/markdown')

with st.expander('Model training'):
    st.caption('Training is an explicit background pipeline run. Ordinary dashboard views only read saved results. The run uses the original CSV inputs and can take several minutes.')
    process = st.session_state.get('training_process')
    running = process is not None and process.poll() is None
    if st.button('Retrain from original inputs', disabled=running):
        log = (ROOT / 'reports/retraining.log').open('w')
        process = subprocess.Popen([sys.executable, '-m', 'src.utils'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        st.session_state['training_process'] = process
        st.info('Training started. Use Check training status to refresh progress.')
    if st.button('Check training status'):
        process = st.session_state.get('training_process')
        if process is None:
            st.info('No training run has been started in this session.')
        elif process.poll() is None:
            st.info('Training is running.')
        elif process.returncode == 0:
            st.cache_data.clear()
            st.success('Training completed. Reload a view to read the updated results.')
            st.dataframe(pd.read_csv(ROOT / 'reports/model_comparison.csv'), hide_index=True)
        else:
            st.error('Training failed. Review the log below.')
        log_path = ROOT / 'reports/retraining.log'
        if log_path.exists():
            st.code('\n'.join(log_path.read_text().splitlines()[-14:]))
