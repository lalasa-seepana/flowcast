"""Honest hold-out metrics, plots and the final technical report."""
import numpy as np
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                             precision_recall_fscore_support, accuracy_score, roc_auc_score,
                             f1_score, brier_score_loss)


def regression_metrics(actual, predicted):
    """MAPE excludes true zero targets and reports their number explicitly."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    nonzero = actual != 0
    return {'rmse': float(root_mean_squared_error(actual, predicted)),
            'mae': float(mean_absolute_error(actual, predicted)),
            'mape': float(np.mean(np.abs((actual[nonzero]-predicted[nonzero])/actual[nonzero]))*100),
            'r2': float(r2_score(actual, predicted)), 'zero_targets': int((~nonzero).sum())}


def classification_metrics(actual, probabilities, task):
    """Class metrics use 0.5 for risk and argmax for four-class congestion."""
    actual = np.asarray(actual, dtype=int)
    predicted = probabilities.argmax(1)
    average = 'macro' if task == 'congestion' else 'binary'
    labels = list(range(4)) if task == 'congestion' else [0, 1]
    p, r, f, _ = precision_recall_fscore_support(actual, predicted, average=average,
                                               labels=labels, zero_division=0)
    result = {'precision': float(p), 'recall': float(r), 'f1': float(f),
              'macro_f1': float(f1_score(actual, predicted, average='macro', labels=labels, zero_division=0)),
              'accuracy': float(accuracy_score(actual, predicted))}
    if task == 'accident':
        result['roc_auc'] = float(roc_auc_score(actual, probabilities[:, 1])) if len(np.unique(actual)) == 2 else None
        result['brier'] = float(brier_score_loss(actual, probabilities[:, 1]))
    return result


def report(df, cfg):
    """Generate every chart and claim from persisted data and model outputs."""
    import json
    import logging
    import time
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.metrics import confusion_matrix, roc_curve
    from src.utils import ROOT, CLASSES, COLORS, markdown_table, save_json
    from src.features import supervised
    from src.ml_models import predict_bundle, forecast_latest
    import joblib
    sns.set_theme(style='whitegrid', context='notebook', palette=['#166b8c', '#de8957', '#2c9c95', '#b95778'])
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 170, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'font.family': 'DejaVu Sans'})
    figures = ROOT / 'reports/figures'
    def save(fig, name):
        fig.savefig(figures / f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
    ml = pd.read_csv(ROOT / 'reports/ml_scoreboard.csv')
    dl = pd.read_csv(ROOT / 'reports/dl_scoreboard.csv')
    scoreboard = pd.concat([ml, dl], ignore_index=True)
    scoreboard.to_csv(ROOT / 'reports/model_comparison.csv', index=False)
    for card_path in (ROOT / 'models').glob(f'flow_*_{cfg["version"]}_card.json'):
        card = json.loads(card_path.read_text())
        persisted = joblib.load(ROOT / 'models' / card['artifact'])
        card['travel_features'] = ['forecast_volume', 'forecast_speed', 'road_capacity', 'training_segment_length_km', 'physical_travel_minutes']
        card['travel_hyperparameters'] = persisted['travel_model'].get_params()
        save_json(card_path, card)
    selected = json.loads((ROOT / 'models/selected_models.json').read_text())
    outputs = {}
    for task, artifact in selected.items():
        path = ROOT / f'data/processed/{artifact.replace(".joblib", "_test.csv.gz")}'
        out = pd.read_csv(path)
        for col in ['timestamp', 'target_timestamp']:
            out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert(cfg['timezone'])
        outputs[task] = out
    flow = outputs['flow']
    h1 = flow.loc[flow.horizon.eq(1)]
    observed = df.loc[df.window_observed].copy()
    numeric = ['traffic_volume', 'avg_speed', 'occupancy', 'travel_time', 'rainfall', 'visibility', 'temperature', 'road_capacity']
    actual = observed.copy()
    for col in ['traffic_volume', 'avg_speed', 'occupancy']:
        actual.loc[~actual[f'{col}_observed'], col] = np.nan
    desc = actual[numeric].describe().T
    desc['median'] = actual[numeric].median()
    desc['mode'] = actual[numeric].mode().iloc[0]
    desc['variance'] = actual[numeric].var()
    desc.to_csv(ROOT / 'reports/descriptive_statistics.csv')
    actual[numeric].cov().to_csv(ROOT / 'reports/covariance.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, col, unit in zip(axes.flat, numeric[:4], ['vehicles / 30 min', 'km/h', '%', 'minutes']):
        sns.histplot(actual[col].dropna(), bins=45, ax=ax, color='#166b8c')
        ax.set(xlabel=f'{col.replace("_", " ").title()} ({unit})', ylabel='Observed windows')
    fig.suptitle('Northline Corridor observed traffic distributions', fontsize=17)
    save(fig, 'eda_distributions')
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    sns.heatmap(actual[numeric].corr(), annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax,
                xticklabels=[c.replace('_', ' ') for c in numeric], yticklabels=[c.replace('_', ' ') for c in numeric])
    ax.set_title('Observed feature correlations')
    save(fig, 'eda_correlations')
    hourly = actual.assign(hour=actual.timestamp.dt.hour).groupby('hour')[['traffic_volume', 'avg_speed']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, col, label in zip(axes, hourly.columns, ['Vehicles / 30 min', 'Speed (km/h)']):
        ax.plot(hourly.index, hourly[col], marker='o', color='#166b8c')
        ax.set(xlabel='Hour of day (IST)', ylabel=label, title=f'Mean {col.replace("_", " ")}')
    save(fig, 'peak_hour_patterns')
    last_day = df.timestamp.max().normalize()
    grid = df.loc[df.timestamp.dt.normalize().eq(last_day)].copy()
    grid['severity'] = grid.congestion_level.map(dict(zip(CLASSES, range(4))))
    grid['clock'] = grid.timestamp.dt.strftime('%H:%M')
    fig, ax = plt.subplots(figsize=(13, 7), layout='constrained')
    from matplotlib.colors import ListedColormap, BoundaryNorm
    hm = grid.pivot(index='road_id', columns='clock', values='severity')
    sns.heatmap(hm, cmap=ListedColormap(COLORS), vmin=-.5, vmax=3.5, ax=ax,
                cbar_kws={'ticks': [0, 1, 2, 3]}, xticklabels=4)
    ax.collections[0].colorbar.set_ticklabels(CLASSES)
    ax.set(title=f'Congestion by road on {last_day:%d %B %Y}', xlabel='Window start (IST)', ylabel='Road segment')
    save(fig, 'congestion_heatmap')
    comparison = actual.groupby('road_id').agg(volume=('traffic_volume', 'mean'), speed=('avg_speed', 'mean'),
                                               travel_time=('travel_time', 'mean'), travel_p95=('travel_time', lambda x: x.quantile(.95)))
    comparison['reliability_ratio'] = comparison.travel_p95/comparison.travel_time
    comparison.to_csv(ROOT / 'reports/road_comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), layout='constrained')
    for ax, col, label in zip(axes, ['volume', 'speed', 'reliability_ratio'], ['Mean vehicles / 30 min', 'Mean speed (km/h)', 'P95 / mean travel time']):
        ax.barh(comparison.index, comparison[col], color='#166b8c')
        ax.invert_yaxis()
        ax.set(xlabel=label)
    fig.suptitle('Road comparison using observed windows', fontsize=17)
    save(fig, 'road_comparison')
    bench = scoreboard.loc[(scoreboard.task == 'traffic_volume') & (scoreboard.horizon == 1)].sort_values('rmse')
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    bars = ax.barh(bench.model, bench.rmse, color=['#2c9c95' if m in ['GRU', 'BiLSTM'] else '#166b8c' for m in bench.model])
    ax.bar_label(bars, fmt='%.1f', padding=4)
    ax.set(xlabel='Test RMSE (vehicles / 30 min; lower is better)', title='Classical and deep models on the same next-window test records')
    ax.invert_yaxis()
    ax.set_xlim(0, bench.rmse.max()*1.13)
    save(fig, 'model_benchmark')
    # Tree importances are descriptive, not causal effects.
    tree_path = ROOT / f'models/flow_xgboost_{cfg["version"]}.joblib'
    bundle = joblib.load(tree_path)
    importance = pd.DataFrame({'feature': bundle['features'], 'importance': bundle['model'].feature_importances_}).sort_values('importance', ascending=False)
    importance.to_csv(ROOT / 'reports/feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    top = importance.head(18).iloc[::-1]
    ax.barh(top.feature.str.replace('_', ' '), top.importance, color='#166b8c')
    ax.set(title='Volume and speed model feature importance', xlabel='XGBoost normalized gain importance')
    save(fig, 'feature_importance')
    representative = sorted(flow.road_id.unique())[len(flow.road_id.unique())//2]
    sample = h1.loc[h1.road_id.eq(representative)].sort_values('target_timestamp')
    sample = sample.loc[sample.target_timestamp < sample.target_timestamp.min()+pd.Timedelta(days=3)]
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    ax.plot(sample.target_timestamp, sample.y_traffic_volume, label='Observed', color='#23394d', lw=1.7)
    ax.plot(sample.target_timestamp, sample.pred_traffic_volume, label='30-minute forecast', color='#de8957', lw=1.5)
    ax.fill_between(sample.target_timestamp, sample.lower_traffic_volume, sample.upper_traffic_volume,
                    alpha=.18, color='#de8957', label='Nominal 90% prediction interval')
    ax.set(title=f'{representative} held-out volume forecasts', xlabel='Target timestamp (IST)', ylabel='Vehicles / 30 min')
    ax.legend(loc='upper left', ncol=3)
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M', tz=ZoneInfo(cfg['timezone'])))
    fig.autofmt_xdate()
    save(fig, 'forecast_confidence')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    residual = h1.y_traffic_volume-h1.pred_traffic_volume
    sns.histplot(residual, bins=45, ax=axes[0], color='#166b8c')
    axes[0].set(xlabel='Actual minus forecast (vehicles / 30 min)', title='Volume residuals')
    axes[1].scatter(h1.pred_traffic_volume, residual, s=2, alpha=.12, color='#166b8c', rasterized=True)
    axes[1].axhline(0, color='#b63758')
    axes[1].set(xlabel='Forecast volume', ylabel='Residual', title='Residual spread versus forecast')
    save(fig, 'residual_analysis')
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    stats.probplot(residual, dist='norm', plot=ax)
    ax.set_title('Normal Q–Q plot of held-out volume residuals')
    save(fig, 'residual_normality')
    con = outputs['congestion'].loc[lambda x: x.horizon == 1]
    cm = confusion_matrix(con.y_congestion, con[[f'p_{j}' for j in range(4)]].to_numpy().argmax(1), labels=range(4))
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(title='Next-window congestion confusion matrix', xlabel='Predicted class', ylabel='Observed class')
    save(fig, 'congestion_confusion_matrix')
    risk = outputs['accident'].loc[lambda x: x.horizon == 1]
    fpr, tpr, _ = roc_curve(risk.y_accident, risk.p_1)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    ax.plot(fpr, tpr, color='#166b8c', label=f'ROC-AUC {roc_auc_score(risk.y_accident, risk.p_1):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
    ax.set(title='Next-window accident-risk ranking', xlabel='False positive rate', ylabel='True positive rate')
    ax.legend()
    save(fig, 'accident_roc')
    joined = h1.merge(df[['road_id', 'timestamp', 'weather_condition']], left_on=['road_id', 'target_timestamp'],
                     right_on=['road_id', 'timestamp'], how='left', suffixes=('', '_observed'))
    joined['hour'] = joined.target_timestamp.dt.hour
    joined['day_of_week'] = joined.target_timestamp.dt.day_name()
    joined['absolute_error'] = (joined.y_traffic_volume-joined.pred_traffic_volume).abs()
    errors = []
    for group in ['hour', 'day_of_week', 'weather_condition']:
        for name, block in joined.groupby(group):
            errors.append({'group': group, 'value': name, 'n': len(block), **regression_metrics(block.y_traffic_volume, block.pred_traffic_volume)})
    all_errors = []
    # Diagnose every trained family on each supported horizon, using actual weather
    # only as a retrospective grouping variable, never as a prediction feature.
    weather_keys = df[['road_id', 'timestamp', 'weather_condition']].rename(columns={'timestamp': 'target_timestamp'})
    for prediction_path in sorted((ROOT / 'data/processed').glob(f'*_{cfg["version"]}_test.csv.gz')):
        predictions = pd.read_csv(prediction_path)
        predictions['target_timestamp'] = pd.to_datetime(predictions.target_timestamp, utc=True).dt.tz_convert(cfg['timezone'])
        predictions = predictions.merge(weather_keys, on=['road_id', 'target_timestamp'], how='left', validate='many_to_one')
        predictions['hour'] = predictions.target_timestamp.dt.hour
        predictions['day_of_week'] = predictions.target_timestamp.dt.day_name()
        task_name = predictions.task.iloc[0] if 'task' in predictions else 'deep'
        targets = ['traffic_volume', 'avg_speed', 'travel_time'] if task_name == 'flow' else (['traffic_volume', 'congestion'] if task_name == 'deep' else [task_name])
        for h, horizon_rows in predictions.groupby('horizon'):
            for grouping in ['hour', 'day_of_week', 'weather_condition']:
                for value, block in horizon_rows.groupby(grouping):
                    for target in targets:
                        if target in ['traffic_volume', 'avg_speed', 'travel_time']:
                            metrics = regression_metrics(block[f'y_{target}'], block[f'pred_{target}'])
                        else:
                            width = 4 if target == 'congestion' else 2
                            metrics = classification_metrics(block[f'y_{target}'], block[[f'p_{j}' for j in range(width)]].to_numpy(), target)
                        all_errors.append({'model': block.model.iloc[0], 'task': target, 'horizon': int(h),
                                           'group': grouping, 'value': value, 'n': len(block), **metrics})
    pd.DataFrame(all_errors).to_csv(ROOT / 'reports/error_breakdown.csv', index=False)
    weather = joined.groupby('weather_condition')[['y_traffic_volume', 'pred_traffic_volume', 'absolute_error']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    weather[['y_traffic_volume', 'pred_traffic_volume']].rename(columns={'y_traffic_volume': 'Observed', 'pred_traffic_volume': 'Forecast'}).plot.bar(ax=axes[0], color=['#166b8c', '#de8957'], rot=0)
    axes[0].set(title='Test volume by observed weather', ylabel='Mean vehicles / 30 min', xlabel='Weather')
    weather.absolute_error.plot.bar(ax=axes[1], color='#2c9c95', rot=0)
    axes[1].set(title='Forecast error by observed weather', ylabel='MAE (vehicles / 30 min)', xlabel='Weather')
    save(fig, 'weather_vs_traffic')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, field, label in zip(axes, ['rainfall', 'visibility'], ['Hourly rainfall (mm)', 'Visibility (metres)']):
        sns.regplot(data=actual.iloc[::20], x=field, y='traffic_volume', fit_reg=False,
                    scatter_kws={'s': 6, 'alpha': .15}, ax=ax)
        ax.set(xlabel=label, ylabel='Observed vehicles / 30 min')
    fig.suptitle('Observed weather and volume (every twentieth row for readability)')
    save(fig, 'weather_observations')
    curves = pd.read_csv(ROOT / 'reports/dl_training_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, kind in zip(axes, ['GRU', 'BiLSTM']):
        for trial, block in curves[curves.model == kind].groupby('trial'):
            ax.plot(block.epoch, block.train_loss, label=f'Trial {trial+1} training', linestyle='--')
            ax.plot(block.epoch, block.validation_loss, label=f'Trial {trial+1} validation')
        ax.set(title=f'{kind} learning curves', xlabel='Epoch', ylabel='Scaled MSE + 0.25 × cross-entropy')
        ax.legend(fontsize=8)
    save(fig, 'dl_learning_curves')
    coverage = []
    for task, target in [('flow', 'traffic_volume'), ('speed', 'avg_speed'), ('travel_time', 'travel_time')]:
        for h, block in outputs[task].groupby('horizon'):
            inside = (block[f'y_{target}'] >= block[f'lower_{target}']) & (block[f'y_{target}'] <= block[f'upper_{target}'])
            coverage.append({'target': target, 'horizon': h, 'coverage': float(inside.mean()),
                             'mean_width': float((block[f'upper_{target}']-block[f'lower_{target}']).mean())})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(ROOT / 'reports/interval_coverage.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for target, block in coverage.groupby('target'):
        ax.plot(block.horizon*30, block.coverage*100, marker='o', label=target.replace('_', ' '))
    ax.axhline(90, color='gray', linestyle='--', label='Nominal 90%')
    ax.set(ylim=(0, 100), xlabel='Horizon (minutes)', ylabel='Observed test coverage (%)', title='Prediction interval coverage on observed test targets')
    ax.legend()
    save(fig, 'interval_coverage')
    calibration_rows = []
    for task, output in [('congestion', con), ('accident', risk)]:
        p = output[[f'p_{j}' for j in range(4 if task == 'congestion' else 2)]].to_numpy()
        confidence = p.max(1) if task == 'congestion' else p[:, 1]
        observed_result = (p.argmax(1) == output.y_congestion).astype(float) if task == 'congestion' else output.y_accident.to_numpy()
        bins = np.minimum((confidence*10).astype(int), 9)
        for b in np.unique(bins):
            take = bins == b
            calibration_rows.append({'task': task, 'bin': int(b), 'n': int(take.sum()),
                                     'mean_probability': float(confidence[take].mean()), 'observed_rate': float(observed_result[take].mean())})
    pd.DataFrame(calibration_rows).to_csv(ROOT / 'reports/classification_calibration.csv', index=False)
    # Benchmark a fresh model reload for 25 roads and a single horizon.
    frame = supervised(df, horizons=[1], require_targets=False)
    count = frame.groupby('timestamp').size()
    origin = count[count == 25].index.max()
    one = frame.loc[frame.timestamp.eq(origin)]
    start = time.perf_counter()
    for artifact in set(selected.values()):
        predict_bundle(joblib.load(ROOT / 'models' / artifact), one)
    inference = time.perf_counter()-start
    # Verify the deep benchmark uses identical road/origin/target keys.
    deep_keys = pd.read_csv(ROOT / f'data/processed/gru_{cfg["version"]}_test.csv.gz')[['road_id', 'timestamp', 'target_timestamp']]
    classical_keys = h1[['road_id', 'timestamp', 'target_timestamp']].astype(str)
    same_keys = set(map(tuple, classical_keys.to_numpy())) == set(map(tuple, deep_keys.astype(str).to_numpy()))
    assert same_keys, 'Classical and deep test keys differ'
    main_volume = regression_metrics(h1.y_traffic_volume, h1.pred_traffic_volume)
    main_con = classification_metrics(con.y_congestion, con[[f'p_{j}' for j in range(4)]].to_numpy(), 'congestion')
    main_risk = classification_metrics(risk.y_accident, risk[['p_0', 'p_1']].to_numpy(), 'accident')
    gru_rmse = float(dl.loc[(dl.model == 'GRU') & (dl.task == 'traffic_volume'), 'rmse'].iloc[0])
    best_classical = float(ml.loc[(ml.task == 'traffic_volume') & (ml.horizon == 1), 'rmse'].min())
    acceptance = {'volume_mape_percent': main_volume['mape'], 'volume_mape_pass': main_volume['mape'] <= 12,
                  'congestion_macro_f1': main_con['macro_f1'], 'congestion_f1_pass': main_con['macro_f1'] >= .8,
                  'accident_roc_auc': main_risk['roc_auc'], 'accident_auc_pass': main_risk['roc_auc'] >= .75,
                  'gru_volume_rmse': gru_rmse, 'best_classical_volume_rmse': best_classical,
                  'deep_beats_classical': gru_rmse < best_classical,
                  'inference_25_roads_one_horizon_seconds': inference, 'inference_pass': inference <= 30,
                  'exact_test_keys_match': same_keys, 'test_records_horizon_1': len(h1)}
    save_json(ROOT / 'reports/acceptance.json', acceptance)
    chosen = pd.DataFrame([{'Output': task, 'Artifact': artifact} for task, artifact in selected.items()])
    checks = pd.DataFrame([
        ['Volume MAPE ≤ 12%', f"{main_volume['mape']:.2f}%", main_volume['mape'] <= 12],
        ['Congestion macro-F1 ≥ 0.80', f"{main_con['macro_f1']:.4f}", main_con['macro_f1'] >= .8],
        ['Accident ROC-AUC ≥ 0.75', f"{main_risk['roc_auc']:.4f}", main_risk['roc_auc'] >= .75],
        ['GRU beats best classical volume RMSE', f'{gru_rmse:.2f} vs {best_classical:.2f}', gru_rmse < best_classical],
        ['25-road inference ≤ 30 seconds', f'{inference:.3f} seconds', inference <= 30]], columns=['Criterion', 'Measured result', 'Passed'])
    text = '# FlowCast final technical report\n\n'
    text += 'FlowCast forecasts the supplied 25-road corridor at 30, 60, 90 and 120 minutes. This report uses only recorded observations and saved model outputs. Results are from a historical replay, not a live traffic feed.\n\n'
    text += '## Acceptance results\n\nThe accuracy criteria below use the primary 30-minute horizon. Every horizon is reported separately in the scoreboard. A failed criterion is a measured limitation, not a completed accuracy claim.\n\n' + markdown_table(checks)
    text += '\n\n## Data and evaluation\n\n'
    text += f'The grid covers 1 January to 31 May 2025 in Asia/Kolkata. Training targets end on {cfg["train_end"]}. Validation ends on {cfg["validation_end"]}. The latest period is held out for testing. There are {len(h1):,} common next-window test records. The GRU, BiLSTM and every classical model use exactly the same test keys.\n\n'
    text += '''Sensor timestamps label the start of each 30-minute window. The origin timestamp is the last observed window label. A prediction is issued after that window ends, 30 minutes later. The horizon measures time from issue to the end of the predicted window. Therefore current-window features are already complete when a forecast is issued.

Features describe an origin and its known future calendar window. Current sensor readings, earlier lags and earlier rolling values are allowed. The target is a later timestamp. The dataset has no weather forecasts, so each horizon uses weather at the origin. This is a persistence assumption, not an observed future weather input. Calendar event flags are date-level. We do not invent event start times.

Two expanding training folds tune small hyperparameter grids. The maximum horizon is embargoed at fold boundaries. All scalers fit training data only. The first half of validation selects model families and supports deep-model early stopping. The later half calibrates class probabilities and regression intervals. Test data never selects models, parameters or interval widths. Invalid or imputed outcomes never become scored labels. The common evaluation population requires all five observed targets so comparisons share the same rows.

## Models and selection

'''
    text += markdown_table(chosen)
    text += '''

Regression compares NumPy gradient descent, scaled linear regression, a decision tree, random forest and XGBoost. Volume and speed share a two-output fit. Within each regression family, training folds choose complexity by volume RMSE. The same settings are used for the speed and travel-time fits. Validation then selects a separate family for each output. A travel model uses volume, speed, capacity, inferred segment length and the physical travel-time estimate. Segment length is the training median of speed × travel time / 60, because no length column was supplied. Travel training uses measured inputs; evaluation uses forecast inputs. This input shift is measured on validation and test. The selected volume, speed and travel models may differ.

Congestion and accident models compare decision tree, random forest, XGBoost and a scaled linear SVM. A linear SVM avoids the memory cost of a full kernel matrix at this data size. Validation sigmoid calibration turns margins and ensemble scores into probabilities. Accident decisions use a declared 0.5 threshold. AUC measures ranking, while recall at 0.5 shows whether a model actually flags rare incidents.

The selection table and CV results record validation scores and fold stability. Selection uses validation RMSE for each regression output, macro-F1 for congestion and AUC for risk. The measured inference time checks the workstation constraint. Trees provide feature importance and readable rules; the linear baseline is easier to explain. A more complex model is kept only when it improves the target's validation metric. AUC selection is not a guarantee of useful incident recall, so the report includes that limitation explicitly.

The GRU has a volume head and a congestion head. The bidirectional LSTM is a historical comparison. Its two directions read only the past input sequence. Both start from seeded random weights. Two GRU candidates vary sequence length, hidden size, layers, dropout, learning rate and batch settings. Adam, gradient clipping, weight decay, dropout, early stopping and learning-rate reduction control training. Best validation-loss weights are restored. No pretrained weights or transfer learning are used. Deep models predict the next window only; the classical models provide all four operational horizons. Deep accident AUC is not reported because those networks have no accident head.

## Mathematics

The feature matrix X has one row per road, origin and horizon. Linear predictions use XW + b. The NumPy baseline minimizes mean squared error with gradient 2 Xᵀ(XW − y) / n. It computes XᵀX and Xᵀy once, then performs explicit gradient-descent updates. These sufficient statistics give the same full-batch gradient. Neural dense layers also use matrix products, and backpropagation supplies gradients for Adam.

Mean, median, mode, variance and standard deviation describe the data. Rolling means and standard deviations describe past flow. Covariance measures joint changes; correlation divides covariance by the two standard deviations. Related features can be redundant, so scaled linear regression provides a useful check against tree models. Scaling uses z = (x − training mean) / training standard deviation. Bounded deep inputs use (x − training minimum) / (training maximum − training minimum). Held-out values can exceed the training range and are not clipped.

RMSE is the square root of mean squared error. MAE is the mean absolute error. MAPE is the mean absolute percentage error on nonzero observed targets; zero-target counts are reported. R² compares residual squared error with variation around the target mean. Precision measures correct positive alerts among all alerts. Recall measures captured positives among all real positives. F1 is their harmonic mean. Macro-F1 gives equal weight to all four congestion classes. A confusion matrix shows the underlying counts. ROC-AUC measures positive-versus-negative ranking over thresholds. Brier score measures squared probability error.

The recurrent loss is scaled volume MSE plus 0.25 times congestion cross-entropy. Cross-entropy penalizes low probability for the observed class. The training curves contain actual losses at every epoch. Normal residual plots check whether a Gaussian assumption is reasonable. The prediction intervals use empirical absolute residual quantiles and do not assume normal errors.

## Uncertainty and operational use

Each regression forecast includes an empirical nominal 90% interval from the later validation period, separately by horizon. These are prediction intervals, not confidence intervals for a mean. Time dependence and changing traffic mean that 90% future coverage is not guaranteed. Test coverage and interval width are reported. Class forecasts show the calibrated probability and all four congestion probabilities. The deep comparison uses uncalibrated softmax class probabilities, labelled as such. Risk confidence is max(p, 1−p), which is confidence in the selected binary class, not certainty about an accident.

Corridor travel time is the sum across the 25 supplied segments. It assumes one traversal of each segment because the files contain no route topology. The sum of segment interval bounds is displayed as a range, not a calibrated corridor 90% interval. No live feed is supplied. The latest replay origin is chosen so all four horizons remain within the provided calendar coverage. Forecasting beyond that coverage requires new calendar and weather data.

## Next-window benchmark

'''
    text += markdown_table(scoreboard.loc[scoreboard.horizon.eq(1), ['model', 'task', 'n', 'rmse', 'mape', 'macro_f1', 'roc_auc']].fillna('Not applicable'))
    text += '\n\n## Error analysis and recommendations\n\n'
    text += f'The selected volume model has next-window RMSE {main_volume["rmse"]:.2f} vehicles and MAPE {main_volume["mape"]:.2f}%. Accident recall at the 0.5 threshold is {main_risk["recall"]:.4f}, with precision {main_risk["precision"]:.4f}. '
    text += ('The GRU improves on the best classical volume RMSE in this test. ' if gru_rmse < best_classical else 'The GRU does not beat the best classical volume RMSE, so its extra complexity is not justified for volume deployment on this run. ')
    text += 'The dashboard uses validation-selected classical models and exposes the deep benchmark separately.\n\n'
    worst = pd.DataFrame(errors).sort_values('rmse', ascending=False).head(5)
    text += 'The selected volume model has its largest next-window group errors in:\n\n' + markdown_table(worst[['group', 'value', 'n', 'rmse', 'mape']])
    text += '\n\nWeather comparisons show associations, not causal effects. Rare weather groups have less evidence. Residual, confusion-matrix, ROC and calibration tables support diagnosis beyond one headline metric. Review the weakest groups before operational use. More reliable future weather and incident predictors would need additional real source data and are outside this delivery.\n\n'
    text += '## Reproduction and deliverables\n\nRun the one-command workflow in README. It regenerates cleaning outputs, model files and cards, comparison tables, PNG figures and this report. The three notebooks explain EDA, cleaning and modelling. The dashboard reads saved artifacts, offers all nine views, validates uploaded sensor files, requests forecasts, exports a selected-period report and starts an explicit separate training process. It does not fit models while rendering an ordinary view.\n'
    (ROOT / 'reports/final_report.md').write_text(text)
    forecast_latest(df).to_csv(ROOT / 'data/processed/latest_forecasts.csv', index=False)
    logging.info('Reports and %d genuine PNG visualizations generated', len(list(figures.glob('*.png'))))
