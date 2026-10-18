# FlowCast final technical report

FlowCast forecasts the supplied 25-road corridor at 30, 60, 90 and 120 minutes. This report uses only recorded observations and saved model outputs. Results are from a historical replay, not a live traffic feed.

## Acceptance results

The accuracy criteria below use the primary 30-minute horizon. Every horizon is reported separately in the scoreboard. A failed criterion is a measured limitation, not a completed accuracy claim.

| Criterion | Measured result | Passed |
| --- | --- | --- |
| Volume MAPE ≤ 12% | 9.02% | True |
| Congestion macro-F1 ≥ 0.80 | 0.7661 | False |
| Accident ROC-AUC ≥ 0.75 | 0.6074 | False |
| GRU beats best classical volume RMSE | 61.97 vs 56.83 | False |
| 25-road inference ≤ 30 seconds | 0.134 seconds | True |

## Data and evaluation

The grid covers 1 January to 31 May 2025 in Asia/Kolkata. Training targets end on 2025-04-15 23:30. Validation ends on 2025-05-08 23:30. The latest period is held out for testing. There are 26,124 common next-window test records. The GRU, BiLSTM and every classical model use exactly the same test keys.

Sensor timestamps label the start of each 30-minute window. The origin timestamp is the last observed window label. A prediction is issued after that window ends, 30 minutes later. The horizon measures time from issue to the end of the predicted window. Therefore current-window features are already complete when a forecast is issued.

Features describe an origin and its known future calendar window. Current sensor readings, earlier lags and earlier rolling values are allowed. The target is a later timestamp. The dataset has no weather forecasts, so each horizon uses weather at the origin. This is a persistence assumption, not an observed future weather input. Calendar event flags are date-level. We do not invent event start times.

Two expanding training folds tune small hyperparameter grids. The maximum horizon is embargoed at fold boundaries. All scalers fit training data only. The first half of validation selects model families and supports deep-model early stopping. The later half calibrates class probabilities and regression intervals. Test data never selects models, parameters or interval widths. Invalid or imputed outcomes never become scored labels. The common evaluation population requires all five observed targets so comparisons share the same rows.

## Models and selection

| Output | Artifact |
| --- | --- |
| flow | flow_random_forest_v1.joblib |
| speed | flow_xgboost_v1.joblib |
| travel_time | flow_random_forest_v1.joblib |
| congestion | congestion_xgboost_v1.joblib |
| accident | accident_random_forest_v1.joblib |

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

| model | task | n | rmse | mape | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Numpy GD | traffic_volume | 26124 | 96.2216 | 22.0212 | Not applicable | Not applicable |
| Numpy GD | avg_speed | 26124 | 4.9814 | 12.5591 | Not applicable | Not applicable |
| Numpy GD | travel_time | 26124 | 1.4121 | 13.0440 | Not applicable | Not applicable |
| Linear Regression | traffic_volume | 26124 | 96.4253 | 21.7735 | Not applicable | Not applicable |
| Linear Regression | avg_speed | 26124 | 4.9815 | 12.7590 | Not applicable | Not applicable |
| Linear Regression | travel_time | 26124 | 1.4402 | 10.5206 | Not applicable | Not applicable |
| Decision Tree | traffic_volume | 26124 | 62.1229 | 9.7675 | Not applicable | Not applicable |
| Decision Tree | avg_speed | 26124 | 3.8417 | 9.1602 | Not applicable | Not applicable |
| Decision Tree | travel_time | 26124 | 1.1196 | 8.5872 | Not applicable | Not applicable |
| Random Forest | traffic_volume | 26124 | 56.8267 | 9.0176 | Not applicable | Not applicable |
| Random Forest | avg_speed | 26124 | 3.6263 | 8.7573 | Not applicable | Not applicable |
| Random Forest | travel_time | 26124 | 1.1058 | 8.1716 | Not applicable | Not applicable |
| XGBoost | traffic_volume | 26124 | 57.2514 | 9.3333 | Not applicable | Not applicable |
| XGBoost | avg_speed | 26124 | 3.4343 | 8.3712 | Not applicable | Not applicable |
| XGBoost | travel_time | 26124 | 1.0910 | 7.7768 | Not applicable | Not applicable |
| Decision Tree | congestion | 26124 | Not applicable | Not applicable | 0.7460 | Not applicable |
| Random Forest | congestion | 26124 | Not applicable | Not applicable | 0.7598 | Not applicable |
| XGBoost | congestion | 26124 | Not applicable | Not applicable | 0.7661 | Not applicable |
| SVM | congestion | 26124 | Not applicable | Not applicable | 0.5818 | Not applicable |
| Decision Tree | accident | 26124 | Not applicable | Not applicable | 0.4976 | 0.5860 |
| Random Forest | accident | 26124 | Not applicable | Not applicable | 0.4976 | 0.6074 |
| XGBoost | accident | 26124 | Not applicable | Not applicable | 0.4976 | 0.6297 |
| SVM | accident | 26124 | Not applicable | Not applicable | 0.4976 | 0.6456 |
| GRU | traffic_volume | 26124 | 61.9687 | 10.6277 | Not applicable | Not applicable |
| GRU | congestion | 26124 | Not applicable | Not applicable | 0.7564 | Not applicable |
| BiLSTM | traffic_volume | 26124 | 63.2537 | 9.9563 | Not applicable | Not applicable |
| BiLSTM | congestion | 26124 | Not applicable | Not applicable | 0.7587 | Not applicable |

## Error analysis and recommendations

The selected volume model has next-window RMSE 56.83 vehicles and MAPE 9.02%. Accident recall at the 0.5 threshold is 0.0000, with precision 0.0000. The GRU does not beat the best classical volume RMSE, so its extra complexity is not justified for volume deployment on this run. The dashboard uses validation-selected classical models and exposes the deep benchmark separately.

The selected volume model has its largest next-window group errors in:

| group | value | n | rmse | mape |
| --- | --- | --- | --- | --- |
| hour | 8 | 1092 | 96.3991 | 8.5731 |
| hour | 9 | 1086 | 94.6802 | 8.9159 |
| hour | 17 | 1093 | 88.5687 | 8.9063 |
| hour | 18 | 1093 | 88.0448 | 8.5338 |
| hour | 7 | 1100 | 78.5191 | 9.6208 |

Weather comparisons show associations, not causal effects. Rare weather groups have less evidence. Residual, confusion-matrix, ROC and calibration tables support diagnosis beyond one headline metric. Review the weakest groups before operational use. More reliable future weather and incident predictors would need additional real source data and are outside this delivery.

## Reproduction and deliverables

Run the one-command workflow in README. It regenerates cleaning outputs, model files and cards, comparison tables, PNG figures and this report. The three notebooks explain EDA, cleaning and modelling. The dashboard reads saved artifacts, offers all nine views, validates uploaded sensor files, requests forecasts, exports a selected-period report and starts an explicit separate training process. It does not fit models while rendering an ordinary view.

## Verification

17 automated tests passed. All three notebooks executed successfully. The tests cover original CSV integrity, grid reconciliation, causal cleaning, lag timing, time splits, model reloads, matching deep/classical test keys, metric calculations, all nine dashboard views, prediction, upload and report exports. Training-interface tests check process launch and status with a test double, so they do not overwrite trained models. Full output is in `reports/test_results.txt`.
