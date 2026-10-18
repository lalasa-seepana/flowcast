# FlowCast: AI-Powered Smart Traffic Flow Prediction

FlowCast predicts traffic volume, speed, congestion, travel time and accident risk for the 25-road Northline Corridor. It follows the supplied FlowCast PRD v1.0 and Data Dictionary v1.0. The three source CSV files cover 1 January to 31 May 2025. The PRD requires timezone-aware timestamps but does not specify a timezone. This project uses `Asia/Kolkata` (India Standard Time), as confirmed by the user.

The dashboard is a historical replay. It does not connect to a live traffic feed. Every chart and score comes from the supplied data or a trained model. See the [final technical report](reports/final_report.md) for measured results and any unmet accuracy targets.

## Setup

Use Python **3.11.14**. From the project root, run one setup command.

Linux or macOS:

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv; if ($?) { .\.venv\Scripts\python -m pip install -r requirements.txt }
```

The requirements pin the required libraries and their resolved dependencies. PyTorch uses CPU wheels on Linux and Windows. The macOS entries use the standard CPU-compatible distributions. No CUDA libraries are needed. The Linux environment was tested here; macOS and Windows commands are provided but were not executed. Windows also needs the Visual C++ Redistributable when it is not already installed. The [XGBoost 3.0 installation guide](https://xgboost.readthedocs.io/en/release_3.0.0/install.html) documents this operating-system prerequisite and the CPU-only wheels.

This workspace already contains a Python 3.11 virtual environment at `.venv`. A local Python runtime is in `.python` because the system did not provide Python 3.11. These environment folders are excluded from Git. You can use the prepared environment directly without another install.

## Reproduce the project

The exact three supplied CSV files must be in `data/raw/`. They are already present in this workspace. Do not edit them. Run the whole workflow with one command:

```bash
.venv/bin/python -m src.utils
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe` in all commands. The workflow validates and cleans the data, builds features, tunes and trains the classical models, trains the recurrent models, evaluates them, writes the reports and PNG figures, executes the three notebooks and runs the automated tests. It uses the full supplied corridor data. Runtime depends on the CPU and available memory. The default uses two CPU threads and lazy sequence batches to limit memory use.

For a focused rerun, use `--stage data`, `--stage ml`, `--stage dl` or `--stage report`. Run these in order if outputs do not exist. Models are seeded and CPU operations are deterministic in the pinned environment. Exact metrics across other CPUs or operating systems may differ slightly because numerical libraries can differ.

Open the dashboard:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Run the tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Open the notebooks:

```bash
.venv/bin/python -m notebook notebooks
```

## Project layout

```text
flowcast-project/
├── data/
│   ├── raw/                 original CSV copies, unchanged
│   ├── interim/             cleaned source tables and validation audit
│   └── processed/           versioned merged data and real forecast outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── ml_models.py
│   ├── dl_model.py
│   ├── evaluate.py
│   └── utils.py
├── models/                  trained artifacts and model cards
├── reports/
│   ├── data_quality.md
│   ├── final_report.md
│   └── figures/             labelled PNGs generated from actual data
├── dashboard/
│   └── app.py
├── tests/
├── requirements.txt
├── config.yaml
└── README.md
```

The classical model implementation is `src/ml_models.py`, as the PRD specifies. The original Word documents remain unchanged in the supplied source folder. The runtime and Git support files support execution and reproducibility.

## Data and modelling rules

The dictionary controls source keys and units. Weather joins by station and aligned hour. Calendar data joins by local date. Both half-hour readings share that hour's weather. Rainfall remains an hourly measurement. Congestion uses volume divided by half the hourly capacity, with boundaries 0.50, 0.80 and 1.00.

The data-quality report records duplicate removal, missing windows, invalid values, weather normalization and label derivation. `data/interim/validation_audit.csv` records each rejected or repaired invalid cell and duplicate by its original source row. Missing windows have their own audit table. Observation flags distinguish real labels from repaired history. Original hashes are in `data/processed/source_manifest.json`.

Historical interpolation is kept separate from causal model values. Sensor timestamps label window starts. A forecast is issued after the last observed window ends, 30 minutes after its timestamp. The 30–120 minute horizon runs from issue time to the end of the target window. Features use completed current and earlier sensor readings, past lags, past rolling statistics and calendar flags known for the target date. Future weather readings never enter a forecast. Long-gap medians use only earlier readings. Scalers fit on training data only. Test metrics exclude imputed target values.

Training targets end on 15 April. Validation ends on 8 May. Test origins start after that boundary. Expanding time-series folds include a four-window gap. The earlier validation half selects models. The later half calibrates probabilities and prediction intervals. Models are saved with their features, parameters, time windows and test scores. Prediction rows identify their model version, road, origin and target time.

Volume and speed models use the same direct four-horizon feature table. Travel-time models use forecast volume and speed, capacity and a segment length inferred only from training observations. The NumPy linear baseline implements loss and gradients directly. Other baselines use scikit-learn and XGBoost. A scaled linear SVM provides the required classification baseline.

The GRU and bidirectional LSTM start with random weights. They predict next-window volume and congestion and use exactly the same next-window test records as the classical models. Their training curves and search results are saved. Both use the same training-fitted scalers, stored once in `models/sequence_v1_scalers.joblib` and referenced by both model cards. The bidirectional model reads both directions of the past input window only and is shown as a historical benchmark.

## Dashboard workflow

The sidebar opens all nine views directly. Analysts start on the congestion heatmap. Planners start on road comparison. Incident coordinators start on forecasts ranked by accident risk. Reviewers start on model performance.

The nine views are Live prediction, Historical trends, Congestion heatmap, Road comparison, Model performance, Feature importance, Forecast visualisation, Prediction confidence and Weather vs traffic. Model performance uses the fixed hold-out period; historical date filters do not redefine test metrics.

Use **Request a forecast** to choose an origin. Predictions reload trained models and do not retrain. The latest complete four-horizon origin is 31 May at 21:30. Missing future calendar coverage stops a forecast instead of inventing flags. Times are historical and should not be treated as current traffic.

Use **Upload sensor data** for a CSV with the original traffic schema. The shared validator and cleaner process it in memory with the supplied weather and calendar tables. Uploaded rows take priority over original rows only on equally complete duplicate keys. The raw files stay unchanged. Upload dates must be within the available weather and calendar coverage. Download the audit to inspect rejected rows and repairs. Refreshed predictions are session-local.

Use **Reports and insights** to export observed aggregates for the selected dates. Use **Model training** to explicitly start a separate pipeline process and inspect its status and metrics. Normal dashboard views only load persisted results. Wait for a training run to complete before reviewing its full set of outputs.

## Reading confidence and results

Regression forecasts have nominal 90% prediction intervals from validation residuals. Check measured test coverage in `reports/interval_coverage.csv`. Time dependence means future coverage is not guaranteed. Congestion confidence is the probability of the selected class. Accident risk is the incident probability. Risk confidence is `max(p, 1-p)` and can be high even when incident recall is low. Deep-model class probabilities are uncalibrated and are labelled as such.

Corridor travel time sums all 25 segment predictions. This assumes one traversal of each segment because route topology was not supplied. Summed segment bounds are a range, not a calibrated corridor interval.

The final report explains mathematics, model selection, uncertainty, error groups and limitations. `reports/model_comparison.csv` gives every model, target and supported horizon. `reports/acceptance.json` records the measured PRD criteria. A missed target is reported honestly. No output is edited to force a passing score.
