<<<<<<< HEAD
# Air Quality Pipeline

Implements the architecture you sketched:

```
Air Quality CSV
      |
Data Preprocessing
      |
  +---+-----------+-------------+
  |               |             |
RandomForest   IsolationForest  Time-series model
(Low/Mod/High) (Normal/Anomaly) (future AQ prediction)
  +---+-----------+-------------+
      |
  Save Models
      |
Python Backend/API
      |
HTML + CSS + JavaScript
```

It's built to work on **any** air-quality CSV, not just one fixed schema —
`preprocessing.py` auto-detects the timestamp column and pollutant columns
(PM2.5, PM10, CO, NO2, SO2, O3, NH3, AQI, etc., under almost any naming
convention) instead of assuming fixed column names.

## Files

| File | Role |
|---|---|
| `preprocessing.py` | Auto-detects columns, cleans data, derives a Low/Moderate/High label if the CSV doesn't already have one |
| `train_models.py` | Trains the Random Forest classifier, Isolation Forest anomaly detector, and the time-series forecaster; saves everything to `models/` |
| `inference.py` | Loads saved models and runs them on new data |
| `api.py` | Flask backend — serves the dashboard and exposes `/api/train`, `/api/predict`, `/api/predict/sample`, `/api/status` |
| `frontend/index.html` | AirGuard AI dashboard — pick a date and area, click "Run AI Analysis", get real classification/anomaly/forecast results from the trained models (no more hardcoded demo numbers) |
| `sample_data/air_quality_sample.csv` | Synthetic sample dataset (with injected anomalies and missing values) so you can try everything immediately without your own data |

## Running it

```bash
pip install flask pandas numpy scikit-learn joblib
python3 api.py
```

Then open `http://localhost:5000`. On first run, if no models exist yet, it
automatically trains on the bundled sample CSV so the dashboard isn't empty.

From the dashboard you can:
- **Train on this file** — upload any air-quality CSV and train all three
  models on it (overwrites the saved models)
- **Run saved models on this file** — run the currently saved models on a
  new CSV (must have the same columns the models were trained on)
- **Use bundled sample data** — quick demo with no upload needed

## About the time-series model

The diagram calls for an LSTM. This code path tries to use a real
TensorFlow/Keras LSTM (`train_forecaster()` in `train_models.py`) **if
TensorFlow is installed in the environment it runs in**. This sandbox
couldn't install TensorFlow (disk/network constraints), so it automatically
falls back to a lag-window MLP regressor that solves the same forecasting
problem with the same input/output contract. If you `pip install tensorflow`
on your own machine (or the ASCEND lab PC) and retrain, it will
transparently switch to the real LSTM — no code changes needed.

## Retraining on your own data

```bash
python3 train_models.py path/to/your_air_quality.csv
```

Or just upload it through the dashboard's "Train on this file" button.
=======
# AirGuard-AI-3
A Python-based machine learning application for air quality classification, anomaly detection, and pollutant forecasting. Features automated data preprocessing, model training, performance evaluation, and an interactive Tkinter desktop dashboard for air quality analysis.
>>>>>>> 0644849839d513c41fd1a19983097b944f6a5976
