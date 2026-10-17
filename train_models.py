"""
train_models.py
----------------
Trains the three-branch pipeline from the architecture diagram on ANY
air-quality CSV:

    Air Quality CSV
          |
    Data Preprocessing   (preprocessing.py)
          |
    +-----+-----+-------------+
    |           |             |
 RandomForest  IsolationForest  Time-series model
 (Low/Mod/High) (Normal/Anomaly) (future AQ prediction)
    +-----+-----+-------------+
          |
    Save Models  ->  models/

Usage:
    python3 train_models.py path/to/any_air_quality.csv
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPRegressor

from preprocessing import Preprocessor

MODELS_DIR = "models"
LOOKBACK = 12  # how many past timesteps the forecaster looks at


# --------------------------------------------------------------------
# Branch 1: Random Forest classification (Low / Moderate / High)
# --------------------------------------------------------------------
def train_classifier(df, feature_cols, label_col):
    X = df[feature_cols].values
    y = df[label_col].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    return clf, scaler, le, {"accuracy": acc, "report": report}


# --------------------------------------------------------------------
# Branch 2: Isolation Forest anomaly detection (Normal / Anomaly)
# --------------------------------------------------------------------
def train_anomaly_detector(df, feature_cols, contamination=0.02):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1,
    )
    iso.fit(X_scaled)

    preds = iso.predict(X_scaled)  # 1 = normal, -1 = anomaly
    n_anomalies = int((preds == -1).sum())

    return iso, scaler, {"n_samples": len(df), "n_anomalies": n_anomalies,
                          "anomaly_rate": round(n_anomalies / len(df), 4)}


# --------------------------------------------------------------------
# Branch 3: Time-series forecasting (future air-quality prediction)
# --------------------------------------------------------------------
def _make_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def train_forecaster(df, target_col, lookback=LOOKBACK):
    series = df[target_col].astype(float).values
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = _make_sequences(series_scaled, lookback)
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    backend = "sklearn_mlp"
    model = None

    # Prefer a real LSTM if TensorFlow happens to be available in the
    # environment this is run in (it is not installed in this sandbox,
    # but this code path activates automatically wherever TF *is*
    # available, e.g. on the student's own machine / ASCEND lab PC).
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        X_train_lstm = X_train.reshape(-1, lookback, 1)
        X_test_lstm = X_test.reshape(-1, lookback, 1)

        model = models.Sequential([
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_lstm, y_train, epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0)
        test_pred = model.predict(X_test_lstm, verbose=0).flatten()
        backend = "tensorflow_lstm"
    except ImportError:
        # Fallback: a lag-feature MLP regressor. Same input/output contract
        # (lookback window -> next value), so the API layer doesn't need
        # to know which backend trained the model.
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=2000, random_state=42, early_stopping=True,
        )
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)

    mae_scaled = float(np.mean(np.abs(test_pred - y_test)))
    # convert MAE back to original units
    mae = float(scaler.inverse_transform([[mae_scaled]])[0][0] - scaler.inverse_transform([[0]])[0][0])

    return model, scaler, {
        "backend": backend,
        "lookback": lookback,
        "target_col": target_col,
        "test_mae_original_units": round(abs(mae), 3),
    }


# --------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------
def train_all(csv_path, out_dir=MODELS_DIR):
    import os
    os.makedirs(out_dir, exist_ok=True)

    raw = pd.read_csv(csv_path)
    pp = Preprocessor()
    clean = pp.fit_transform(raw)
    schema = pp.schema

    print(f"Loaded {len(raw)} rows. Detected schema:\n{json.dumps(schema, indent=2)}")

    # --- classification ---
    clf, clf_scaler, le, clf_metrics = train_classifier(
        clean, schema["feature_cols"], schema["label_col"]
    )
    joblib.dump(clf, f"{out_dir}/rf_classifier.joblib")
    joblib.dump(clf_scaler, f"{out_dir}/rf_scaler.joblib")
    joblib.dump(le, f"{out_dir}/label_encoder.joblib")
    print(f"[1/3] Random Forest classifier — accuracy: {clf_metrics['accuracy']:.3f}")

    # --- anomaly detection ---
    iso, iso_scaler, iso_metrics = train_anomaly_detector(clean, schema["feature_cols"])
    joblib.dump(iso, f"{out_dir}/isolation_forest.joblib")
    joblib.dump(iso_scaler, f"{out_dir}/iso_scaler.joblib")
    print(f"[2/3] Isolation Forest — flagged {iso_metrics['n_anomalies']}/{iso_metrics['n_samples']} "
          f"rows as anomalies ({iso_metrics['anomaly_rate']:.1%})")

    # --- forecasting ---
    forecaster, fc_scaler, fc_metrics = train_forecaster(clean, schema["primary_pollutant"])
    if fc_metrics["backend"] == "tensorflow_lstm":
        forecaster.save(f"{out_dir}/forecaster_lstm.keras")
    else:
        joblib.dump(forecaster, f"{out_dir}/forecaster_mlp.joblib")
    joblib.dump(fc_scaler, f"{out_dir}/forecaster_scaler.joblib")
    print(f"[3/3] Forecaster ({fc_metrics['backend']}) on '{fc_metrics['target_col']}' — "
          f"test MAE: {fc_metrics['test_mae_original_units']}")

    # --- save schema + all metrics for the API/frontend to use ---
    pp.save_schema(f"{out_dir}/schema.json")
    metadata = {
        "schema": schema,
        "classifier": {"accuracy": clf_metrics["accuracy"], "classes": list(le.classes_)},
        "anomaly": iso_metrics,
        "forecaster": fc_metrics,
        "rows_trained_on": len(clean),
    }
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nAll models saved to '{out_dir}/'")
    return metadata


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/air_quality_sample.csv"
    train_all(csv_path)
