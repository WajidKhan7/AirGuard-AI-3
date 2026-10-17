"""
inference.py
-------------
Loads the models saved by train_models.py and runs the three branches
on new data:

    classify(df)   -> Low / Moderate / High per row
    detect_anomalies(df) -> Normal / Anomaly per row
    forecast(df, steps)  -> next N predicted values of the primary pollutant
"""

import json
import numpy as np
import pandas as pd
import joblib
import os

from preprocessing import Preprocessor
from train_models import LOOKBACK, _make_sequences

MODELS_DIR = "models"


class AirQualityPipeline:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.pp = Preprocessor().load_schema(f"{models_dir}/schema.json")

        self.clf = joblib.load(f"{models_dir}/rf_classifier.joblib")
        self.clf_scaler = joblib.load(f"{models_dir}/rf_scaler.joblib")
        self.label_encoder = joblib.load(f"{models_dir}/label_encoder.joblib")

        self.iso = joblib.load(f"{models_dir}/isolation_forest.joblib")
        self.iso_scaler = joblib.load(f"{models_dir}/iso_scaler.joblib")

        self.fc_scaler = joblib.load(f"{models_dir}/forecaster_scaler.joblib")
        if os.path.exists(f"{models_dir}/forecaster_lstm.keras"):
            import tensorflow as tf
            self.forecaster = tf.keras.models.load_model(f"{models_dir}/forecaster_lstm.keras")
            self.forecaster_backend = "tensorflow_lstm"
        else:
            self.forecaster = joblib.load(f"{models_dir}/forecaster_mlp.joblib")
            self.forecaster_backend = "sklearn_mlp"

        with open(f"{models_dir}/metadata.json") as f:
            self.metadata = json.load(f)

    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame):
        return self.pp.transform(df)

    def classify(self, df: pd.DataFrame):
        clean = self.preprocess(df)
        X = clean[self.pp.schema["feature_cols"]].values
        X_scaled = self.clf_scaler.transform(X)
        preds = self.clf.predict(X_scaled)
        labels = self.label_encoder.inverse_transform(preds)
        proba = self.clf.predict_proba(X_scaled)
        confidence = proba.max(axis=1)
        return pd.DataFrame({
            "category": labels,
            "confidence": confidence.round(3),
        })

    def detect_anomalies(self, df: pd.DataFrame):
        clean = self.preprocess(df)
        X = clean[self.pp.schema["feature_cols"]].values
        X_scaled = self.iso_scaler.transform(X)
        preds = self.iso.predict(X_scaled)          # 1 normal, -1 anomaly
        scores = self.iso.decision_function(X_scaled)  # higher = more normal
        return pd.DataFrame({
            "status": np.where(preds == 1, "Normal", "Anomaly"),
            "anomaly_score": scores.round(4),
        })

    def forecast(self, df: pd.DataFrame, steps: int = 24):
        clean = self.preprocess(df)
        target_col = self.pp.schema["primary_pollutant"]
        series = clean[target_col].astype(float).values
        lookback = self.metadata["forecaster"]["lookback"]

        if len(series) < lookback:
            raise ValueError(f"Need at least {lookback} rows of history to forecast.")

        series_scaled = self.fc_scaler.transform(series.reshape(-1, 1)).flatten()
        window = list(series_scaled[-lookback:])
        preds_scaled = []

        for _ in range(steps):
            x = np.array(window[-lookback:]).reshape(1, -1)
            if self.forecaster_backend == "tensorflow_lstm":
                x = x.reshape(1, lookback, 1)
                next_val = float(self.forecaster.predict(x, verbose=0).flatten()[0])
            else:
                next_val = float(self.forecaster.predict(x)[0])
            preds_scaled.append(next_val)
            window.append(next_val)

        preds = self.fc_scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return pd.DataFrame({
            "step": range(1, steps + 1),
            f"predicted_{target_col}": preds.round(2),
        })

    def analyze_snapshot(self, df: pd.DataFrame, date: str = None, area: str = None,
                          forecast_steps: int = 7):
        """
        Single-point-in-time view for a dashboard: pick the row(s) matching
        `date` (or the most recent row if no date/timestamp match), classify
        it, check it for anomaly, tag each sensor Low/Moderate/High against
        the dataset's own distribution, and forecast forward from there.
        """
        clean = self.preprocess(df)
        feature_cols = self.pp.schema["feature_cols"]
        ts_col = self.pp.schema.get("timestamp_col")

        snapshot_row = None
        matched_date = None
        if ts_col and ts_col in clean.columns and date:
            day = pd.to_datetime(date).date()
            mask = clean[ts_col].dt.date == day
            if mask.any():
                snapshot_row = clean.loc[mask, feature_cols].mean().to_frame().T
                matched_date = str(day)

        if snapshot_row is None:
            snapshot_row = clean.iloc[[-1]][feature_cols]
            if ts_col and ts_col in clean.columns:
                matched_date = str(pd.to_datetime(clean.iloc[-1][ts_col]).date())

        cls = self.classify(snapshot_row)
        anom = self.detect_anomalies(snapshot_row)

        # Per-sensor Low/Moderate/High tags, judged against this dataset's
        # own distribution (so it works for any pollutant/unit).
        sensor_status = {}
        for col in self.pp.schema["pollutant_cols"]:
            q1, q2 = clean[col].quantile([0.5, 0.85])
            v = float(snapshot_row[col].values[0])
            if v <= q1:
                status = "Normal"
            elif v <= q2:
                status = "Elevated"
            else:
                status = "High"
            sensor_status[col] = {"value": round(v, 2), "status": status}
        for col in self.pp.schema["weather_cols"]:
            sensor_status[col] = {"value": round(float(snapshot_row[col].values[0]), 2),
                                   "status": None}

        try:
            fc = self.forecast(df, steps=forecast_steps)
            forecast_records = json.loads(fc.to_json(orient="records"))
        except ValueError:
            forecast_records = []

        date_range = None
        if ts_col and ts_col in clean.columns:
            date_range = {
                "start": str(pd.to_datetime(clean[ts_col].min()).date()),
                "end": str(pd.to_datetime(clean[ts_col].max()).date()),
            }

        return {
            "matched_date": matched_date,
            "area": area,
            "date_range": date_range,
            "classification": {
                "category": cls["category"].iloc[0],
                "confidence": float(cls["confidence"].iloc[0]),
            },
            "anomaly": {
                "status": anom["status"].iloc[0],
                "score": float(anom["anomaly_score"].iloc[0]),
            },
            "sensors": sensor_status,
            "primary_pollutant": self.pp.schema["primary_pollutant"],
            "forecast": forecast_records,
        }

    def run_full_pipeline(self, df: pd.DataFrame, forecast_steps: int = 24):
        clean = self.preprocess(df)
        cls = self.classify(df)
        anom = self.detect_anomalies(df)
        fc = self.forecast(df, steps=forecast_steps)

        combined = clean.reset_index(drop=True).copy()
        combined["category"] = cls["category"].values
        combined["confidence"] = cls["confidence"].values
        combined["status"] = anom["status"].values
        combined["anomaly_score"] = anom["anomaly_score"].values

        ts_col = self.pp.schema.get("timestamp_col")
        display_cols = ([ts_col] if ts_col else []) + \
                        self.pp.schema["feature_cols"] + \
                        ["category", "confidence", "status", "anomaly_score"]

        return {
            "rows": json.loads(combined[display_cols].to_json(orient="records", date_format="iso")),
            "forecast": json.loads(fc.to_json(orient="records")),
            "summary": {
                "n_rows": len(combined),
                "category_counts": combined["category"].value_counts().to_dict(),
                "n_anomalies": int((combined["status"] == "Anomaly").sum()),
                "primary_pollutant": self.pp.schema["primary_pollutant"],
                "forecaster_backend": self.forecaster_backend,
            },
        }


if __name__ == "__main__":
    pipe = AirQualityPipeline()
    df = pd.read_csv("sample_data/air_quality_sample.csv")
    result = pipe.run_full_pipeline(df, forecast_steps=12)
    print(json.dumps(result["summary"], indent=2))
    print("Sample forecast:", result["forecast"][:3])
