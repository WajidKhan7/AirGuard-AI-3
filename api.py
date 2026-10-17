"""
api.py
------
Python backend / API layer of the pipeline:

    Save Models
        |
    Python Backend/API   <-- this file
        |
    HTML + CSS + JavaScript

Endpoints:
    GET  /                       -> serves the dashboard (frontend/index.html)
    GET  /api/status             -> whether trained models exist + their metrics
    POST /api/train              -> upload a CSV, train all 3 models on it, save them
    POST /api/predict            -> upload a CSV, run it through the saved models
    POST /api/predict/sample     -> run the saved models on the bundled sample CSV
"""

import os
import io
import json
import traceback

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from train_models import train_all, MODELS_DIR
from inference import AirQualityPipeline

app = Flask(__name__, static_folder="frontend", static_url_path="")

SAMPLE_CSV = "sample_data/air_quality_sample.csv"


def _models_exist():
    required = ["rf_classifier.joblib", "isolation_forest.joblib",
                "forecaster_scaler.joblib", "schema.json", "metadata.json"]
    return all(os.path.exists(f"{MODELS_DIR}/{f}") for f in required)


def _load_pipeline():
    if not _models_exist():
        return None
    return AirQualityPipeline(MODELS_DIR)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    if not _models_exist():
        return jsonify({"trained": False})
    with open(f"{MODELS_DIR}/metadata.json") as f:
        metadata = json.load(f)
    return jsonify({"trained": True, "metadata": metadata})


@app.route("/api/train", methods=["POST"])
def train():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded. Attach a CSV as 'file'."}), 400
    try:
        df = pd.read_csv(io.BytesIO(file.read()))
        tmp_path = "uploaded_train.csv"
        df.to_csv(tmp_path, index=False)
        metadata = train_all(tmp_path, out_dir=MODELS_DIR)
        return jsonify({"success": True, "metadata": metadata})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


def _predict_from_df(df, forecast_steps):
    pipe = _load_pipeline()
    if pipe is None:
        return jsonify({"error": "No trained models yet. Train first via /api/train "
                                  "or use the sample dataset."}), 400
    try:
        result = pipe.run_full_pipeline(df, forecast_steps=forecast_steps)
        return jsonify({"success": True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded. Attach a CSV as 'file'."}), 400
    forecast_steps = int(request.form.get("forecast_steps", 24))
    df = pd.read_csv(io.BytesIO(file.read()))
    return _predict_from_df(df, forecast_steps)


@app.route("/api/predict/sample", methods=["POST"])
def predict_sample():
    forecast_steps = int(request.form.get("forecast_steps", 24))
    df = pd.read_csv(SAMPLE_CSV)
    return _predict_from_df(df, forecast_steps)


# Remembers which uploaded file the currently-saved models were trained on,
# so /api/analyze only retrains when the dataset actually changes.
_last_trained_filename = {"name": None}


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    One-shot endpoint for the dashboard's 'Run AI Analysis' button:
    trains on the uploaded CSV if needed, then returns a single-date
    snapshot (classification, anomaly, per-sensor tags, forecast).
    """
    file = request.files.get("file")
    date = request.form.get("date")
    area = request.form.get("area")
    forecast_steps = int(request.form.get("forecast_steps", 7))

    try:
        if file is not None:
            raw_bytes = file.read()
            df = pd.read_csv(io.BytesIO(raw_bytes))
            if file.filename != _last_trained_filename["name"] or not _models_exist():
                tmp_path = "uploaded_train.csv"
                with open(tmp_path, "wb") as f:
                    f.write(raw_bytes)
                train_all(tmp_path, out_dir=MODELS_DIR)
                _last_trained_filename["name"] = file.filename
        else:
            if not _models_exist():
                df = pd.read_csv(SAMPLE_CSV)
                train_all(SAMPLE_CSV, out_dir=MODELS_DIR)
                _last_trained_filename["name"] = "sample"
            else:
                df = pd.read_csv(SAMPLE_CSV)

        pipe = _load_pipeline()
        result = pipe.analyze_snapshot(df, date=date, area=area, forecast_steps=forecast_steps)
        return jsonify({"success": True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    if not _models_exist():
        print("No trained models found — training on the bundled sample dataset first...")
        train_all(SAMPLE_CSV, out_dir=MODELS_DIR)
    app.run(host="0.0.0.0", port=5000, debug=False)
