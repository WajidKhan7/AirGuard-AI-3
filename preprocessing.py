"""
preprocessing.py
-----------------
Generic preprocessing for ANY air-quality CSV.

Works by auto-detecting:
  - a timestamp/date column (if present)
  - pollutant / sensor reading columns (numeric columns whose names match
    common air-quality tokens, falling back to "all numeric columns" if
    nothing matches)

Produces a clean DataFrame plus a `schema` dict describing what was found,
so downstream training/inference code never has to hard-code column names.
"""

import re
import json
import numpy as np
import pandas as pd

# Tokens we look for (case-insensitive) when guessing pollutant columns.
POLLUTANT_TOKENS = [
    "pm2.5", "pm25", "pm2_5", "pm10", "pm1",
    "co2", "co", "no2", "nox", "no", "so2", "o3", "ozone",
    "nh3", "aqi", "voc", "tvoc", "benzene", "toluene", "xylene",
    "smoke", "particulate", "dust",
    "nitrogen", "sulfur", "sulphur", "ammonia", "methane", "carbon",
]

# Tokens that hint a column is a timestamp.
TIME_TOKENS = ["date", "time", "timestamp", "datetime", "dt"]

# Tokens for auxiliary weather columns (kept as features, not labels).
WEATHER_TOKENS = ["temp", "humid", "pressure", "wind", "rain"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", name.lower())


def detect_timestamp_column(df: pd.DataFrame):
    for col in df.columns:
        n = _norm(col)
        if any(tok in n for tok in TIME_TOKENS):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.5:  # most values parse as dates
                return col
    # fallback: try every non-numeric column, see if it parses well as a date
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return col
    return None


def detect_pollutant_columns(df: pd.DataFrame, exclude=()):
    numeric_cols = [c for c in df.columns if c not in exclude and
                     pd.api.types.is_numeric_dtype(df[c])]
    matched = []
    for col in numeric_cols:
        n = _norm(col)
        if any(tok in n for tok in POLLUTANT_TOKENS):
            matched.append(col)
    if matched:
        return matched
    # Fallback: no recognizable pollutant names -> use all numeric columns
    # that aren't obviously weather-only.
    fallback = [c for c in numeric_cols
                if not any(tok in _norm(c) for tok in WEATHER_TOKENS)]
    return fallback or numeric_cols


def detect_weather_columns(df: pd.DataFrame, exclude=()):
    numeric_cols = [c for c in df.columns if c not in exclude and
                     pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in numeric_cols if any(tok in _norm(c) for tok in WEATHER_TOKENS)]


# --- AQI categorisation -----------------------------------------------
# Simplified 3-class scheme (Low / Moderate / High), primarily driven by
# PM2.5 (US EPA-style breakpoints) when available, otherwise by whichever
# primary pollutant column was detected.
PM25_BREAKPOINTS = [
    (0, 12, "Low"),
    (12, 35.4, "Moderate"),
    (35.4, float("inf"), "High"),
]

GENERIC_BREAKPOINTS_BY_PERCENTILE = True  # used when no PM2.5-like column


def _choose_primary_pollutant(pollutant_cols):
    priority = ["pm2.5", "pm25", "pm2_5", "aqi", "pm10", "no2", "co", "so2", "o3"]
    for tok in priority:
        for col in pollutant_cols:
            if _norm(col) == _norm(tok) or tok in _norm(col):
                return col
    return pollutant_cols[0] if pollutant_cols else None


def compute_aqi_category(df: pd.DataFrame, pollutant_cols, existing_label_col=None):
    """Return (labels: pd.Series, primary_col: str) - Low/Moderate/High."""
    if existing_label_col and existing_label_col in df.columns:
        return df[existing_label_col].astype(str), existing_label_col

    primary = _choose_primary_pollutant(pollutant_cols)
    if primary is None:
        raise ValueError("No numeric pollutant columns detected to derive a label from.")

    values = df[primary].astype(float)
    is_pm25 = "pm2" in _norm(primary)

    if is_pm25:
        def categorize(v):
            for lo, hi, label in PM25_BREAKPOINTS:
                if lo <= v < hi:
                    return label
            return "High"
        labels = values.apply(categorize)
    else:
        # Percentile-based tiers so this works for *any* pollutant/unit.
        q1, q2 = values.quantile([0.5, 0.85])
        def categorize(v):
            if v <= q1:
                return "Low"
            elif v <= q2:
                return "Moderate"
            return "High"
        labels = values.apply(categorize)

    return labels, primary


class Preprocessor:
    """Fit on one CSV, remembers the detected schema, reusable at inference time."""

    def __init__(self):
        self.schema = {}

    def fit_transform(self, df: pd.DataFrame, label_col: str = None):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        ts_col = detect_timestamp_column(df)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values(ts_col).reset_index(drop=True)

        exclude = {ts_col} if ts_col else set()
        pollutant_cols = detect_pollutant_columns(df, exclude=exclude)
        weather_cols = detect_weather_columns(df, exclude=exclude)
        feature_cols = list(dict.fromkeys(pollutant_cols + weather_cols))

        if not feature_cols:
            raise ValueError("No usable numeric columns found in this CSV.")

        # Clean: coerce numeric, interpolate missing values, drop fully-empty rows
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[feature_cols] = df[feature_cols].interpolate(limit_direction="both")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        df = df.drop_duplicates()

        labels, primary_col = compute_aqi_category(df, pollutant_cols, existing_label_col=label_col)
        df["_aqi_category"] = labels.values

        self.schema = {
            "timestamp_col": ts_col,
            "pollutant_cols": pollutant_cols,
            "weather_cols": weather_cols,
            "feature_cols": feature_cols,
            "primary_pollutant": primary_col,
            "label_col": "_aqi_category",
        }
        return df

    def transform(self, df: pd.DataFrame):
        """Apply a previously-fit schema to new/incoming data (inference time)."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        ts_col = self.schema.get("timestamp_col")
        feature_cols = self.schema["feature_cols"]

        if ts_col and ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values(ts_col).reset_index(drop=True)

        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[feature_cols] = df[feature_cols].interpolate(limit_direction="both")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median()).fillna(0)
        return df

    def save_schema(self, path):
        with open(path, "w") as f:
            json.dump(self.schema, f, indent=2, default=str)

    def load_schema(self, path):
        with open(path) as f:
            self.schema = json.load(f)
        return self


if __name__ == "__main__":
    df = pd.read_csv("sample_data/air_quality_sample.csv")
    pp = Preprocessor()
    clean = pp.fit_transform(df)
    print("Detected schema:")
    print(json.dumps(pp.schema, indent=2))
    print(clean[["_aqi_category"]].value_counts())
