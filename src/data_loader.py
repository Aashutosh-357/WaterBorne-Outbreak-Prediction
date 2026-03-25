"""
data_loader.py - Data Loading & Preprocessing Pipeline
=======================================================
Handles loading the raw waterborne disease dataset, cleaning it,
creating the target variable, encoding categorical features,
and saving the processed data for the Streamlit app.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


# ──────────────────────────────────────────────────────────────
# PATHS (Relative to project root)
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "waterborne_disease.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "waterborne_processed.csv")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# Features used for training & prediction
FEATURES = [
    'is_urban', 'population_density', 'water_source', 'water_treatment',
    'ph', 'avg_temperature_c', 'avg_rainfall_mm', 'avg_humidity_pct', 'flooding'
]

CATEGORICAL_COLS = ['water_source', 'water_treatment']


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw CSV dataset.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with all raw columns.
    """
    print(f"📂 Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} columns.")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary target: 0 = No Disease (Safe), 1 = Outbreak Risk.
    """
    df = df.copy()
    df['Outbreak_Risk'] = np.where(df['disease'] == 'No_Disease', 0, 1)

    safe_count = (df['Outbreak_Risk'] == 0).sum()
    outbreak_count = (df['Outbreak_Risk'] == 1).sum()
    print(f"   🏷️  Target created → Safe: {safe_count:,} | Outbreak: {outbreak_count:,}")
    return df


def encode_categorical_features(df: pd.DataFrame, fit: bool = True):
    """
    Encodes categorical columns (water_source, water_treatment) into integers.

    Args:
        df: DataFrame with the feature columns.
        fit: If True, fits new encoders. If False, loads saved encoders.

    Returns:
        Tuple of (encoded DataFrame, dict of LabelEncoders).
    """
    df = df.copy()
    encoders = {}

    if fit:
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"   🔢 Encoded '{col}' → {len(le.classes_)} unique categories")

        # Save encoders for later use in the Streamlit app
        os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)
        print(f"   💾 Encoders saved to: {ENCODERS_PATH}")
    else:
        # Load pre-fitted encoders
        encoders = joblib.load(ENCODERS_PATH)
        for col in CATEGORICAL_COLS:
            df[col] = encoders[col].transform(df[col].astype(str))

    return df, encoders


def prepare_data(save_processed: bool = True):
    """
    Full data preparation pipeline:
    1. Load raw data
    2. Create target variable
    3. Extract features
    4. Encode categorical columns
    5. Save processed data

    Returns:
        Tuple of (X, y, encoders)
    """
    print("\n" + "=" * 60)
    print("  DATA LOADING & PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Target variable
    df = create_target_variable(df)

    # Step 3: Extract features + target
    X = df[FEATURES].copy()
    y = df['Outbreak_Risk']

    # Step 4: Encode
    X, encoders = encode_categorical_features(X, fit=True)

    # Step 5: Save processed data
    if save_processed:
        processed = X.copy()
        processed['Outbreak_Risk'] = y.values
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        processed.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"   💾 Processed data saved to: {PROCESSED_DATA_PATH}")

    print("=" * 60)
    print(f"  ✅ Pipeline complete! X shape: {X.shape}, y shape: {y.shape}")
    print("=" * 60 + "\n")

    return X, y, encoders


def load_processed_data(path: str = PROCESSED_DATA_PATH):
    """
    Loads the pre-processed CSV (used by the Streamlit app for speed).
    """
    df = pd.read_csv(path)
    X = df.drop('Outbreak_Risk', axis=1)
    y = df['Outbreak_Risk']
    return X, y


# ──────────────────────────────────────────────────────────────
# Run as standalone script
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, encoders = prepare_data(save_processed=True)
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
