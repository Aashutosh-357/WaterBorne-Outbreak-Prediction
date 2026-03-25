"""
model_trainer.py - XGBoost Model Training & Evaluation
=======================================================
Provides functions to train, evaluate, and save the XGBoost
classifier for waterborne disease outbreak prediction.
Can be run as a standalone script to retrain the model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import xgboost as xgb

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import prepare_data, FEATURES

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")


def build_xgboost_model(params: dict = None) -> xgb.XGBClassifier:
    """
    Creates an XGBoost classifier with optimized hyperparameters
    for large-scale waterborne disease data.

    Args:
        params: Optional dict to override default hyperparameters.

    Returns:
        Configured (untrained) XGBClassifier.
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42,
        'tree_method': 'hist',       # Optimized for 1M+ rows
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    if params:
        default_params.update(params)

    return xgb.XGBClassifier(**default_params)


def train_model(X_train, y_train, params: dict = None) -> xgb.XGBClassifier:
    """
    Trains an XGBoost model on the provided training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Optional hyperparameter overrides.

    Returns:
        Trained XGBClassifier.
    """
    print("🤖 Training XGBoost model...")
    model = build_xgboost_model(params)
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the model and returns performance metrics.

    Args:
        model: Trained XGBClassifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict containing accuracy, classification report, confusion matrix, and AUC-ROC.
    """
    print("\n📊 Evaluating model on test data...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Outbreak (1)'])
    cm = confusion_matrix(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    print(f"   🎯 Accuracy : {accuracy * 100:.2f}%")
    print(f"   📈 AUC-ROC  : {auc_roc:.4f}")
    print(f"\n{report}")

    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def save_model(model, path: str = MODEL_PATH):
    """Saves the trained model to a .pkl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"   💾 Model saved to: {path}")


def load_model(path: str = MODEL_PATH) -> xgb.XGBClassifier:
    """Loads a trained model from a .pkl file."""
    model = joblib.load(path)
    print(f"   📂 Model loaded from: {path}")
    return model


def get_feature_importance(model, feature_names: list = None) -> pd.DataFrame:
    """
    Extracts feature importance from the trained model.

    Returns:
        DataFrame sorted by importance (descending).
    """
    if feature_names is None:
        feature_names = FEATURES

    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return fi_df


def full_training_pipeline(test_size: float = 0.2, save: bool = True):
    """
    End-to-end training pipeline:
    1. Load & preprocess data
    2. Split into train/test
    3. Train model
    4. Evaluate
    5. Save model

    Returns:
        Tuple of (model, metrics)
    """
    print("\n" + "=" * 60)
    print("  FULL MODEL TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Data preparation
    X, y, encoders = prepare_data(save_processed=True)

    # Step 2: Split
    print(f"\n✂️  Splitting data ({int((1 - test_size) * 100)}/{int(test_size * 100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"   Training: {len(X_train):,} records | Testing: {len(X_test):,} records")

    # Step 3: Train
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Feature importance
    fi_df = get_feature_importance(model, list(X.columns))
    print("\n🏆 Feature Importance Ranking:")
    for _, row in fi_df.iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"   {row['Feature']:.<30s} {bar} {row['Importance']:.4f}")

    # Step 6: Save
    if save:
        save_model(model)

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60 + "\n")

    return model, metrics


# ──────────────────────────────────────────────────────────────
# Run as standalone script to retrain the model
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, metrics = full_training_pipeline()
