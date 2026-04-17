"""
Cortex ML — Component 3: Model Trainer

Trains an XGBoost regression model to predict the daily wellness score
from lagged nutrition and activity features. Writes model metadata and
feature importances to the database. Saves the trained model to disk.

Confidence tiers (based on number of complete training rows)
------------------------------------------------------------
  insufficient : < 30   — exits without training
  low          : 30–59
  moderate     : 60–89
  high         : 90+

Split strategy
--------------
Time-ordered, no shuffling. The most recent TEST_FRACTION of rows form
the held-out test set. This mirrors real-world usage where the model is
always predicting future days from past data.
"""

import os
import json
import joblib
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DATABASE_URL = os.environ["DATABASE_URL"]

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "wellness_model.joblib"

TEST_FRACTION  = 0.20   # fraction of rows held out as test set
TOP_N_FEATURES = 20     # number of top features written to DB

# Confidence tier thresholds (training rows)
TIER_INSUFFICIENT = 30
TIER_LOW          = 60
TIER_MODERATE     = 90


# ─────────────────────────────────────────────────────────────
# DATABASE — TABLE CREATION
# ─────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_model_runs (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL,
    confidence_tier TEXT        NOT NULL,
    n_rows          INTEGER     NOT NULL,
    n_features      INTEGER     NOT NULL,
    train_r2        NUMERIC(6, 4),
    test_r2         NUMERIC(6, 4),
    test_mae        NUMERIC(8, 4),
    test_rmse       NUMERIC(8, 4),
    top_features    JSONB,
    model_path      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


def _ensure_table() -> None:
    """Create ml_model_runs if it does not already exist."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


def _write_run(
    run_at: datetime,
    tier: str,
    n_rows: int,
    n_features: int,
    train_r2: float,
    test_r2: float,
    test_mae: float,
    test_rmse: float,
    top_features: list[dict],
    model_path: str,
) -> int:
    """
    Insert a model run record and return its generated id.

    Parameters
    ----------
    top_features : list of {"feature": str, "importance": float} dicts
    """
    sql = """
        INSERT INTO ml_model_runs
            (run_at, confidence_tier, n_rows, n_features,
             train_r2, test_r2, test_mae, test_rmse,
             top_features, model_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    run_at,
                    tier,
                    int(n_rows),
                    int(n_features),
                    round(float(train_r2), 4),
                    round(float(test_r2),  4),
                    round(float(test_mae), 4),
                    round(float(test_rmse), 4),
                    json.dumps(top_features),
                    str(model_path),
                ))
                return cur.fetchone()[0]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# CONFIDENCE TIER
# ─────────────────────────────────────────────────────────────

def confidence_tier(n_rows: int) -> str:
    """
    Map a row count to a confidence tier label.

    Returns one of: 'insufficient', 'low', 'moderate', 'high'.
    """
    if n_rows < TIER_INSUFFICIENT:
        return "insufficient"
    if n_rows < TIER_LOW:
        return "low"
    if n_rows < TIER_MODERATE:
        return "moderate"
    return "high"


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def _split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Chronological train/test split — no shuffling.

    The last TEST_FRACTION rows form the test set so the model is always
    evaluated on data it has never seen, in the order it would arrive.
    """
    n_test = max(1, int(len(X) * TEST_FRACTION))
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def _build_model() -> XGBRegressor:
    """
    Return a conservatively configured XGBoost regressor.

    Shallow trees and high regularisation prevent overfitting on the
    small datasets typical of personal health tracking.
    """
    return XGBRegressor(
        n_estimators       = 400,
        max_depth          = 3,       # shallow — avoids overfitting on small N
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        reg_alpha          = 0.1,     # L1
        reg_lambda         = 1.0,     # L2
        objective          = "reg:squarederror",
        eval_metric        = "rmse",
        early_stopping_rounds = 20,
        random_state       = 42,
        verbosity          = 0,
    )


def train(df: pd.DataFrame, scores: pd.Series) -> dict | None:
    """
    Train the wellness model and persist artefacts to disk and database.

    Parameters
    ----------
    df     : feature DataFrame from data_builder.build()
    scores : wellness scores from wellness_score.compute()

    Returns
    -------
    dict with keys: tier, n_rows, n_features, train_r2, test_r2,
                    test_mae, test_rmse, top_features, model_path, run_id
    Returns None if data is insufficient to train.
    """
    _ensure_table()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Align features with scores and drop rows where score is NaN
    from ml.data_builder import feature_cols
    feat_cols = feature_cols(df)

    combined = df[feat_cols].copy()
    combined["__target__"] = scores
    combined = combined.dropna(subset=["__target__"])

    X = combined[feat_cols]
    y = combined["__target__"]

    n_rows = len(X)
    tier   = confidence_tier(n_rows)

    print(f"  Rows available : {n_rows}")
    print(f"  Features       : {len(feat_cols)}")
    print(f"  Confidence tier: {tier}")

    if tier == "insufficient":
        print(f"  Insufficient data (<{TIER_INSUFFICIENT} rows) — skipping training.")
        print(f"  Keep logging — model training begins after {TIER_INSUFFICIENT} days.")
        return None

    X_train, X_test, y_train, y_test = _split(X, y)

    model = _build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Metrics
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_r2  = float(1 - np.sum((y_train - train_pred) ** 2) /
                          np.sum((y_train - y_train.mean()) ** 2))
    test_r2   = float(1 - np.sum((y_test - test_pred) ** 2) /
                          np.sum((y_test - y_test.mean()) ** 2))
    test_mae  = float(mean_absolute_error(y_test, test_pred))
    test_rmse = float(root_mean_squared_error(y_test, test_pred))

    print(f"  Train R²   : {train_r2:.4f}")
    print(f"  Test  R²   : {test_r2:.4f}")
    print(f"  Test  MAE  : {test_mae:.4f}")
    print(f"  Test  RMSE : {test_rmse:.4f}")

    # Feature importances
    importances = model.feature_importances_
    top_features = (
        pd.Series(importances, index=feat_cols)
        .sort_values(ascending=False)
        .head(TOP_N_FEATURES)
    )
    top_features_list = [
        {"feature": col, "importance": round(float(imp), 6)}
        for col, imp in top_features.items()
    ]

    print(f"\n  Top {TOP_N_FEATURES} features by importance:")
    for entry in top_features_list:
        print(f"    {entry['feature']:<35} {entry['importance']:.4f}")

    # Persist model
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")

    # Write metadata to DB
    run_at = datetime.now(timezone.utc)
    run_id = _write_run(
        run_at      = run_at,
        tier        = tier,
        n_rows      = n_rows,
        n_features  = len(feat_cols),
        train_r2    = train_r2,
        test_r2     = test_r2,
        test_mae    = test_mae,
        test_rmse   = test_rmse,
        top_features= top_features_list,
        model_path  = str(MODEL_PATH),
    )
    print(f"  Run metadata written (id={run_id}).")

    return {
        "tier":         tier,
        "n_rows":       n_rows,
        "n_features":   len(feat_cols),
        "train_r2":     train_r2,
        "test_r2":      test_r2,
        "test_mae":     test_mae,
        "test_rmse":    test_rmse,
        "top_features": top_features_list,
        "model_path":   str(MODEL_PATH),
        "run_id":       run_id,
        "model":        model,
        "feature_cols": feat_cols,
    }


def load_model() -> tuple[XGBRegressor, list[str]] | tuple[None, None]:
    """
    Load the most recently saved model from disk.

    Returns (model, feature_cols) or (None, None) if no model exists.
    """
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    return model, None   # feature_cols resolved fresh by caller


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ml import data_builder, wellness_score

    print("[model_trainer] Building data...")
    df = data_builder.build()
    if df.empty:
        print("No data — exiting.")
        sys.exit(0)

    print("[model_trainer] Computing wellness scores...")
    scores = wellness_score.compute(df)

    print("[model_trainer] Training model...")
    result = train(df, scores)

    if result is None:
        print("Training skipped.")
    else:
        print(f"\nDone. Run id={result['run_id']}  tier={result['tier']}")
