# Cortex ML

Weekly ML pipeline that learns which previous-day inputs (nutrition,
activity, sleep, recovery) predict the user's morning blood pressure.

---

## Targets

Two regression targets, trained as separate XGBoost models:

| Target | Definition |
|---|---|
| `AM_systolic`  | Average of the two morning systolic readings (≈ 8 am) |
| `AM_diastolic` | Average of the two morning diastolic readings (≈ 8 am) |

AM BP is driven entirely by the **previous day's** nutrition/activity and
the **previous night's** sleep/recovery. Never use same-day activity or
HRV as inputs — they are not available at reading time.

---

## What it does

Each Sunday the pipeline runs:

1. **Loads your data** from the biometrics and nutrition tables, applies
   1-day lags to inputs, and returns a clean modelling DataFrame.
2. *(Coming next)* **Trains one XGBoost model per BP target** (systolic
   and diastolic) against that DataFrame once the `bp_readings` table
   is live. Model metadata and feature importances are written to
   `ml_model_runs`.

The wellness-score stack (composite 0–100 score, outcome tracker,
stack optimiser) has been removed. BP is the only target now.

---

## Files

| File | Purpose |
|---|---|
| `data_builder.py`  | Loads and transforms data; lags inputs by 1 day |
| `model_trainer.py` | Generic XGBoost trainer — call once per target |
| `pipeline.py`      | Orchestrates stages; entry point for GitHub Actions |
| `models/`          | Saved model files (created automatically on first run) |

---

## Database tables

| Table | Contents |
|---|---|
| `ml_model_runs`   | One row per training run — target, metrics, confidence tier, top features |
| `ml_pipeline_log` | One row per pipeline execution — status, duration, any error messages |

`ml_recommendations` and `ml_recommendation_outcomes` (wellness-era)
are dropped by `database/alter.py` on next migration.

---

## Confidence tiers

| Tier | Rows | Behaviour |
|---|---|---|
| `insufficient` | < 30 | Exits without training |
| `low` | 30–59 | Trains; treat results as directional only |
| `moderate` | 60–89 | Trains; reasonable reliability |
| `high` | 90+ | Full confidence |

---

## Running manually

```bash
# Full pipeline
DATABASE_URL=your_url python -m ml.pipeline

# Data builder alone
DATABASE_URL=your_url python ml/data_builder.py
```
