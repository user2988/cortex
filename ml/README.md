# Cortex ML

Weekly ML pipeline that learns what nutrition and activity inputs predict
your best recovery, sleep, and cardiovascular outcomes — then recommends
the targets most likely to improve your wellness score.

---

## What it does

Each Sunday it runs four stages in sequence:

1. **Loads your data** from the existing biometrics and nutrition tables
2. **Computes a daily wellness score** (0–100) as a weighted composite of
   your sleep and cardiovascular metrics, normalised entirely to your own
   personal range — no population benchmarks
3. **Trains an XGBoost model** to predict that score from yesterday's
   nutrition and activity inputs
4. **Optimises your activity targets** to find the combination the model
   predicts will maximise your wellness score, within safe constraints

Results are written to three new database tables and are never mixed with
the existing Fitbit / Cronometer pipeline.

---

## Files

| File | Purpose |
|---|---|
| `data_builder.py` | Loads and transforms data; lags inputs by 1 day |
| `wellness_score.py` | Computes the composite 0–100 wellness score |
| `model_trainer.py` | Trains and evaluates the XGBoost model |
| `stack_optimiser.py` | Finds optimal activity targets via differential evolution |
| `pipeline.py` | Orchestrates all four stages; entry point for GitHub Actions |
| `models/` | Saved model files (created automatically on first run) |

---

## Database tables created

| Table | Contents |
|---|---|
| `ml_model_runs` | One row per training run — metrics, confidence tier, top features |
| `ml_recommendations` | One row per weekly recommendation — current vs predicted wellness, full JSONB output |
| `ml_pipeline_log` | One row per pipeline execution — status, duration, any error messages |

---

## Confidence tiers

The model reports a confidence tier based on how many complete training rows are available:

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

# Individual components
DATABASE_URL=your_url python ml/data_builder.py
DATABASE_URL=your_url python ml/wellness_score.py
DATABASE_URL=your_url python ml/model_trainer.py
DATABASE_URL=your_url python ml/stack_optimiser.py
```

---

## Constraints

- Activity recommendations are capped at **40 % above your 30-day average**
- All activity bounds are hardcoded absolute limits that cannot be exceeded
- Only features the model considers important (above mean importance) are varied
- Supplements will be optimised once the supplements table is added

---

## Adding supplements (future)

When the supplements table exists:
1. Add a supplement pivot step to `data_builder.py`
2. Add supplement bounds to `stack_optimiser.SUPPLEMENT_BOUNDS`
3. Filter to supplements logged ≥ 10 times before optimising
4. Populate the `recommendations["supplements"]` list in the output
