"""
Unit tests for BP-related logic in analysis.py and ml/bp_target.py.

Tests are split into:
  - MAP computation (pure functions, no DB)
  - BP daily aggregates (pure transform, no DB)
  - save_bp_log / load_bp_logs (DB mocked via monkeypatch)
  - load_pipeline_log / load_model_runs error paths (DB mocked)
  - ml.bp_target.compute alignment (mocked DB)
"""

import sys
import os
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is importable without a DATABASE_URL requirement at import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub DATABASE_URL so psycopg2.connect never fires at module import
os.environ.setdefault("DATABASE_URL", "postgresql://stub:stub@stub/stub")

import analysis  # noqa: E402  (after env var set)
from ml import bp_target  # noqa: E402


# ─────────────────────────────────────────────────────────────
# MAP computation — pure function, no DB
# ─────────────────────────────────────────────────────────────

class TestMapValue:
    def test_typical_values(self):
        # 120/80 → (120 + 160) / 3 = 93.33…
        result = analysis._map_value(120, 80)
        assert abs(result - 93.333) < 0.001

    def test_systolic_none(self):
        assert analysis._map_value(None, 80) is None

    def test_diastolic_none(self):
        assert analysis._map_value(120, None) is None

    def test_both_none(self):
        assert analysis._map_value(None, None) is None

    def test_string_numerics(self):
        # Should coerce strings to float
        result = analysis._map_value("120", "80")
        assert abs(result - 93.333) < 0.001

    def test_non_numeric_string(self):
        assert analysis._map_value("abc", 80) is None

    def test_zero_diastolic(self):
        result = analysis._map_value(60, 0)
        assert abs(result - 20.0) < 0.001

    def test_bp_target_map_value_matches(self):
        # ml/bp_target._map_value must agree with analysis._map_value
        for sys_v, dia_v in [(120, 80), (140, 90), (100, 70)]:
            assert bp_target._map_value(sys_v, dia_v) == analysis._map_value(sys_v, dia_v)


# ─────────────────────────────────────────────────────────────
# BP daily aggregates — pure transform, no DB
# ─────────────────────────────────────────────────────────────

def _make_raw_bp(*rows) -> pd.DataFrame:
    """Build a raw blood_pressure_logs-shaped DataFrame from row tuples:
    (date, session, r1_sys, r1_dia, r2_sys, r2_dia)
    """
    cols = ["date", "session",
            "reading_1_systolic", "reading_1_diastolic",
            "reading_2_systolic", "reading_2_diastolic", "logged_at"]
    data = [(*r, datetime.now()) for r in rows]
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestBpDailyAggregates:
    def _run(self, raw_df):
        with patch.object(analysis, "load_bp_logs", return_value=raw_df):
            return analysis.load_bp_daily_aggregates()

    def test_single_pm_reading(self):
        raw = _make_raw_bp(("2024-01-01", "PM", 120, 80, None, None))
        agg = self._run(raw)
        assert len(agg) == 1
        expected_map = (120 + 2 * 80) / 3
        assert abs(agg.iloc[0]["pm_map"] - expected_map) < 0.01
        assert abs(agg.iloc[0]["map_mmhg"] - expected_map) < 0.01

    def test_two_readings_averaged(self):
        # Two PM readings; MAP of each averaged
        raw = _make_raw_bp(("2024-01-01", "PM", 120, 80, 122, 82))
        agg = self._run(raw)
        map1 = (120 + 2 * 80) / 3
        map2 = (122 + 2 * 82) / 3
        expected = (map1 + map2) / 2
        assert abs(agg.iloc[0]["pm_map"] - expected) < 0.01

    def test_am_and_pm_daily_mean(self):
        raw = _make_raw_bp(
            ("2024-01-02", "AM", 118, 78, None, None),
            ("2024-01-02", "PM", 122, 82, None, None),
        )
        agg = self._run(raw)
        assert len(agg) == 1
        am_map = (118 + 2 * 78) / 3
        pm_map = (122 + 2 * 82) / 3
        expected_daily = (am_map + pm_map) / 2
        assert abs(agg.iloc[0]["map_mmhg"] - expected_daily) < 0.01
        assert agg.iloc[0]["n_sessions"] == 2

    def test_empty_input_returns_empty_df(self):
        agg = self._run(pd.DataFrame())
        assert agg.empty
        assert list(agg.columns) == ["date", "map_mmhg", "n_sessions", "am_map", "pm_map"]

    def test_multiple_days_sorted_descending(self):
        raw = _make_raw_bp(
            ("2024-01-01", "PM", 120, 80, None, None),
            ("2024-01-03", "PM", 130, 85, None, None),
            ("2024-01-02", "PM", 125, 82, None, None),
        )
        agg = self._run(raw)
        assert len(agg) == 3
        dates = agg["date"].tolist()
        assert dates == sorted(dates, reverse=True)


# ─────────────────────────────────────────────────────────────
# save_bp_log — DB mocked
# ─────────────────────────────────────────────────────────────

class TestSaveBpLog:
    def _mock_conn(self):
        cur = MagicMock()
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return conn, cur

    def test_upsert_called_with_correct_args(self):
        conn, cur = self._mock_conn()
        with patch("analysis.psycopg2.connect", return_value=conn):
            analysis.save_bp_log(
                date=date(2024, 1, 1),
                session="PM",
                r1_sys=120, r1_dia=80,
                r2_sys=122, r2_dia=82,
            )
        args = cur.execute.call_args[0]
        sql, params = args[0], args[1]
        assert "ON CONFLICT" in sql
        assert params == (date(2024, 1, 1), "PM", 120, 80, 122, 82)

    def test_upsert_with_optional_reading_none(self):
        conn, cur = self._mock_conn()
        with patch("analysis.psycopg2.connect", return_value=conn):
            analysis.save_bp_log(
                date=date(2024, 1, 2),
                session="AM",
                r1_sys=115, r1_dia=75,
                r2_sys=None, r2_dia=None,
            )
        args = cur.execute.call_args[0]
        params = args[1]
        assert params[4] is None
        assert params[5] is None


# ─────────────────────────────────────────────────────────────
# load_pipeline_log / load_model_runs — error path smoke tests
# ─────────────────────────────────────────────────────────────

class TestLoaderErrorPaths:
    def _conn_that_raises(self, exc_msg="table does not exist"):
        cur = MagicMock()
        cur.execute.side_effect = Exception(exc_msg)
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return conn

    def test_pipeline_log_returns_tuple(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.description = [("run_at",), ("status",), ("duration_sec",), ("stage",), ("error_message",)]
        cur.fetchall.return_value = []
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        with patch("analysis.psycopg2.connect", return_value=conn):
            result = analysis.load_pipeline_log()
        assert isinstance(result, tuple)
        df, err = result
        assert isinstance(df, pd.DataFrame)
        assert err is None

    def test_pipeline_log_db_error_returns_message(self):
        conn = self._conn_that_raises("relation ml_pipeline_log does not exist")
        with patch("analysis.psycopg2.connect", return_value=conn):
            df, err = analysis.load_pipeline_log()
        assert df.empty
        assert err is not None
        assert "DB error" in err

    def test_model_runs_returns_tuple(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.description = [
            ("id",), ("run_at",), ("confidence_tier",), ("n_rows",), ("n_features",),
            ("train_r2",), ("test_r2",), ("test_mae",), ("test_rmse",), ("top_features",), ("model_path",),
        ]
        cur.fetchall.return_value = []
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        with patch("analysis.psycopg2.connect", return_value=conn):
            result = analysis.load_model_runs()
        assert isinstance(result, tuple)
        df, err = result
        assert isinstance(df, pd.DataFrame)
        assert err is None

    def test_model_runs_db_error_returns_message(self):
        conn = self._conn_that_raises("relation ml_model_runs does not exist")
        with patch("analysis.psycopg2.connect", return_value=conn):
            df, err = analysis.load_model_runs()
        assert df.empty
        assert err is not None
        assert "DB error" in err


# ─────────────────────────────────────────────────────────────
# ml.bp_target.compute — alignment and reindex
# ─────────────────────────────────────────────────────────────

class TestBpTargetCompute:
    def _mock_pm_series(self, rows: dict[str, float]) -> pd.Series:
        idx = pd.to_datetime(list(rows.keys()))
        return pd.Series(list(rows.values()), index=idx, name="pm_map")

    def test_aligned_to_df_index(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"x": [1, 2, 3]}, index=dates)
        mock_series = self._mock_pm_series({"2024-01-01": 93.0, "2024-01-03": 95.0})
        with patch.object(bp_target, "_load_pm_map", return_value=mock_series):
            result = bp_target.compute(df)
        assert result.index.tolist() == dates.tolist()
        assert abs(result["2024-01-01"] - 93.0) < 0.01
        assert pd.isna(result["2024-01-02"])  # no reading on this day
        assert abs(result["2024-01-03"] - 95.0) < 0.01

    def test_name_is_pm_map(self):
        dates = pd.to_datetime(["2024-01-01"])
        df = pd.DataFrame({"x": [1]}, index=dates)
        mock_series = self._mock_pm_series({"2024-01-01": 90.0})
        with patch.object(bp_target, "_load_pm_map", return_value=mock_series):
            result = bp_target.compute(df)
        assert result.name == "pm_map"

    def test_no_pm_readings_all_nan(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        df = pd.DataFrame({"x": [1, 2]}, index=dates)
        empty_series = pd.Series(dtype=float, name="pm_map")
        with patch.object(bp_target, "_load_pm_map", return_value=empty_series):
            result = bp_target.compute(df)
        assert result.isna().all()
        assert len(result) == len(df)


# ─────────────────────────────────────────────────────────────
# Migration column presence — schema expectations
# ─────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = {
    "blood_pressure_logs": {
        "date", "session",
        "reading_1_systolic", "reading_1_diastolic",
        "reading_2_systolic", "reading_2_diastolic",
        "logged_at",
    },
    "ml_recommendations": {
        "current_map_avg", "predicted_map",
    },
    "ml_recommendation_outcomes": {
        "map_before_avg", "map_after_avg", "map_delta",
    },
}


class TestSchemaColumns:
    """
    Smoke tests: if a real DATABASE_URL is configured, connect and verify
    that the expected columns exist in each table. Skipped when DATABASE_URL
    is the stub placeholder or when psycopg2 cannot connect.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_real_db(self):
        db_url = os.environ.get("DATABASE_URL", "")
        if "stub" in db_url:
            pytest.skip("No real DATABASE_URL configured")

    def _get_columns(self, table: str) -> set[str]:
        import psycopg2
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s", (table,)
                )
                return {row[0] for row in cur.fetchall()}
        finally:
            conn.close()

    @pytest.mark.parametrize("table,cols", list(EXPECTED_COLUMNS.items()))
    def test_table_has_expected_columns(self, table, cols):
        actual = self._get_columns(table)
        missing = cols - actual
        assert not missing, f"Table '{table}' is missing columns: {missing}"
