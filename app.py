"""Cortex — Insights, Experiments & Explorer (v2)"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

import analysis

st.set_page_config(page_title="Cortex", layout="wide", page_icon="📊")
analysis.ensure_schema()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
GRAY   = "#888888"

BRIGHT_GREEN = "rgba(0,230,120,1.0)"
GRAD_MID     = "rgba(120,140,220,0.8)"
GRAD_OLD     = "rgba(120,120,120,0.35)"

ANALYSIS_TYPES = [
    "Pearson Correlation", "Spearman Correlation", "Lagged Correlation",
    "Rolling Average", "30-Day Trend (OLS)", "Multiple OLS Regression",
    "Anomaly Detection", "Forecast (7-Day)", "Decomposition",
]
SINGLE_VAR = {"30-Day Trend (OLS)", "Anomaly Detection", "Forecast (7-Day)", "Decomposition"}
MULTI_PRED = {"Multiple OLS Regression"}

# ── Variable taxonomy ────────────────────────────────────────
# Keys: "Category  ·  Subcategory"  Values: list of column names (shown raw)

VAR_A_TREE = {
    "Activity  ·  Volume":       ["steps", "distance_km", "calories_burned",
                                   "sedentary_min", "lightly_active_min"],
    "Activity  ·  Intensity":    ["active_zone_min", "very_active_min",
                                   "fairly_active_min"],
    "Activity  ·  Zones":        ["time_in_fat_burn_min", "time_in_cardio_min",
                                   "time_in_peak_min"],
}

VAR_B_TREE = {
    "Sleep  ·  Primary":              ["sleep_efficiency_pct", "sleep_duration_min"],
    "Sleep  ·  Architecture":         ["deep_sleep_min", "rem_sleep_min",
                                        "light_sleep_min", "awake_min"],
    "Sleep  ·  Behavioural":          ["time_in_bed_min"],
    "Cardiovascular  ·  Heart":       ["hrv_ms", "hrv_deep_rmssd", "rhr_bpm"],
    "Cardiovascular  ·  Oxygen":      ["spo2_avg_pct", "spo2_min_pct", "spo2_max_pct"],
    "Cardiovascular  ·  Respiratory": ["respiratory_rate", "vo2_max"],
}

def _flat_cols(tree):
    """All column names across a tree."""
    out = []
    for cols in tree.values(): out.extend(cols)
    return out

VAR_A_COLS = _flat_cols(VAR_A_TREE)
VAR_B_COLS = _flat_cols(VAR_B_TREE)

def col_label(col):
    return analysis.COL_LABELS.get(col, col)

A_CATS = ["Activity"]
A_SUBS = {
    "Activity":  ["Volume", "Intensity", "Zones"],
}
B_CATS = ["Sleep", "Cardiovascular"]
B_SUBS = {
    "Sleep":          ["Primary", "Architecture", "Behavioural"],
    "Cardiovascular": ["Heart", "Oxygen", "Respiratory"],
}

def _picker(panel_id, cats, subs_map, tree, header):
    """Category → Subcategory → Variable. Returns chosen column name."""
    st.markdown(f"**{header}**")
    cat = st.selectbox("", cats,          key=f"{panel_id}_cat", label_visibility="collapsed")
    sub = st.selectbox("", subs_map[cat], key=f"{panel_id}_sub_{cat}", label_visibility="collapsed")
    grp = f"{cat}  ·  {sub}"
    return st.selectbox("", tree[grp],    key=f"{panel_id}_var_{grp}", label_visibility="collapsed")

TREND_METRICS = [
    ("HRV",             "hrv_ms",           "ms",  "bio", True),
    ("RHR",             "rhr_bpm",          "bpm", "bio", False),
    ("Sleep Efficiency","sleep_efficiency_pct","%", "bio", True),
]

# ─────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_data(days):
    return analysis.load_data(days if days > 0 else None)

@st.cache_data(ttl=60)
def get_findings():
    return analysis.load_findings()

@st.cache_data(ttl=60)
def get_experiments():
    return analysis.load_experiments()



@st.cache_data(ttl=300)
def get_daily_scores():
    return analysis.load_daily_scores(days=90)


@st.cache_data(ttl=300)
def get_score_recommendations():
    return analysis.load_score_recommendations()


def bust_cache():
    get_data.clear(); get_findings.clear()
    get_experiments.clear()
    get_daily_scores.clear(); get_score_recommendations.clear()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def sparkline(series: pd.Series, color: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=list(range(len(series))), y=series.values,
        mode="lines", line=dict(color=color, width=2),
    ))
    fig.update_layout(height=45, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def r2_color(r2):
    if r2 >= 0.50: return GREEN
    if r2 >= 0.30: return ORANGE
    return GRAY

def scatter_ols(a, b, coef, intercept, x_label, y_label,
                color=BLUE, extra_traces=None) -> go.Figure:
    x_range = np.linspace(float(a.min()), float(a.max()), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a.values, y=b.values, mode="markers",
        text=a.index.strftime("%Y-%m-%d"),
        hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
        marker=dict(color=color, size=7, opacity=0.8),
    ))
    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)
    fig.add_trace(go.Scatter(
        x=x_range, y=coef * x_range + intercept, mode="lines",
        line=dict(color=ORANGE, dash="dash", width=2),
        showlegend=False,
    ))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label,
                      height=450, margin=dict(t=20))
    return fig

def stat_bar(r2=None, p=None, coef=None, n=None, label=None, extra=None):
    cols = st.columns(4 + (len(extra) if extra else 0))
    i = 0
    if r2   is not None: cols[i].metric("R²",          f"{r2:.4f}");   i += 1
    if p    is not None: cols[i].metric("p-value",      f"{p:.4f}");    i += 1
    if coef is not None: cols[i].metric("Coefficient",  f"{coef:.4f}"); i += 1
    if n    is not None: cols[i].metric("Data Points",  n);             i += 1
    if extra:
        for lbl, val in extra: cols[i].metric(lbl, val); i += 1
    if label: st.markdown(f"**{label}**")

# ─────────────────────────────────────────────────────────────
# SIDEBAR + PAGE ROUTING
# ─────────────────────────────────────────────────────────────

st.sidebar.title("Cortex")
st.sidebar.caption("v2 — personal analytics")
st.sidebar.divider()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "exp_detail_id" not in st.session_state:
    st.session_state.exp_detail_id = None
if "saved_view_id" not in st.session_state:
    st.session_state.saved_view_id = None

page = st.sidebar.radio("Navigation", ["Dashboard", "Insights", "Experiments", "Explorer"],
                         key="page", label_visibility="collapsed")

if page != "Experiments":
    st.session_state.exp_detail_id = None

if st.sidebar.button("↺ Refresh", use_container_width=True):
    bust_cache()
    st.rerun()

# ─────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────

if page == "Dashboard":

    # ── CSS ─────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppViewContainer"] { background: #0D1117; }
    [data-testid="stSidebar"] { background: #161B22 !important; border-right: 1px solid #21262D; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .dash-kpi-card {
        background: #161B22; border: 1px solid #21262D; border-radius: 6px;
        padding: 14px 12px 12px; text-align: center; min-height: 86px;
        transition: border-color 0.15s;
    }
    .dash-kpi-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 8.5px; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase; color: #484F58; margin-bottom: 6px;
    }
    .dash-kpi-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 26px;
        font-weight: 300; color: #E6EDF3; line-height: 1.1;
    }
    .dash-kpi-unit { font-size: 10px; color: #6E7681; margin-left: 2px; }
    .dash-kpi-delta { font-family: 'Inter', sans-serif; font-size: 10px; margin-top: 4px; }
    .dash-section {
        font-family: 'Inter', sans-serif; font-size: 9.5px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #6E7681;
        margin: 28px 0 12px; padding: 0 0 10px 10px;
        border-bottom: 1px solid #21262D; border-left: 2px solid #30363D;
    }
    .dash-chart-label {
        font-family: 'Inter', sans-serif; font-size: 11.5px;
        font-weight: 600; color: #8B949E; margin-bottom: 3px; letter-spacing: -0.01em;
    }
    .dash-stat-row {
        font-family: 'IBM Plex Mono', monospace; font-size: 9px;
        color: #484F58; font-weight: 400;
    }
    .finding-row {
        background: #161B22; border: 1px solid #21262D; border-radius: 6px;
        padding: 9px 12px; margin-bottom: 5px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .exp-card {
        background: #161B22; border: 1px solid #21262D;
        border-radius: 6px; padding: 10px 12px; margin-bottom: 6px;
    }
    .empty-panel {
        background: #161B22; border: 1px solid #21262D; border-radius: 6px;
        padding: 28px; text-align: center;
        color: #484F58; font-size: 12px; font-family: 'Inter', sans-serif;
    }
    /* ── Section sliders ── */
    .st-key-sleep_days  [data-testid="stSlider"],
    .st-key-cv_range    [data-testid="stSlider"],
    .st-key-act_days    [data-testid="stSlider"] { padding-top: 6px; }
    .st-key-sleep_days  p,
    .st-key-cv_range    p,
    .st-key-act_days    p {
        font-family: 'IBM Plex Mono', monospace; font-size: 9px;
        font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase;
        color: #6E7681; margin-bottom: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Data ────────────────────────────────────────────────
    scores      = get_daily_scores()
    df_all      = get_data(90)
    findings_df = get_findings()
    exps_df     = get_experiments()

    # Build a date-indexed view of daily_scores for fallback trend charts.
    # daily_scores has hrv_ms, rhr_bpm, spo2_avg_pct, sleep_duration_min
    # denormalised into it, so CVD/sleep charts work even when biometrics is sparse.
    if not scores.empty and "date" in scores.columns:
        _sc_idx = scores.set_index("date").sort_index()
    else:
        _sc_idx = pd.DataFrame()

    # Merge: prefer biometrics when available, fall back to daily_scores columns
    def _get_series(col):
        """Return the best available time series for a column."""
        if not df_all.empty and col in df_all.columns:
            s = df_all[col].dropna()
            if len(s) >= 3:
                return s
        if not _sc_idx.empty and col in _sc_idx.columns:
            s = _sc_idx[col].dropna()
            if len(s) >= 3:
                return s
        return pd.Series(dtype=float)

    # ── Plotly base layout ──────────────────────────────────
    _CL = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#8B949E"),
        margin=dict(l=6, r=6, t=36, b=6),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="#30363D",
                   tickcolor="#30363D", zeroline=False,
                   tickfont=dict(size=9, family="IBM Plex Mono, monospace")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="#30363D",
                   tickcolor="#30363D", zeroline=False,
                   tickfont=dict(size=9, family="IBM Plex Mono, monospace")),
        hoverlabel=dict(bgcolor="#1C2230", bordercolor="#30363D",
                        font=dict(family="Inter", size=11, color="#E6EDF3")),
    )
    _CFG = {"displayModeBar": False}

    # ── Helpers ─────────────────────────────────────────────
    def _windowed(col, days):
        """Return the best available series for col, trimmed to the last `days` days."""
        s = _get_series(col).sort_index()
        if s.empty:
            return s
        return s[s.index >= s.index.max() - pd.Timedelta(days=days - 1)]

    def _latest(col):
        s = _get_series(col).sort_index()
        return float(s.iloc[-1]) if len(s) else None

    def _delta(col):
        s = _get_series(col).sort_index()
        if len(s) < 2: return None, "#484F58"
        d = float(s.iloc[-1]) - float(s.iloc[-2])
        return d, ("#10B981" if d > 0 else "#EF4444" if d < 0 else "#484F58")

    def _score_color(v):
        if v is None: return "#484F58"
        return "#10B981" if v >= 70 else "#F59E0B" if v >= 45 else "#EF4444"

    def _fmt(v):
        if v is None: return "—"
        if v >= 10000: return f"{v:,.0f}"
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"

    def _metric_bg(col, val):
        if val is None: return None
        if col == "hrv_ms":
            return "#10B981" if val >= 40 else "#F59E0B" if val >= 25 else "#EF4444"
        if col == "rhr_bpm":
            return "#10B981" if val <= 60 else "#F59E0B" if val <= 75 else "#EF4444"
        if col == "spo2_avg_pct":
            return "#10B981" if val >= 95 else "#F59E0B" if val >= 90 else "#EF4444"
        if col == "steps":
            return "#10B981" if val >= 10000 else "#F59E0B" if val >= 5000 else "#EF4444"
        if col == "active_zone_min":
            return "#10B981" if val >= 30 else "#F59E0B" if val >= 15 else "#EF4444"
        return None

    def _kpi(label, value, unit="", delta=None, dcolor="#484F58", vcolor=None, bg=None):
        vc  = vcolor or "#E6EDF3"
        dh  = ""
        if delta is not None:
            arr = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            dh  = f"<div class='dash-kpi-delta' style='color:{dcolor}'>{arr} {abs(delta):.1f}</div>"
        top = f"border-top:2px solid {bg};" if bg else ""
        return (f"<div class='dash-kpi-card' style='{top}'>"
                f"<div class='dash-kpi-label'>{label}</div>"
                f"<div class='dash-kpi-value' style='color:{vc}'>{_fmt(value)}"
                f"<span class='dash-kpi-unit'>{unit}</span></div>{dh}</div>")

    def _section(label):
        st.markdown(f"<div class='dash-section'>{label}</div>", unsafe_allow_html=True)

    def _chart_label(title, s=None, n=None):
        stat = ""
        if s is not None and len(s.dropna()) > 0:
            c = s.dropna().sort_index()
            n = len(c)
            stat = (f"&ensp;<span class='dash-stat-row'>"
                    f"now {c.iloc[-1]:.1f} &middot; "
                    f"min {c.min():.1f} &middot; "
                    f"avg {c.mean():.1f} &middot; "
                    f"max {c.max():.1f}</span>")
        nd = f"&ensp;<span class='dash-stat-row'>({n}d)</span>" if n is not None else ""
        st.markdown(f"<div class='dash-chart-label'>{title}{nd}{stat}</div>",
                    unsafe_allow_html=True)

    def _trend(series, color, fill, height=190, ref=None, rlabel=""):
        roll   = series.rolling(7, min_periods=3).mean()
        mean_v = series.mean()
        std_v  = series.std()
        fig    = go.Figure()
        # Raw daily values — visible dots with a subtle outline ring
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="markers",
            marker=dict(
                color=color, size=6, opacity=0.55,
                line=dict(color=color, width=0),
            ),
            showlegend=False,
            hovertemplate="%{x|%b %-d}: %{y:.1f}<extra></extra>",
        ))
        # 7-day rolling average — clean line, no per-point markers
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values, mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=fill,
            showlegend=False,
            hovertemplate="%{x|%b %-d} (7d avg): %{y:.1f}<extra></extra>",
        ))
        if std_v > 0 and len(series) >= 10:
            _anom = series[(series - mean_v).abs() > 2 * std_v]
            if len(_anom):
                fig.add_trace(go.Scatter(
                    x=_anom.index, y=_anom.values, mode="markers",
                    marker=dict(color="#F59E0B", size=9, symbol="circle-open",
                                line=dict(width=2, color="#F59E0B")),
                    showlegend=False,
                    hovertemplate="%{x|%b %-d}: %{y:.1f} ⚠<extra></extra>",
                ))
        if ref is not None:
            fig.add_hline(y=ref, line_dash="dot", line_color="#484F58", line_width=1,
                          annotation_text=rlabel, annotation_font_color="#6E7681",
                          annotation_font_size=9)
        fig.update_layout(**_CL, height=height)
        return fig

    # ── HEADER ──────────────────────────────────────────────
    today_str = pd.Timestamp.today().strftime("%A, %b %-d, %Y")
    if not df_all.empty:
        _hrs = (pd.Timestamp.now() - df_all.index.max()).total_seconds() / 3600
        _sc  = "#10B981" if _hrs < 6 else "#F59E0B" if _hrs < 24 else "#EF4444"
        _st  = f"Synced {_hrs:.0f}h ago" if _hrs < 24 else f"Stale — {_hrs/24:.0f}d ago"
    else:
        _sc, _st = "#484F58", "No data"

    _hc1, _hc3 = st.columns([6, 2])
    _hc1.markdown(
        "<span style='font-family:Inter;font-size:22px;font-weight:600;"
        "color:#E6EDF3;letter-spacing:-0.02em'>Cortex</span>"
        f"<span style='font-family:Inter;font-size:13px;color:#484F58;"
        f"margin-left:12px'>{today_str}</span>",
        unsafe_allow_html=True,
    )
    _hc3.markdown(
        f"<div style='text-align:right;padding-top:4px'>"
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        f"color:{_sc}'>● {_st}</span></div>",
        unsafe_allow_html=True,
    )

    # ── DATA STATUS BAR ─────────────────────────────────────
    _n_bio    = len(df_all) if not df_all.empty else 0
    _n_scores = len(scores) if not scores.empty else 0
    _n_find   = len(findings_df) if not findings_df.empty else 0
    _n_exp    = len(exps_df) if not exps_df.empty else 0
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#484F58;"
        f"margin-bottom:12px;padding:6px 0;border-bottom:1px solid #21262D'>"
        f"biometrics {_n_bio}d &nbsp;·&nbsp; "
        f"scores {_n_scores}d &nbsp;·&nbsp; "
        f"findings {_n_find} &nbsp;·&nbsp; "
        f"experiments {_n_exp}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── TODAY AT A GLANCE ───────────────────────────────────
    def _recovery_score():
        pts = 50.0
        # HRV: last night vs personal baseline — strongest recovery signal, ±30 pts
        _h = _get_series("hrv_ms").sort_index()
        if len(_h) >= 14:
            last_h = float(_h.iloc[-1])
            base_h = float(_h.iloc[:-1].mean())  # exclude last night from baseline
            if base_h > 0:
                pts += max(-30, min(30, (last_h / base_h - 1) * 100))
        # RHR: last night vs baseline — lower than usual = good, ±20 pts
        _r = _get_series("rhr_bpm").sort_index()
        if len(_r) >= 14:
            last_r = float(_r.iloc[-1])
            base_r = float(_r.iloc[:-1].mean())
            if base_r > 0:
                pts -= max(-20, min(20, (last_r / base_r - 1) * 100))
        # Sleep efficiency: absolute pp diff from baseline, ±15 pts
        _e = _get_series("sleep_efficiency_pct").sort_index()
        if len(_e) >= 14:
            last_e = float(_e.iloc[-1])
            base_e = float(_e.iloc[:-1].mean())
            pts += max(-15, min(15, (last_e - base_e) * 1.5))
        return max(0, min(100, round(pts)))

    def _streak(col, threshold, direction="above"):
        s = _get_series(col)
        if len(s) < 1: return 0
        count = 0
        for v in reversed(s.sort_index().values):
            if pd.isna(v): break
            meets = (v >= threshold) if direction == "above" else (v <= threshold)
            if meets: count += 1
            else: break
        return count

    _rec        = _recovery_score()
    _rec_color  = "#10B981" if _rec >= 70 else "#F59E0B" if _rec >= 45 else "#EF4444"
    _rec_label  = "Good" if _rec >= 70 else "Fair" if _rec >= 45 else "Low"
    _stk_steps  = _streak("steps", 8000)
    _stk_sleep  = _streak("sleep_efficiency_pct", 80)

    _hrv_s   = _get_series("hrv_ms")
    _rhr_s   = _get_series("rhr_bpm")
    _eff_s   = _get_series("sleep_efficiency_pct")
    _dur_s   = _get_series("sleep_duration_min")
    _hrv_now = float(_hrv_s.iloc[-1]) if len(_hrv_s) else None
    _rhr_now = float(_rhr_s.iloc[-1]) if len(_rhr_s) else None
    _eff_now = float(_eff_s.iloc[-1]) if len(_eff_s) else None
    _dur_now = float(_dur_s.iloc[-1]) / 60 if len(_dur_s) else None
    _hrv_base = float(_hrv_s.mean()) if len(_hrv_s) >= 14 else None

    _ins_parts = []
    if _hrv_now and _hrv_base:
        _hp = (_hrv_now / _hrv_base - 1) * 100
        if abs(_hp) >= 5:
            _ins_parts.append(f"HRV {abs(_hp):.0f}% {'above' if _hp > 0 else 'below'} baseline")
    if _stk_steps >= 3:
        _ins_parts.append(f"{_stk_steps}-day steps streak")
    if _stk_sleep >= 3:
        _ins_parts.append(f"{_stk_sleep}-day sleep streak")
    _insight = " · ".join(_ins_parts) if _ins_parts else "Keep logging daily to build your baseline."

    _tc1, _tc2, _tc3, _tc4 = st.columns([1, 1.2, 1.2, 3])

    _tc1.markdown(
        f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;"
        f"padding:12px 10px;text-align:center'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#484F58;margin-bottom:4px'>Recovery</div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:34px;font-weight:300;"
        f"color:{_rec_color};line-height:1.1'>{_rec}</div>"
        f"<div style='font-family:Inter,sans-serif;font-size:10px;color:{_rec_color}'>{_rec_label}</div>"
        f"</div>", unsafe_allow_html=True)

    _stk_sc = "#2DD4BF" if _stk_steps >= 3 else "#F59E0B" if _stk_steps >= 1 else "#484F58"
    _stk_ec = "#2DD4BF" if _stk_sleep >= 3 else "#F59E0B" if _stk_sleep >= 1 else "#484F58"
    _tc2.markdown(
        f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;padding:12px 10px'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#484F58;margin-bottom:6px'>Streaks</div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#8B949E'>"
        f"Steps ≥8k &nbsp;<span style='color:{_stk_sc};font-size:14px'>{_stk_steps}d</span></div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#8B949E;margin-top:4px'>"
        f"Sleep ≥80% &nbsp;<span style='color:{_stk_ec};font-size:14px'>{_stk_sleep}d</span></div>"
        f"</div>", unsafe_allow_html=True)

    _tc3.markdown(
        f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;padding:12px 10px'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#484F58;margin-bottom:6px'>Last Night</div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8B949E'>"
        f"Sleep &nbsp;<span style='color:#E6EDF3'>{f'{_dur_now:.1f}h' if _dur_now else '—'}</span>"
        f"&nbsp; Eff &nbsp;<span style='color:#E6EDF3'>{f'{_eff_now:.0f}%' if _eff_now else '—'}</span></div>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#8B949E;margin-top:4px'>"
        f"HRV &nbsp;<span style='color:#2DD4BF'>{f'{_hrv_now:.0f}ms' if _hrv_now else '—'}</span>"
        f"&nbsp; RHR &nbsp;<span style='color:#EF4444'>{f'{_rhr_now:.0f}bpm' if _rhr_now else '—'}</span></div>"
        f"</div>", unsafe_allow_html=True)

    _tc4.markdown(
        f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;"
        f"padding:12px 14px'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#484F58;margin-bottom:6px'>Summary</div>"
        f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#8B949E;line-height:1.6'>"
        f"{_insight}</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── KPI STRIP ───────────────────────────────────────────
    _hrv_d, _hrv_dc   = _delta("hrv_ms")
    _rhr_d, _rhr_dc_r = _delta("rhr_bpm")
    _rhr_dc           = "#EF4444" if (_rhr_d or 0) > 0 else "#10B981" if (_rhr_d or 0) < 0 else "#484F58"
    _stps_d, _stps_dc = _delta("steps")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _ks = st.columns(6)
    for _kc, _html in zip(_ks, [
        _kpi("HRV RMSSD",  _latest("hrv_ms"),         "ms",  _hrv_d, _hrv_dc,  bg=_metric_bg("hrv_ms", _latest("hrv_ms"))),
        _kpi("Resting HR", _latest("rhr_bpm"),         "bpm", _rhr_d, _rhr_dc,  bg=_metric_bg("rhr_bpm", _latest("rhr_bpm"))),
        _kpi("SpO₂",       _latest("spo2_avg_pct"),    "%",   bg=_metric_bg("spo2_avg_pct", _latest("spo2_avg_pct"))),
        _kpi("Steps",      _latest("steps"),           "",    _stps_d, _stps_dc, bg=_metric_bg("steps", _latest("steps"))),
        _kpi("Active Min", _latest("active_zone_min"), "min", bg=_metric_bg("active_zone_min", _latest("active_zone_min"))),
        _kpi("Calories",    _latest("calories_burned"),  "kcal"),
    ]):
        _kc.markdown(_html, unsafe_allow_html=True)

    # ── 7-DAY ROLLING vs BASELINE ─────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    _rm_cfg = [
        ("hrv_ms",               "HRV",        "ms",   False),
        ("rhr_bpm",              "RHR",         "bpm",  True),
        ("spo2_avg_pct",         "SpO₂",        "%",    False),
        ("steps",                "Steps",       "",     False),
        ("sleep_efficiency_pct", "Sleep Eff",   "%",    False),
        ("calories_burned",      "Calories",    "kcal", False),
    ]
    _rm_cols = st.columns(len(_rm_cfg))
    for _rmc, (_rmk, _rml, _rmu, _rminv) in zip(_rm_cols, _rm_cfg):
        _rms = _get_series(_rmk).sort_index()
        if len(_rms) >= 14:
            _r7  = float(_rms.iloc[-7:].mean())
            _r90 = float(_rms.mean())
            _rpct = (_r7 / _r90 - 1) * 100 if _r90 != 0 else 0
            if _rminv: _rpct = -_rpct
            _rcc = "#10B981" if _rpct >= 3 else "#EF4444" if _rpct <= -3 else "#484F58"
            _rarr = "↑" if _rpct >= 3 else "↓" if _rpct <= -3 else "→"
            _rmc.markdown(
                f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;"
                f"padding:5px 8px;text-align:center'>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
                f"letter-spacing:.1em;text-transform:uppercase;color:#484F58'>{_rml}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:15px;font-weight:300;"
                f"color:{_rcc}'>{_rarr} {abs(_rpct):.1f}%</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;color:#484F58'>"
                f"7d avg {_r7:.1f}{_rmu}</div>"
                f"</div>", unsafe_allow_html=True)
        else:
            _rmc.markdown(
                f"<div style='background:#161B22;border:1px solid #21262D;border-radius:4px;"
                f"padding:5px 8px;text-align:center'>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;font-weight:600;"
                f"letter-spacing:.1em;text-transform:uppercase;color:#484F58'>{_rml}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#484F58'>—</div>"
                f"</div>", unsafe_allow_html=True)

    # ── VITAL SIGNS GAUGES ──────────────────────────────────
    _section("Vital Signs — Yesterday")
    _gv1, _gv2, _gv3, _gv4 = st.columns(4)

    def _gauge_fig(label, value, min_v, max_v, unit, color_steps):
        _gval = value if value is not None else min_v
        _gc = "#484F58"
        for _thr, _c in color_steps:
            if _gval >= _thr:
                _gc = _c
        _fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_gval,
            number=dict(
                suffix=f" {unit}" if unit else "",
                font=dict(family="IBM Plex Mono, monospace", size=20, color="#E6EDF3"),
            ),
            title=dict(text=label,
                       font=dict(family="Inter, sans-serif", size=10, color="#484F58")),
            gauge=dict(
                axis=dict(range=[min_v, max_v], tickwidth=1, tickcolor="#30363D",
                          tickfont=dict(size=8, family="IBM Plex Mono, monospace",
                                        color="#484F58"),
                          nticks=5),
                bar=dict(color=_gc, thickness=0.65),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=1, bordercolor="#21262D",
                steps=[dict(range=[min_v, max_v], color="rgba(255,255,255,0.03)")],
                threshold=dict(line=dict(color="#E6EDF3", width=2),
                               thickness=0.75, value=_gval),
            ),
        ))
        _cll = {k: v for k, v in _CL.items() if k != "margin"}
        _fig_g.update_layout(**_cll)
        _fig_g.update_layout(height=170, margin=dict(l=12, r=12, t=30, b=8))
        return _fig_g

    for _gcol, _glbl, _gcn, _gmn, _gmx, _gu, _gcs in [
        (_gv1, "Resting HR",    "rhr_bpm",      40,  100,   "bpm",
         [(40, "#4A90D9"), (60, "#10B981"), (70, "#F59E0B"), (80, "#EF4444")]),
        (_gv2, "SpO₂",          "spo2_avg_pct", 88,  100,   "%",
         [(88, "#EF4444"), (90, "#F59E0B"), (95, "#10B981"), (98, "#2DD4BF")]),
        (_gv3, "HRV RMSSD",    "hrv_ms",         0,  120,   "ms",
         [(0,  "#EF4444"), (20, "#F59E0B"), (40, "#10B981"), (60, "#2DD4BF")]),
        (_gv4, "Today's Steps", "steps",          0, 15000,  "",
         [(0,  "#EF4444"), (5000, "#F59E0B"), (10000, "#10B981"), (13000, "#2DD4BF")]),
    ]:
        with _gcol:
            _gv = _latest(_gcn)
            if _gv is not None:
                st.plotly_chart(_gauge_fig(_glbl, _gv, _gmn, _gmx, _gu, _gcs),
                                width="stretch", config=_CFG)
            else:
                st.markdown(f"<div class='empty-panel'>{_glbl}<br>No data</div>",
                            unsafe_allow_html=True)

    # ── SLEEP ARCHITECTURE ──────────────────────────────────
    _slsl1, _slsl2 = st.columns([6, 2])
    _slsl1.markdown("<div class='dash-section'>Sleep Architecture</div>", unsafe_allow_html=True)
    _sleep_days = _slsl2.select_slider(
        "Sleep window", [7, 30, 90],
        value=st.session_state.get("sleep_days", 30),
        key="sleep_days", format_func=lambda d: f"{d}d",
    )
    _df_w = (df_all[df_all.index >= df_all.index.max() - pd.Timedelta(days=_sleep_days - 1)]
             if not df_all.empty else pd.DataFrame())
    _sa1, _sa2 = st.columns([3, 1])

    _stage_cfg = [
        ("deep_sleep_min",  "Deep (N3)", "#2DD4BF"),
        ("rem_sleep_min",   "REM",       "#8B5CF6"),
        ("light_sleep_min", "Light",     "#4A90D9"),
        ("awake_min",       "Awake",     "#3D4451"),
    ]
    _scols = [c for c, _, _ in _stage_cfg]

    with _sa1:
        if not _df_w.empty and any(c in _df_w.columns for c in _scols):
            _sd = _df_w[[c for c in _scols if c in _df_w.columns]].dropna(how="all")
            if not _sd.empty:
                _chart_label("Sleep Stage Breakdown (min)", n=len(_sd))
                _f = go.Figure()
                for _c, _n, _clr in _stage_cfg:
                    if _c in _sd.columns:
                        _f.add_trace(go.Bar(
                            x=_sd.index, y=_sd[_c], name=_n, marker_color=_clr,
                            hovertemplate=f"{_n}: %{{y:.0f}} min<extra></extra>",
                        ))
                _f.update_layout(**_CL, barmode="stack", height=230, bargap=0.15,
                                 legend=dict(orientation="h", y=1.2, x=0,
                                             font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(_f, width="stretch", config=_CFG)

    with _sa2:
        if not _df_w.empty and any(c in _df_w.columns for c in _scols):
            _avgs = _df_w[[c for c in _scols if c in _df_w.columns]].mean()
            if _avgs.sum() > 0:
                _chart_label(f"{_sleep_days}d Avg", n=len(_df_w.dropna(how="all")))
                _f = go.Figure(go.Pie(
                    labels=["Deep", "REM", "Light", "Awake"],
                    values=[_avgs.get(c, 0) for c in _scols],
                    marker_colors=["#2DD4BF", "#8B5CF6", "#4A90D9", "#3D4451"],
                    hole=0.65, textinfo="percent",
                    textfont=dict(size=9, family="IBM Plex Mono"),
                    hovertemplate="%{label}: %{value:.0f} min<extra></extra>",
                ))
                _f.add_annotation(
                    text=f"{_avgs.sum()/60:.1f}h", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=15, family="IBM Plex Mono", color="#E6EDF3"),
                )
                _f.update_layout(**_CL, height=230, showlegend=False)
                st.plotly_chart(_f, width="stretch", config=_CFG)

    _se1, _se2 = st.columns(2)
    with _se1:
        _s = _windowed("sleep_efficiency_pct", _sleep_days)
        if len(_s) >= 3:
            _chart_label("Sleep Efficiency (%)", _s)
            st.plotly_chart(_trend(_s, "#4A90D9", "rgba(74,144,217,0.12)",
                                   height=180, ref=85, rlabel="85%"),
                            width="stretch", config=_CFG)
    with _se2:
        _raw = _windowed("sleep_duration_min", _sleep_days)
        if len(_raw) >= 3:
            _sh = _raw / 60
            _chart_label("Sleep Duration (h)", _sh)
            _f = go.Figure()
            _f.add_trace(go.Bar(x=_sh.index, y=_sh.values, marker_color="#8B5CF6",
                                opacity=0.5, showlegend=False,
                                hovertemplate="%{x|%b %-d}: %{y:.1f}h<extra></extra>"))
            _f.add_trace(go.Scatter(x=_sh.index,
                                    y=_sh.rolling(7, min_periods=3).mean().values,
                                    mode="lines", line=dict(color="#8B5CF6", width=2),
                                    showlegend=False))
            _f.add_hline(y=8, line_dash="dot", line_color="#30363D", line_width=1,
                         annotation_text="8h", annotation_font_color="#484F58",
                         annotation_font_size=9)
            _f.update_layout(**_CL, height=180, bargap=0.2)
            st.plotly_chart(_f, width="stretch", config=_CFG)

    # Sleep stage composition — normalised % per night
    _scomp_cols = [c for c in _scols if not _df_w.empty and c in _df_w.columns]
    if len(_scomp_cols) >= 2:
        _scd2 = _df_w[_scomp_cols].dropna(how="all")
        if len(_scd2) >= 5:
            _sct2 = _scd2.sum(axis=1).replace(0, np.nan)
            _scp2 = _scd2.div(_sct2, axis=0).fillna(0) * 100
            _chart_label("Sleep Stage Composition (% of night)", n=len(_scd2))
            _scf2 = go.Figure()
            for _c, _n, _clr in _stage_cfg:
                if _c in _scp2.columns:
                    _scf2.add_trace(go.Bar(
                        x=_scp2.index, y=_scp2[_c], name=_n, marker_color=_clr,
                        hovertemplate=f"{_n}: %{{y:.1f}}%<extra></extra>",
                    ))
            _scf2.update_layout(**_CL, barmode="stack", height=160, bargap=0.1,
                                legend=dict(orientation="h", y=1.25, x=0,
                                            font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
            _scf2.update_layout(yaxis=dict(**_CL["yaxis"], range=[0, 100]))
            st.plotly_chart(_scf2, width="stretch", config=_CFG)

    # ── CARDIOVASCULAR ──────────────────────────────────────
    _cv_days_opts = [7, 14, 30, 60, 90]
    _cvsl1, _cvsl2 = st.columns([6, 2])
    _cvsl1.markdown("<div class='dash-section'>Cardiovascular</div>", unsafe_allow_html=True)
    _cv_days = _cvsl2.select_slider(
        "Cardiovascular window", _cv_days_opts,
        value=st.session_state.get("cv_range", 90),
        key="cv_range", format_func=lambda d: f"{d}d",
        label_visibility="collapsed",
    )

    def _cv_series(col):
        s = _get_series(col).sort_index()
        if s.empty: return s
        return s[s.index >= s.index.max() - pd.Timedelta(days=_cv_days - 1)]

    _cv1, _cv2 = st.columns(2)
    _cv3, _cv4 = st.columns(2)
    for _col, _m, _lbl, _clr, _fill, _ref, _rl in [
        (_cv1, "hrv_ms",           "HRV RMSSD (ms)",           "#2DD4BF", "rgba(45,212,191,0.12)",  None, ""),
        (_cv2, "rhr_bpm",          "Resting Heart Rate (bpm)", "#EF4444", "rgba(239,68,68,0.12)",   None, ""),
        (_cv3, "spo2_avg_pct",     "SpO₂ Average (%)",         "#7EC8A4", "rgba(126,200,164,0.12)", 95,   "95%"),
        (_cv4, "respiratory_rate", "Respiratory Rate (br/min)","#F59E0B", "rgba(245,158,11,0.12)",  None, ""),
    ]:
        with _col:
            _s = _cv_series(_m)
            if len(_s) >= 3:
                _chart_label(_lbl, _s)
                st.plotly_chart(_trend(_s, _clr, _fill, height=220, ref=_ref, rlabel=_rl),
                                width="stretch", config=_CFG)
            else:
                st.markdown(f"<div class='empty-panel'>{_lbl}<br>No data yet.</div>",
                            unsafe_allow_html=True)

    # Extended cardiovascular row: RHR distribution, VO₂ Max, Sedentary time
    _cv5, _cv6, _cv7 = st.columns(3)

    with _cv5:
        _rhr_h = _cv_series("rhr_bpm")
        if len(_rhr_h) >= 5:
            _chart_label(f"RHR Distribution ({len(_rhr_h)} days)")
            _rhrf = go.Figure(go.Histogram(
                x=_rhr_h.values,
                xbins=dict(start=float(_rhr_h.min()) - 0.5,
                           end=float(_rhr_h.max()) + 0.5, size=2),
                marker_color="#EF4444", opacity=0.65,
                hovertemplate="%{x} bpm: %{y} days<extra></extra>",
            ))
            _rhrf.update_layout(**_CL, height=200, bargap=0.06)
            st.plotly_chart(_rhrf, width="stretch", config=_CFG)
        else:
            st.markdown("<div class='empty-panel'>RHR Distribution<br>Need ≥5 days</div>",
                        unsafe_allow_html=True)

    with _cv6:
        _vos = _cv_series("vo2_max")
        if len(_vos) >= 3:
            _chart_label("VO₂ Max (mL/kg/min)", _vos)
            st.plotly_chart(_trend(_vos, "#2DD4BF", "rgba(45,212,191,0.12)", height=220),
                            width="stretch", config=_CFG)
        else:
            st.markdown("<div class='empty-panel'>VO₂ Max<br>No data yet.</div>",
                        unsafe_allow_html=True)

    with _cv7:
        _seds = _cv_series("sedentary_min")
        if len(_seds) >= 3:
            _sedh = _seds / 60
            _chart_label("Sedentary Time (h)", _sedh)
            st.plotly_chart(_trend(_sedh, "#F59E0B", "rgba(245,158,11,0.12)",
                                   height=220, ref=8, rlabel="8h"),
                            width="stretch", config=_CFG)
        else:
            st.markdown("<div class='empty-panel'>Sedentary Time<br>No data yet.</div>",
                        unsafe_allow_html=True)

    # ── ACTIVITY ────────────────────────────────────────────
    _acsl1, _acsl2 = st.columns([6, 2])
    _acsl1.markdown("<div class='dash-section'>Activity</div>", unsafe_allow_html=True)
    _act_days = _acsl2.select_slider(
        "Activity window", [7, 14, 30, 60, 90],
        value=st.session_state.get("act_days", 30),
        key="act_days", format_func=lambda d: f"{d}d",
    )
    _act_df = (df_all[df_all.index >= df_all.index.max() - pd.Timedelta(days=_act_days - 1)]
               if not df_all.empty else pd.DataFrame())

    _act1, _act2 = st.columns([2, 1])

    with _act1:
        _s = _act_df["steps"].dropna() if not _act_df.empty and "steps" in _act_df.columns else pd.Series(dtype=float)
        if len(_s) >= 3:
            _chart_label("Daily Steps", _s)
            _f = go.Figure()
            _f.add_trace(go.Bar(x=_s.index, y=_s.values, marker_color="#4A90D9",
                                opacity=0.5, showlegend=False,
                                hovertemplate="%{x|%b %-d}: %{y:,.0f}<extra></extra>"))
            _f.add_trace(go.Scatter(x=_s.index,
                                    y=_s.rolling(7, min_periods=3).mean().values,
                                    mode="lines", line=dict(color="#E6EDF3", width=1.5),
                                    showlegend=False))
            _f.add_hline(y=10000, line_dash="dot", line_color="#30363D", line_width=1,
                         annotation_text="10k", annotation_font_color="#484F58",
                         annotation_font_size=9)
            _f.update_layout(**_CL, height=200, bargap=0.2)
            st.plotly_chart(_f, width="stretch", config=_CFG)

    with _act2:
        _zcols = ["time_in_fat_burn_min", "time_in_cardio_min",
                  "time_in_peak_min", "lightly_active_min"]
        if not _act_df.empty and any(c in _act_df.columns for c in _zcols):
            _zavg = _act_df[[c for c in _zcols if c in _act_df.columns]].mean()
            if _zavg.sum() > 0:
                _chart_label("Avg Activity Zones", n=len(_act_df))
                _f = go.Figure(go.Pie(
                    labels=["Fat Burn", "Cardio", "Peak", "Light"],
                    values=[_zavg.get(c, 0) for c in _zcols],
                    marker_colors=["#F59E0B", "#EF4444", "#8B5CF6", "#4A90D9"],
                    hole=0.65, textinfo="percent",
                    textfont=dict(size=9, family="IBM Plex Mono"),
                    hovertemplate="%{label}: %{value:.0f} min/day avg<extra></extra>",
                ))
                _f.update_layout(**_CL, height=200, showlegend=False)
                st.plotly_chart(_f, width="stretch", config=_CFG)

    _act3, _act4 = st.columns(2)
    for _col, _m, _lbl, _clr, _fill in [
        (_act3, "calories_burned", "Calories Burned (kcal)", "#F59E0B", "rgba(245,158,11,0.08)"),
        (_act4, "distance_km",     "Distance (km)",          "#7EC8A4", "rgba(126,200,164,0.08)"),
    ]:
        with _col:
            _s = _act_df[_m].dropna() if not _act_df.empty and _m in _act_df.columns else pd.Series(dtype=float)
            if len(_s) >= 3:
                _chart_label(_lbl, _s)
                _f = go.Figure()
                _f.add_trace(go.Bar(x=_s.index, y=_s.values, marker_color=_clr,
                                    opacity=0.5, showlegend=False,
                                    hovertemplate=f"%{{x|%b %-d}}: %{{y:.1f}}<extra></extra>"))
                _f.add_trace(go.Scatter(x=_s.index,
                                        y=_s.rolling(7, min_periods=3).mean().values,
                                        mode="lines", line=dict(color=_clr, width=2),
                                        showlegend=False))
                _f.update_layout(**_CL, height=180, bargap=0.2)
                st.plotly_chart(_f, width="stretch", config=_CFG)

    # ── ACTIVITY BREAKDOWN ──────────────────────────────────
    with st.expander(f"Activity Breakdown — {_act_days}d", expanded=True):
        _ab1, _ab2 = st.columns(2)

        with _ab1:
            _aint_cfg = [
                ("lightly_active_min", "Light",    "#4A90D9"),
                ("fairly_active_min",  "Moderate", "#F59E0B"),
                ("very_active_min",    "Intense",  "#EF4444"),
            ]
            _aint_any = not _act_df.empty and any(c in _act_df.columns for c, _, _ in _aint_cfg)
            if _aint_any:
                _chart_label("Activity Intensity (min/day)", n=len(_act_df))
                _aif = go.Figure()
                for _c, _n, _clr in _aint_cfg:
                    if _c in _act_df.columns:
                        _aif.add_trace(go.Bar(
                            x=_act_df.index, y=_act_df[_c].fillna(0), name=_n,
                            marker_color=_clr,
                            hovertemplate=f"{_n}: %{{y:.0f}} min<extra></extra>",
                        ))
                _aif.update_layout(**_CL, barmode="stack", height=220, bargap=0.15,
                                   legend=dict(orientation="h", y=1.18, x=0,
                                               font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(_aif, width="stretch", config=_CFG)
            else:
                st.markdown("<div class='empty-panel'>Activity Intensity<br>No data yet.</div>",
                            unsafe_allow_html=True)

        with _ab2:
            _zone_cfg2 = [
                ("time_in_fat_burn_min", "Fat Burn", "#F59E0B"),
                ("time_in_cardio_min",   "Cardio",   "#EF4444"),
                ("time_in_peak_min",     "Peak",     "#8B5CF6"),
            ]
            _zone_any2 = not _act_df.empty and any(c in _act_df.columns for c, _, _ in _zone_cfg2)
            if _zone_any2:
                _chart_label("HR Zones (min/day)", n=len(_act_df))
                _zf = go.Figure()
                for _c, _n, _clr in _zone_cfg2:
                    if _c in _act_df.columns:
                        _zf.add_trace(go.Bar(
                            x=_act_df.index, y=_act_df[_c].fillna(0), name=_n,
                            marker_color=_clr,
                            hovertemplate=f"{_n}: %{{y:.0f}} min<extra></extra>",
                        ))
                _zf.update_layout(**_CL, barmode="stack", height=220, bargap=0.15,
                                  legend=dict(orientation="h", y=1.18, x=0,
                                              font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(_zf, width="stretch", config=_CFG)
            else:
                st.markdown("<div class='empty-panel'>HR Zones<br>No data yet.</div>",
                            unsafe_allow_html=True)

    # ── STEPS WEEKLY HEATMAP ─────────────────────────────────
    with st.expander("Steps — Weekly Pattern", expanded=True):
        _stps_s = _get_series("steps")
        if len(_stps_s) >= 14:
            _shm_df = _stps_s.reset_index()
            _shm_df.columns = ["date", "steps"]
            _shm_df["dow"]  = _shm_df["date"].dt.dayofweek
            _iso             = _shm_df["date"].dt.isocalendar()
            _shm_df["week"] = _iso.week.astype(int)
            _shm_df["year"] = _iso.year.astype(int)
            _shm_df["yw"]   = _shm_df.apply(
                lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1)
            _pvt = _shm_df.pivot_table(values="steps", index="dow",
                                        columns="yw", aggfunc="sum")
            _pvt = _pvt.reindex(range(7))
            _dlbl = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            _chart_label("Step count by day-of-week × calendar week (teal = more steps)", n=len(_stps_s))
            _hmf = go.Figure(go.Heatmap(
                z=_pvt.values, x=_pvt.columns.tolist(), y=_dlbl,
                colorscale=[[0, "#161B22"], [0.3, "#1A4A6E"],
                            [0.65, "#2DD4BF"], [1.0, "#A7F3D0"]],
                showscale=False,
                hovertemplate="%{x} %{y}: %{z:,.0f} steps<extra></extra>",
            ))
            _hmf.update_layout(**_CL, height=220)
            _hmf.update_layout(margin=dict(l=40, r=16, t=30, b=20))
            st.plotly_chart(_hmf, width="stretch", config=_CFG)
        else:
            st.markdown(
                "<div class='empty-panel'>Weekly step pattern — need ≥14 days of data.</div>",
                unsafe_allow_html=True)

    # ── PERSONAL RECORDS ────────────────────────────────────
    with st.expander("Personal Records", expanded=False):
        _df_all_time = get_data(0)
        _pr_rows = []
        for _prc, _prl, _prd, _pru in [
            ("hrv_ms",               "Best HRV",          "max", "ms"),
            ("rhr_bpm",              "Lowest RHR",        "min", "bpm"),
            ("steps",                "Most Steps",        "max", ""),
            ("sleep_efficiency_pct", "Best Sleep Eff",    "max", "%"),
            ("vo2_max",              "Best VO₂ Max",      "max", "mL/kg/min"),
            ("sleep_duration_min",   "Longest Sleep",     "max", "h"),
        ]:
            _prs = pd.Series(dtype=float)
            if not _df_all_time.empty and _prc in _df_all_time.columns:
                _prs = _df_all_time[_prc].dropna()
            elif not _sc_idx.empty and _prc in _sc_idx.columns:
                _prs = _sc_idx[_prc].dropna()
            if len(_prs) >= 1:
                _prv = float(_prs.max() if _prd == "max" else _prs.min())
                _prdt = _prs.idxmax() if _prd == "max" else _prs.idxmin()
                _prcur = float(_prs.iloc[-1])
                _prpct = _prcur / _prv * 100 if _prv != 0 else 0
                _is_dur = _prc == "sleep_duration_min"
                _prv_d  = f"{_prv/60:.1f}" if _is_dur else f"{_prv:.1f}"
                _prc_d  = f"{_prcur/60:.1f}" if _is_dur else f"{_prcur:.1f}"
                _pru2   = "h" if _is_dur else _pru
                _pr_rows.append({
                    "Metric":    _prl,
                    "Best":      f"{_prv_d} {_pru2}".strip(),
                    "Date":      pd.Timestamp(_prdt).strftime("%b %-d, %Y"),
                    "Current":   f"{_prc_d} {_pru2}".strip(),
                    "% of Best": f"{_prpct:.0f}%",
                })
        if _pr_rows:
            st.dataframe(pd.DataFrame(_pr_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='empty-panel'>No records yet — keep logging.</div>",
                        unsafe_allow_html=True)

    # ── INTELLIGENCE ────────────────────────────────────────
    _section("Intelligence")
    _int1, _int2 = st.columns([3, 2])

    with _int1:
        st.markdown("<div class='dash-chart-label'>Top Correlations</div>",
                    unsafe_allow_html=True)
        if not findings_df.empty:
            for _, _r in findings_df.sort_values("r_squared", ascending=False).head(6).iterrows():
                _a  = analysis.COL_LABELS.get(_r["variable_a"], _r["variable_a"])
                _b  = analysis.COL_LABELS.get(_r["variable_b"], _r["variable_b"]) if _r["variable_b"] else "—"
                _r2 = float(_r["r_squared"])
                _co = float(_r["coefficient"])
                _lg = int(_r["lag_days"])
                _r2c = "#2DD4BF" if _r2 >= 0.5 else "#F59E0B" if _r2 >= 0.3 else "#484F58"
                _dc  = "#10B981" if _co > 0 else "#EF4444"
                st.markdown(
                    f"<div class='finding-row'>"
                    f"<div style='font-family:Inter;font-size:12px;color:#E6EDF3'>{_a}"
                    f"<span style='color:#484F58'> → </span>{_b}"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    f"color:#484F58;margin-left:8px'>lag +{_lg}d</span></div>"
                    f"<div style='display:flex;align-items:center;gap:8px;flex-shrink:0'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    f"color:{_dc}'>{'↑' if _co > 0 else '↓'}</span>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:14px;"
                    f"font-weight:600;color:{_r2c}'>R²{_r2:.2f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='empty-panel'>Patterns computed weekly —"
                        " check back after Sunday's analysis run.</div>",
                        unsafe_allow_html=True)

    with _int2:
        st.markdown("<div class='dash-chart-label'>Active Experiments</div>",
                    unsafe_allow_html=True)
        _ae = exps_df[~exps_df["is_complete"]] if not exps_df.empty else pd.DataFrame()
        if not _ae.empty:
            for _, _exp in _ae.head(4).iterrows():
                _es  = pd.Timestamp(_exp["start_date"])
                _ee  = pd.Timestamp(_exp["end_date"])
                _el  = max(0, (min(pd.Timestamp.today().normalize(), _ee) - _es).days)
                _dur = int(_exp["duration_days"])
                _pct = min(_el / _dur, 1.0) if _dur > 0 else 0
                _bar = int(_pct * 18)
                _a   = analysis.COL_LABELS.get(_exp["variable_a"], _exp["variable_a"])
                _b   = analysis.COL_LABELS.get(_exp["variable_b"], _exp["variable_b"])
                st.markdown(
                    f"<div class='exp-card'>"
                    f"<div style='font-family:Inter;font-size:12px;color:#E6EDF3;"
                    f"margin-bottom:3px'>{_exp['name']}</div>"
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    f"color:#484F58;margin-bottom:6px'>{_a} → {_b}</div>"
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                    f"color:#2DD4BF'>{'█' * _bar}{'░' * (18 - _bar)}</div>"
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    f"color:#484F58;margin-top:3px'>Day {_el} of {_dur}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='empty-panel'>No active experiments.</div>",
                        unsafe_allow_html=True)

    st.stop()  # end Dashboard

# ─────────────────────────────────────────────────────────────
# INSIGHTS PAGE
# ─────────────────────────────────────────────────────────────

if page == "Insights":
    st.title("Insights")

    scores = get_daily_scores()
    recs   = get_score_recommendations()

    MIN_SCORE_DAYS = 7

    def score_color(s):
        if s >= 70: return GREEN
        if s >= 45: return ORANGE
        return RED

    def score_delta_str(series: pd.Series):
        """Yesterday vs day before — direction arrow and delta."""
        valid = series.dropna()
        if len(valid) < 2:
            return "→", GRAY
        delta = float(valid.iloc[0]) - float(valid.iloc[1])
        if delta > 2:
            return f"↑ {delta:.0f}", GREEN
        if delta < -2:
            return f"↓ {abs(delta):.0f}", RED
        return "→ stable", GRAY

    # ══════════════════════════════════════════════════════════
    # SCORES — two columns
    # ══════════════════════════════════════════════════════════
    col_sleep, col_heart = st.columns(2)

    for col, score_col, comp_cols, comp_labels, title, icon in [
        (
            col_sleep, "sleep_score",
            ["duration_score", "deep_score", "rem_score", "efficiency_score"],
            ["Duration", "Deep %", "REM %", "Efficiency"],
            "Sleep Score", "🌙",
        ),
        (
            col_heart, "heart_score",
            ["hrv_score", "rhr_score", "spo2_score"],
            ["HRV", "Resting HR", "SpO₂"],
            "Heart Score", "❤️",
        ),
    ]:
        with col:
            st.markdown(f"#### {icon} {title}")

            if scores.empty or scores[score_col].dropna().empty:
                st.caption(
                    f"Score appears after {MIN_SCORE_DAYS} days of biometric data. "
                    "Sync your Fitbit daily to build your baseline."
                )
            else:
                latest_score = float(scores[score_col].dropna().iloc[0])
                arrow, acolor = score_delta_str(scores[score_col])
                s_color = score_color(latest_score)

                st.markdown(
                    f"<div style='font-size:3.2em;font-weight:700;color:{s_color};line-height:1'>"
                    f"{latest_score:.0f}"
                    f"<span style='font-size:0.4em;color:{GRAY}'>/100</span></div>"
                    f"<div style='color:{acolor};font-size:1em;margin-bottom:0.5em'>{arrow}</div>",
                    unsafe_allow_html=True,
                )

                # Component bar chart
                comp_data = scores[comp_cols].dropna(how="all").iloc[0]
                valid_comps = [(lbl, float(comp_data[col]))
                               for lbl, col in zip(comp_labels, comp_cols)
                               if pd.notna(comp_data[col])]

                if valid_comps:
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in valid_comps],
                        y=[l for l, _ in valid_comps],
                        orientation="h",
                        marker_color=[score_color(v) for _, v in valid_comps],
                        text=[f"{v:.0f}" for _, v in valid_comps],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=40 * len(valid_comps) + 40,
                        margin=dict(l=0, r=40, t=10, b=0),
                        xaxis=dict(range=[0, 100], showticklabels=False),
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, width="stretch")

                # 30-day trend
                trend = scores[["date", score_col]].dropna().sort_values("date")
                if len(trend) >= 3:
                    fig2 = go.Figure(go.Scatter(
                        x=trend["date"], y=trend[score_col],
                        mode="lines",
                        line=dict(color=s_color, width=2),
                        fill="tozeroy",
                        fillcolor=f"rgba({','.join(str(int(s_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
                        hovertemplate="%{x|%b %-d}: %{y:.0f}<extra></extra>",
                    ))
                    fig2.update_layout(
                        height=140,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=True, showgrid=False),
                        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#f0f0f0"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig2, width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # TOP FINDINGS
    # ══════════════════════════════════════════════════════════
    st.markdown("#### Top Findings")

    _findings = get_findings()
    _auto = _findings[~_findings["pinned"].astype(bool)] if not _findings.empty else pd.DataFrame()

    if _auto.empty:
        st.caption("No findings yet — the weekly job runs every Sunday and will surface your strongest biometric patterns here.")
    else:
        _auto = _auto.sort_values("r_squared", ascending=False).head(5)
        for _, row in _auto.iterrows():
            a_lbl = analysis.COL_LABELS.get(row["variable_a"], row["variable_a"])
            b_lbl = analysis.COL_LABELS.get(row["variable_b"], row["variable_b"])
            r2    = float(row["r_squared"])
            coef  = float(row["coefficient"])
            lag   = int(row["lag_days"])
            n     = int(row["sample_size"])
            direction = "↑" if coef > 0 else "↓"
            lag_str   = f" · {lag}d lag" if lag > 0 else ""
            r2_color  = GREEN if r2 >= 0.15 else ORANGE if r2 >= 0.07 else GRAY

            with st.container(border=True):
                c1, c2 = st.columns([7, 2])
                c1.markdown(f"**{a_lbl}** → **{b_lbl}**{lag_str}")
                c1.caption(f"{direction} {abs(coef):.3f} coefficient · {n} days of data")
                c2.markdown(
                    f"<div style='text-align:right;padding-top:0.3em'>"
                    f"<div style='font-size:0.75rem;color:{GRAY}'>R²</div>"
                    f"<div style='font-size:1.6em;font-weight:700;color:{r2_color}'>{r2:.3f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════
    st.markdown("#### Activity Recommendations")

    if recs.empty:
        if scores.empty or len(scores) < MIN_SCORE_DAYS:
            st.caption(
                f"Recommendations appear after {MIN_SCORE_DAYS} days of data. "
                "Keep syncing daily — the more data, the more specific the insights."
            )
        else:
            st.caption(
                "No strong activity patterns found yet. "
                "Recommendations strengthen as your data grows."
            )
    else:
        n_days_data = int(recs["sample_size"].max()) if not recs.empty else 0
        st.caption(f"Based on {n_days_data} days of your data. Updated daily.")

        for _, rec in recs.head(6).iterrows():
            target    = str(rec["target_score"])
            delta     = float(rec["score_delta"])
            in_range  = float(rec["avg_score_in_range"])
            out_range = float(rec["avg_score_outside"])
            icon      = "🌙" if target == "sleep" else "❤️"
            tag_color = BLUE if target == "sleep" else RED

            with st.container(border=True):
                rc1, rc2 = st.columns([7, 3])
                with rc1:
                    st.markdown(
                        f"<span style='font-size:0.7rem;font-weight:600;color:{tag_color};"
                        f"text-transform:uppercase;letter-spacing:0.05em'>"
                        f"{icon} {target.capitalize()} Score</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**{rec['activity_label'].capitalize()}:**  "
                        f"{rec['optimal_min_fmt']} – {rec['optimal_max_fmt']}"
                    )
                    st.caption(rec["recommendation_text"])
                with rc2:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:0.4em'>"
                        f"<div style='font-size:0.75rem;color:{GRAY}'>Score impact</div>"
                        f"<div style='font-size:1.8em;font-weight:700;color:{GREEN}'>+{delta:.0f}</div>"
                        f"<div style='font-size:0.7rem;color:{GRAY}'>{out_range:.0f} → {in_range:.0f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.stop()  # end Insights

# ─────────────────────────────────────────────────────────────
# EXPERIMENTS PAGE
# ─────────────────────────────────────────────────────────────

if page == "Experiments":
    df = get_data(0)

    # ── Experiment detail view ───────────────────────────────
    if st.session_state.exp_detail_id:
        exp_id   = st.session_state.exp_detail_id
        exps     = get_experiments()
        exp_row  = exps[exps["id"] == exp_id].iloc[0]

        if st.button("← Back"):
            st.session_state.exp_detail_id = None
            st.rerun()

        var_a = exp_row["variable_a"]
        var_b = exp_row["variable_b"]
        a_lbl = analysis.COL_LABELS.get(var_a, var_a)
        b_lbl = analysis.COL_LABELS.get(var_b, var_b)

        start     = pd.Timestamp(exp_row["start_date"])
        end       = pd.Timestamp(exp_row["end_date"])
        today     = pd.Timestamp.today().normalize()
        elapsed   = max(0, (min(today, end) - start).days)
        duration  = int(exp_row["duration_days"])
        complete  = bool(exp_row["is_complete"])

        phase_lbl = (f"DAYS 1 TO {elapsed} OF {duration}"
                     if not complete else f"DAYS 1 TO {duration}")
        st.markdown(f"<span style='color:{ORANGE};font-size:0.85em;font-weight:600'>"
                    f"{'COMPLETE' if complete else 'ACTIVE'} — {phase_lbl}</span>",
                    unsafe_allow_html=True)
        st.title(exp_row["name"])
        st.caption(f"{a_lbl} → {b_lbl}"
                   + (f" · {int(exp_row['lag_days'])}-day lag" if exp_row["lag_days"] else ""))

        result = analysis.run_experiment_analysis(df, exp_row)

        if "error" in result:
            st.warning(result["error"])
            st.stop()

        # Stats pills
        r2   = result["r2"]
        coef = result["coefficient"]
        c1, c2, c3 = st.columns(3)
        c1.metric("R²",         f"{r2:.4f}")
        c2.metric("p-value",    f"{result['p_value']:.4f}")
        c3.metric("Coefficient", f"{coef:.4f}")
        st.markdown(f"**{result['label']}**")
        st.caption(f"Based on {result['n']} days of experiment data")

        # Chart toggle
        view = st.radio("", ["Experiment view", "Gradient view"],
                        horizontal=True, label_visibility="collapsed")

        all_p  = result["all_paired"]
        pre    = result["pre"]
        during = result["during"]
        dot_color = BRIGHT_GREEN if coef >= 0 else "#ff4444"

        fig = go.Figure()

        if view == "Experiment view":
            show_all = st.toggle("Show full history", value=True)
            if show_all and len(pre):
                fig.add_trace(go.Scatter(
                    x=pre[var_a].values, y=pre[var_b].values, mode="markers",
                    name="Before experiment",
                    text=pre.index.strftime("%Y-%m-%d"),
                    hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
                    marker=dict(color="rgba(150,150,150,0.45)", size=7),
                ))
            fig.add_trace(go.Scatter(
                x=during[var_a].values, y=during[var_b].values, mode="markers",
                name="During experiment",
                text=during.index.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
                marker=dict(color=dot_color, size=9, opacity=0.9),
            ))

        else:  # Gradient view — color by recency
            n = len(all_p)
            positions = np.linspace(0, 1, n) if n > 1 else [1.0]
            fig.add_trace(go.Scatter(
                x=all_p[var_a].values, y=all_p[var_b].values, mode="markers",
                name="All data",
                text=all_p.index.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
                marker=dict(
                    color=positions,
                    colorscale=[[0, GRAD_OLD], [0.5, GRAD_MID], [1.0, BRIGHT_GREEN]],
                    size=9, showscale=False,
                ),
            ))
            # Legend hint
            st.caption("● Oldest data    ● Mid-period    ● Most recent")

        # OLS line across full paired data
        x_min = float(all_p[var_a].min())
        x_max = float(all_p[var_a].max())
        x_rng = np.linspace(x_min, x_max, 100)
        fig.add_trace(go.Scatter(
            x=x_rng, y=result["full_slope"] * x_rng + result["full_intercept"],
            mode="lines",
            line=dict(color=ORANGE, dash="dash", width=2),
            showlegend=False,
        ))
        fig.update_layout(xaxis_title=a_lbl, yaxis_title=b_lbl,
                          height=460, margin=dict(t=20),
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, width="stretch")

        # Interpretation
        if complete:
            interp = exp_row.get("interpretation")
            if not interp:
                with st.spinner("Generating interpretation…"):
                    interp = analysis.generate_interpretation(
                        var_a, var_b, r2, result["p_value"], coef,
                        int(exp_row["lag_days"]), result["n"],
                        result["pre_avg_a"], result["pre_avg_b"],
                        result["during_avg_a"], result["during_avg_b"],
                    )
                    analysis.store_interpretation(exp_id, interp)
                    get_experiments.clear()
            st.info(f"**What this shows:** {interp}")

        st.stop()

    # ── Experiment list view ─────────────────────────────────
    st.title("Experiments")
    exps = get_experiments()
    active = exps[~exps["is_complete"]] if not exps.empty else pd.DataFrame()
    past   = exps[exps["is_complete"]]  if not exps.empty else pd.DataFrame()

    tab_new, tab_active, tab_past = st.tabs(
        [f"New", f"Active ({len(active)})", f"Past ({len(past)})"]
    )

    with tab_new:
        n_active = len(active)
        if n_active >= 3:
            st.warning("You have 3 active experiments — complete or delete one before adding another.")
        else:
            # Variable pickers outside the form (cascading selectboxes need reruns)
            ec1, ec2 = st.columns(2)
            with ec1:
                _ea_cat = st.selectbox("VARIABLE A — Input / Driver", A_CATS, key="exp_a_cat")
                _ea_sub = st.selectbox("", A_SUBS[_ea_cat], key=f"exp_a_sub_{_ea_cat}", label_visibility="collapsed")
                _ea_grp = f"{_ea_cat}  ·  {_ea_sub}"
                st.selectbox("", VAR_A_TREE[_ea_grp], key=f"exp_a_var_{_ea_grp}", label_visibility="collapsed")
            with ec2:
                _eb_cat = st.selectbox("VARIABLE B — Output / Target", B_CATS, key="exp_b_cat")
                _eb_sub = st.selectbox("", B_SUBS[_eb_cat], key=f"exp_b_sub_{_eb_cat}", label_visibility="collapsed")
                _eb_grp = f"{_eb_cat}  ·  {_eb_sub}"
                st.selectbox("", VAR_B_TREE[_eb_grp], key=f"exp_b_var_{_eb_grp}", label_visibility="collapsed")

            st.divider()

            with st.form("new_exp"):
                name = st.text_input("Hypothesis / name",
                                     placeholder="e.g. Reducing sodium impact on deep sleep")
                c3, c4, c5 = st.columns(3)
                lag    = c3.selectbox("Lag (days)", [0, 1, 2, 3])
                method = c4.radio("Method", ["pearson", "spearman"], horizontal=True)
                dur    = c5.number_input("Duration (days, min 14)", min_value=14,
                                         max_value=365, value=30, step=1)
                start  = st.date_input("Start date", min_value=pd.Timestamp.today().date())
                submit = st.form_submit_button("Create experiment", type="primary")
                if submit:
                    _fa_cat = st.session_state.get("exp_a_cat", A_CATS[0])
                    _fa_sub = st.session_state.get(f"exp_a_sub_{_fa_cat}", A_SUBS[_fa_cat][0])
                    _fa_grp = f"{_fa_cat}  ·  {_fa_sub}"
                    _var_a  = st.session_state.get(f"exp_a_var_{_fa_grp}", VAR_A_TREE[_fa_grp][0])
                    _fb_cat = st.session_state.get("exp_b_cat", B_CATS[0])
                    _fb_sub = st.session_state.get(f"exp_b_sub_{_fb_cat}", B_SUBS[_fb_cat][0])
                    _fb_grp = f"{_fb_cat}  ·  {_fb_sub}"
                    _var_b  = st.session_state.get(f"exp_b_var_{_fb_grp}", VAR_B_TREE[_fb_grp][0])
                    if not name:
                        st.error("Give the experiment a name.")
                    else:
                        analysis.create_experiment(
                            name=name, variable_a=_var_a, variable_b=_var_b,
                            lag_days=lag, method=method,
                            start_date=start, duration_days=int(dur),
                        )
                        get_experiments.clear()
                        st.success("Experiment created.")
                        st.rerun()

    def exp_card(row):
        a_lbl = analysis.COL_LABELS.get(row["variable_a"], row["variable_a"])
        b_lbl = analysis.COL_LABELS.get(row["variable_b"], row["variable_b"])
        start = pd.Timestamp(row["start_date"])
        end   = pd.Timestamp(row["end_date"])
        dur   = int(row["duration_days"])
        today = pd.Timestamp.today().normalize()
        elapsed = max(0, (min(today, end) - start).days)
        complete = bool(row["is_complete"])

        with st.container(border=True):
            h1, h2 = st.columns([7, 2])
            h1.markdown(f"**{row['name']}**  \n"
                        f"{a_lbl} → {b_lbl}"
                        + (f" · lag {int(row['lag_days'])}d" if row["lag_days"] else ""))
            status_color = GREEN if complete else ORANGE
            status_txt   = "Complete" if complete else f"Day {elapsed} of {dur}"
            h2.markdown(f"<div style='text-align:right;color:{status_color}'>"
                        f"{status_txt}</div>", unsafe_allow_html=True)
            bc1, bc2 = st.columns([3, 1])
            if bc1.button("View details", key=f"view_{row['id']}"):
                st.session_state.exp_detail_id = int(row["id"])
                st.rerun()
            if bc2.button("Delete", key=f"del_exp_{row['id']}"):
                analysis.delete_experiment(int(row["id"]))
                get_experiments.clear()
                st.rerun()

    with tab_active:
        if active.empty:
            st.caption("No active experiments.")
        else:
            for _, row in active.iterrows():
                exp_card(row)

    with tab_past:
        if past.empty:
            st.caption("No completed experiments yet.")
        else:
            for _, row in past.iterrows():
                exp_card(row)

    st.stop()  # end Experiments

# ─────────────────────────────────────────────────────────────
# EXPLORER PAGE
# ─────────────────────────────────────────────────────────────

if page == "Explorer":
    st.sidebar.divider()
    analysis_type = st.sidebar.selectbox("Analysis Type", ANALYSIS_TYPES)

    days_map   = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90, "All data": 0}
    days_label = st.sidebar.selectbox("Data Range", list(days_map.keys()), index=3)
    days       = days_map[days_label]
    df         = get_data(days)

    n_bio = df[analysis.BIOMETRIC_COLS].dropna(how="all").shape[0]
    date_range = (f"{df.index.min().date()} → {df.index.max().date()}" if len(df) else "—")
    st.sidebar.caption(f"**{n_bio}** biometric days  \n{date_range}")
    st.sidebar.divider()

    if   analysis_type == "Lagged Correlation": lag = st.sidebar.selectbox("Lag (days)", [0,1,2,3], index=1); corr_method = st.sidebar.radio("Method", ["Pearson","Spearman"], horizontal=True).lower()
    elif analysis_type == "Rolling Average":    window = st.sidebar.selectbox("Window (days)", [7,14]); corr_method = st.sidebar.radio("Method", ["Pearson","Spearman"], horizontal=True).lower()
    elif analysis_type == "Decomposition":      period = st.sidebar.selectbox("Period (days)", [7, 14, 30])
    elif analysis_type == "Anomaly Detection":  window = st.sidebar.slider("Baseline window", 14, 60, 30); threshold = st.sidebar.slider("Threshold (SD)", 1.0, 3.0, 1.5, 0.1)

    run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    st.title("Explorer")

    if analysis_type in SINGLE_VAR:
        _cl, _cr = st.columns(2)
        with _cl:
            var_a = _picker("sv_a", A_CATS, A_SUBS, VAR_A_TREE, "VARIABLE A — Input / Driver")
        with _cr:
            _picker("sv_b", B_CATS, B_SUBS, VAR_B_TREE, "Or pick an Output variable")
        var_b = predictors = outcome = outcome_label = None

    elif analysis_type in MULTI_PRED:
        _cl, _cr = st.columns([3, 2])
        with _cl:
            st.markdown("**VARIABLE A — Input / Driver**")
            _a_cat = st.selectbox("", A_CATS, key="ols_a_cat", label_visibility="collapsed")
            _a_sub = st.selectbox("", A_SUBS[_a_cat], key=f"ols_a_sub_{_a_cat}", label_visibility="collapsed")
            _ag = f"{_a_cat}  ·  {_a_sub}"
            predictors = st.multiselect("", VAR_A_TREE[_ag], default=VAR_A_TREE[_ag][:1],
                                        key=f"ols_preds_{_ag}", label_visibility="collapsed")
        with _cr:
            outcome = _picker("ols_b", B_CATS, B_SUBS, VAR_B_TREE, "VARIABLE B — Output / Target")
        outcome_label = col_label(outcome)
        var_a = var_b = None

    else:
        _cl, _cr = st.columns(2)
        with _cl:
            var_a = _picker("a", A_CATS, A_SUBS, VAR_A_TREE, "VARIABLE A — Input / Driver")
        with _cr:
            var_b = _picker("b", B_CATS, B_SUBS, VAR_B_TREE, "VARIABLE B — Output / Target")
        predictors = outcome = outcome_label = None

    st.divider()

    if "result" not in st.session_state:
        st.session_state.result = None; st.session_state.result_type = None; st.session_state.result_meta = {}

    if run_clicked:
        st.session_state.saved_view_id = None
        with st.spinner("Running…"):
            if   analysis_type == "Pearson Correlation":     res = analysis.pearson_correlation(df, var_a, var_b)
            elif analysis_type == "Spearman Correlation":    res = analysis.spearman_correlation(df, var_a, var_b)
            elif analysis_type == "Lagged Correlation":      res = analysis.lagged_correlation(df, var_a, var_b, lag, corr_method)
            elif analysis_type == "Rolling Average":         res = analysis.rolling_avg_correlation(df, var_a, var_b, window, corr_method)
            elif analysis_type == "30-Day Trend (OLS)":      res = analysis.ols_trend(df, var_a)
            elif analysis_type == "Multiple OLS Regression": res = analysis.multiple_ols(df, predictors, outcome)
            elif analysis_type == "Anomaly Detection":       res = analysis.anomaly_detection(df, var_a, window, threshold)
            elif analysis_type == "Forecast (7-Day)":        res = analysis.forecast(df, var_a)
            elif analysis_type == "Decomposition":           res = analysis.decompose(df, var_a, period)
            else: res = {"error": "Unknown analysis type"}
        st.session_state.result = res; st.session_state.result_type = analysis_type
        st.session_state.result_meta = {
            "var_a": var_a, "var_b": var_b,
            "var_a_label": col_label(var_a) if var_a else None,
            "var_b_label": col_label(var_b) if var_b else None,
            "analysis_type": analysis_type,
            "lag": locals().get("lag", 0), "window": locals().get("window"),
            "period": locals().get("period"), "threshold": locals().get("threshold"),
            "predictors": locals().get("predictors"), "outcome": locals().get("outcome"),
            "outcome_label": col_label(outcome) if locals().get("outcome") else None,
        }

    # ── Saved Analyses ───────────────────────────────────────────
    _saved = get_findings()
    _saved = _saved[_saved["pinned"].astype(bool)] if not _saved.empty else pd.DataFrame()

    if not _saved.empty:
        st.markdown("#### Saved Analyses")
        for _, srow in _saved.iterrows():
            sid = int(srow["id"])
            s_a = srow["variable_a"]; s_b = srow["variable_b"]
            s_r2 = float(srow["r_squared"]) if srow["r_squared"] is not None else 0
            s_atype = srow["analysis_type"]
            s_date  = pd.Timestamp(srow["calculated_at"]).strftime("%Y-%m-%d")
            s_name  = f"{s_a} → {s_b}" if s_b else s_a

            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([5, 2, 2])
                rc1.markdown(f"**{s_name}**  \n{s_atype} · {s_date}")
                rc2.metric("R²", f"{s_r2:.3f}")
                bc1, bc2 = rc3.columns(2)
                if bc1.button("View", key=f"sv_view_{sid}"):
                    st.session_state.saved_view_id = sid
                    st.session_state.result = None
                    st.rerun()
                if bc2.button("✕", key=f"sv_del_{sid}"):
                    analysis.delete_finding(sid)
                    get_findings.clear()
                    if st.session_state.saved_view_id == sid:
                        st.session_state.saved_view_id = None
                    st.rerun()
        st.divider()

    # ── Saved analysis replay ─────────────────────────────────────
    if st.session_state.saved_view_id is not None:
        _sv_row = _saved[_saved["id"] == st.session_state.saved_view_id]
        if _sv_row.empty:
            st.session_state.saved_view_id = None
        else:
            _sv = _sv_row.iloc[0]
            _sv_a  = _sv["variable_a"]; _sv_b = _sv["variable_b"]
            _sv_lag = int(_sv["lag_days"]) if _sv["lag_days"] else 0
            _sv_atype = _sv["analysis_type"]
            _sv_cutoff = pd.Timestamp(_sv["calculated_at"]).tz_localize(None).normalize()
            _sv_al = col_label(_sv_a); _sv_bl = col_label(_sv_b) if _sv_b else None

            if st.button("← Back to Explorer"):
                st.session_state.saved_view_id = None
                st.rerun()

            st.markdown(f"### {_sv_a} {'→ ' + _sv_b if _sv_b else ''}  \n"
                        f"<span style='opacity:0.6'>{_sv_atype} · saved {_sv_cutoff.strftime('%Y-%m-%d')}</span>",
                        unsafe_allow_html=True)

            _hist = get_data(0)
            _hist = _hist[_hist.index <= _sv_cutoff]

            with st.spinner("Reconstructing…"):
                if   _sv_atype == "Pearson Correlation":  _sr = analysis.pearson_correlation(_hist, _sv_a, _sv_b)
                elif _sv_atype == "Spearman Correlation": _sr = analysis.spearman_correlation(_hist, _sv_a, _sv_b)
                elif _sv_atype == "Lagged Correlation":   _sr = analysis.lagged_correlation(_hist, _sv_a, _sv_b, _sv_lag)
                elif _sv_atype == "30-Day Trend (OLS)":   _sr = analysis.ols_trend(_hist, _sv_a)
                elif _sv_atype == "Rolling Average":       _sr = analysis.rolling_avg_correlation(_hist, _sv_a, _sv_b, 7)
                else: _sr = {"error": f"Chart replay not supported for {_sv_atype}"}

            if "error" in _sr:
                st.warning(_sr["error"])
            else:
                stat_bar(_sr.get("r2"), _sr.get("p_value"), _sr.get("coefficient"), _sr.get("n"), _sr.get("label"))
                if _sv_atype in ("Pearson Correlation", "Spearman Correlation"):
                    st.plotly_chart(scatter_ols(_sr["series_a"], _sr["series_b"],
                        _sr["coefficient"], _sr["intercept"], _sv_al, _sv_bl), width="stretch")
                elif _sv_atype == "Lagged Correlation":
                    st.plotly_chart(scatter_ols(_sr["series_a"], _sr["series_b"],
                        _sr["coefficient"], _sr["intercept"],
                        f"{_sv_al} (day 0)", f"{_sv_bl} (+{_sv_lag}d)"), width="stretch")
                elif _sv_atype == "30-Day Trend (OLS)":
                    _fig = go.Figure()
                    _fig.add_trace(go.Scatter(x=_sr["series"].index, y=_sr["series"].values,
                        mode="lines+markers", name=_sv_al, line=dict(color=BLUE), marker=dict(size=5)))
                    _fig.add_trace(go.Scatter(x=_sr["fitted"].index, y=_sr["fitted"].values,
                        mode="lines", name="Trend", line=dict(color=GREEN, dash="dash", width=2)))
                    _fig.update_layout(xaxis_title="Date", yaxis_title=_sv_al, height=450, margin=dict(t=20))
                    st.plotly_chart(_fig, width="stretch")
                elif _sv_atype == "Rolling Average":
                    _fig = make_subplots(specs=[[{"secondary_y": True}]])
                    _fig.add_trace(go.Scatter(x=_sr["series_a"].index, y=_sr["series_a"].values,
                        name=_sv_al, line=dict(color=BLUE)), secondary_y=False)
                    _fig.add_trace(go.Scatter(x=_sr["series_b"].index, y=_sr["series_b"].values,
                        name=_sv_bl, line=dict(color=ORANGE)), secondary_y=True)
                    _fig.update_layout(height=450, margin=dict(t=20))
                    st.plotly_chart(_fig, width="stretch")
            st.stop()

    result = st.session_state.result; rtype = st.session_state.result_type; meta = st.session_state.result_meta

    if result is None:
        st.info("Configure your analysis in the sidebar and click **Run Analysis**.")
        st.stop()
    if "error" in result:
        st.error(result["error"]); st.stop()

    if   rtype in MULTI_PRED:  title = f"Multiple OLS — {meta['outcome_label']}"
    elif rtype in SINGLE_VAR:  title = f"{rtype} — {meta['var_a_label']}"
    else:
        title = f"{meta['var_a_label']}  ×  {meta['var_b_label']}"
        if rtype == "Lagged Correlation" and meta["lag"]: title += f"  (lag {meta['lag']}d)"
        elif rtype == "Rolling Average": title += f"  ({meta['window']}-day rolling)"
    st.title(title)

    if rtype in ("Pearson Correlation", "Spearman Correlation"):
        stat_bar(result["r2"], result["p_value"], result["coefficient"], result["n"], result["label"])
        st.plotly_chart(scatter_ols(result["series_a"], result["series_b"], result["coefficient"],
            result["intercept"], meta["var_a_label"], meta["var_b_label"]), width="stretch")

    elif rtype == "Lagged Correlation":
        stat_bar(result["r2"], result["p_value"], result["coefficient"], result["n"], result["label"])
        st.plotly_chart(scatter_ols(result["series_a"], result["series_b"], result["coefficient"],
            result["intercept"], f"{meta['var_a_label']} (day 0)",
            f"{meta['var_b_label']} (+{meta['lag']}d)"), width="stretch")

    elif rtype == "Rolling Average":
        stat_bar(result["r2"], result["p_value"], result["coefficient"], result["n"], result["label"])
        a, b = result["series_a"], result["series_b"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=a.index, y=a.values, name=meta["var_a_label"], line=dict(color=BLUE)), secondary_y=False)
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name=meta["var_b_label"], line=dict(color=ORANGE)), secondary_y=True)
        fig.update_layout(height=450, margin=dict(t=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    elif rtype == "30-Day Trend (OLS)":
        coef = result["coefficient"]
        stat_bar(result["r2"], result["p_value"], coef, result["n"], result["label"],
                 extra=[("↑↓ per day", f"{coef:+.4f}")])
        series, fitted = result["series"], result["fitted"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                                 name=meta["var_a_label"], line=dict(color=BLUE), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode="lines", name="Trend",
                                 line=dict(color=GREEN, dash="dash", width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                          height=450, margin=dict(t=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    elif rtype == "Multiple OLS Regression":
        stat_bar(r2=result["r2"], p=result["p_value"], n=result["n"],
                 extra=[("R² adj", f"{result['r2_adj']:.4f}")])
        mn = min(float(result["actual"].min()), float(result["fitted"].min()))
        mx = max(float(result["actual"].max()), float(result["fitted"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["actual"].values, y=result["fitted"].values, mode="markers",
                                 marker=dict(color=BLUE, size=7, opacity=0.8),
                                 hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=GREEN, dash="dash"), showlegend=False))
        ol = meta["outcome_label"]
        fig.update_layout(xaxis_title=f"Actual {ol}", yaxis_title=f"Predicted {ol}", height=450, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.subheader("Coefficients")
        pl = [col_label(p) for p in meta["predictors"]]
        st.dataframe(pd.DataFrame({"Variable": pl,
            "Coefficient": [result["coefficients"][p] for p in meta["predictors"]],
            "p-value":     [result["p_values"][p]     for p in meta["predictors"]],
            "Significant": ["✓" if result["p_values"][p] < 0.05 else "✗" for p in meta["predictors"]],
        }), use_container_width=True, hide_index=True)

    elif rtype == "Anomaly Detection":
        series, anom = result["series"], result["anomalies"]
        st.metric("Anomalies", f"{result['n_anomalies']} days")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                 name=meta["var_a_label"], line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=result["rolling_mean"].index, y=result["rolling_mean"].values,
                                 mode="lines", name="Baseline", line=dict(color=GREEN, dash="dot")))
        if anom.any():
            fig.add_trace(go.Scatter(x=series[anom].index, y=series[anom].values, mode="markers",
                                     name="Anomaly", text=series[anom].index.strftime("%Y-%m-%d"),
                                     hovertemplate="%{text}: %{y:.2f}<extra></extra>",
                                     marker=dict(color=RED, size=11, symbol="circle-open", line=dict(width=2))))
        fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                          height=450, margin=dict(t=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    elif rtype == "Forecast (7-Day)":
        series = result["series"]; fc = result["forecast"]
        proj = fc[~fc["ds"].isin(series.index)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                                 name="Historical", line=dict(color=BLUE), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=proj["ds"], y=proj["yhat"], mode="lines+markers",
                                 name="Forecast", line=dict(color=ORANGE, dash="dash"), marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=list(proj["ds"]) + list(proj["ds"])[::-1],
            y=list(proj["yhat_upper"]) + list(proj["yhat_lower"])[::-1],
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence"))
        fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                          height=450, margin=dict(t=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    elif rtype == "Decomposition":
        components = [("Observed", result["observed"], BLUE), ("Trend", result["trend"], GREEN),
                      ("Seasonal", result["seasonal"], ORANGE), ("Residual", result["residual"], RED)]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=[c[0] for c in components], vertical_spacing=0.07)
        for i, (_, s, color) in enumerate(components, 1):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     line=dict(color=color), showlegend=False), row=i, col=1)
        fig.update_layout(height=800, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")

    # Save to Findings
    saveable = rtype not in ("Anomaly Detection","Forecast (7-Day)","Decomposition","Multiple OLS Regression")
    if saveable:
        st.divider()
        if st.button("Save to Findings", type="secondary"):
            try:
                analysis.save_finding(
                    meta["var_a"], meta["var_b"],
                    float(result["r2"]), float(result["p_value"]),
                    float(result["coefficient"]),
                    int(meta.get("lag", 0) or 0), rtype, int(result["n"]),
                    pinned=True)
                get_findings.clear()
                st.toast("Saved to Findings.", icon="✅")
            except Exception as e:
                st.error(f"Save failed: {e}")

