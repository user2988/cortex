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
    st.session_state.page = "Insights"
if "exp_detail_id" not in st.session_state:
    st.session_state.exp_detail_id = None
if "saved_view_id" not in st.session_state:
    st.session_state.saved_view_id = None

page = st.sidebar.radio("", ["Insights", "Experiments", "Explorer"],
                         key="page", label_visibility="collapsed")

if page != "Experiments":
    st.session_state.exp_detail_id = None

if st.sidebar.button("↺ Refresh", use_container_width=True):
    bust_cache()
    st.rerun()

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
                    st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(fig2, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
                        _sr["coefficient"], _sr["intercept"], _sv_al, _sv_bl), use_container_width=True)
                elif _sv_atype == "Lagged Correlation":
                    st.plotly_chart(scatter_ols(_sr["series_a"], _sr["series_b"],
                        _sr["coefficient"], _sr["intercept"],
                        f"{_sv_al} (day 0)", f"{_sv_bl} (+{_sv_lag}d)"), use_container_width=True)
                elif _sv_atype == "30-Day Trend (OLS)":
                    _fig = go.Figure()
                    _fig.add_trace(go.Scatter(x=_sr["series"].index, y=_sr["series"].values,
                        mode="lines+markers", name=_sv_al, line=dict(color=BLUE), marker=dict(size=5)))
                    _fig.add_trace(go.Scatter(x=_sr["fitted"].index, y=_sr["fitted"].values,
                        mode="lines", name="Trend", line=dict(color=GREEN, dash="dash", width=2)))
                    _fig.update_layout(xaxis_title="Date", yaxis_title=_sv_al, height=450, margin=dict(t=20))
                    st.plotly_chart(_fig, use_container_width=True)
                elif _sv_atype == "Rolling Average":
                    _fig = make_subplots(specs=[[{"secondary_y": True}]])
                    _fig.add_trace(go.Scatter(x=_sr["series_a"].index, y=_sr["series_a"].values,
                        name=_sv_al, line=dict(color=BLUE)), secondary_y=False)
                    _fig.add_trace(go.Scatter(x=_sr["series_b"].index, y=_sr["series_b"].values,
                        name=_sv_bl, line=dict(color=ORANGE)), secondary_y=True)
                    _fig.update_layout(height=450, margin=dict(t=20))
                    st.plotly_chart(_fig, use_container_width=True)
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
            result["intercept"], meta["var_a_label"], meta["var_b_label"]), use_container_width=True)

    elif rtype == "Lagged Correlation":
        stat_bar(result["r2"], result["p_value"], result["coefficient"], result["n"], result["label"])
        st.plotly_chart(scatter_ols(result["series_a"], result["series_b"], result["coefficient"],
            result["intercept"], f"{meta['var_a_label']} (day 0)",
            f"{meta['var_b_label']} (+{meta['lag']}d)"), use_container_width=True)

    elif rtype == "Rolling Average":
        stat_bar(result["r2"], result["p_value"], result["coefficient"], result["n"], result["label"])
        a, b = result["series_a"], result["series_b"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=a.index, y=a.values, name=meta["var_a_label"], line=dict(color=BLUE)), secondary_y=False)
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name=meta["var_b_label"], line=dict(color=ORANGE)), secondary_y=True)
        fig.update_layout(height=450, margin=dict(t=20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

    elif rtype == "Decomposition":
        components = [("Observed", result["observed"], BLUE), ("Trend", result["trend"], GREEN),
                      ("Seasonal", result["seasonal"], ORANGE), ("Residual", result["residual"], RED)]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=[c[0] for c in components], vertical_spacing=0.07)
        for i, (_, s, color) in enumerate(components, 1):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     line=dict(color=color), showlegend=False), row=i, col=1)
        fig.update_layout(height=800, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

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

