"""Cortex — Statistical Explorer (v2)"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import analysis

st.set_page_config(page_title="Cortex Explorer", layout="wide", page_icon="📊")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

ANALYSIS_TYPES = [
    "Pearson Correlation",
    "Spearman Correlation",
    "Lagged Correlation",
    "Rolling Average",
    "30-Day Trend (OLS)",
    "Multiple OLS Regression",
    "Anomaly Detection",
    "Forecast (7-Day)",
    "Decomposition",
]

SINGLE_VAR = {"30-Day Trend (OLS)", "Anomaly Detection", "Forecast (7-Day)", "Decomposition"}
MULTI_PRED = {"Multiple OLS Regression"}

# Variable option lists with category prefix for readability
BIO_OPTIONS = {f"[Bio]  {analysis.COL_LABELS[c]}": c for c in analysis.BIOMETRIC_COLS}
NUT_OPTIONS = {f"[Nut]  {analysis.COL_LABELS[c]}": c for c in analysis.NUTRITION_COLS}
ALL_OPTIONS = {**BIO_OPTIONS, **NUT_OPTIONS}
ALL_LABELS  = list(ALL_OPTIONS.keys())

COL_TO_LABEL = {v: k for k, v in ALL_OPTIONS.items()}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_data(days: int) -> pd.DataFrame:
    return analysis.load_data(days if days > 0 else None)


def data_summary(df: pd.DataFrame) -> str:
    n_bio = df[analysis.BIOMETRIC_COLS].dropna(how="all").shape[0]
    n_nut = df[analysis.NUTRITION_COLS].dropna(how="all").shape[0]
    date_range = f"{df.index.min().date()} → {df.index.max().date()}" if len(df) else "—"
    return f"**{n_bio}** biometric days · **{n_nut}** nutrition days  \n{date_range}"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.title("Cortex Explorer")
st.sidebar.caption("Statistical analysis engine — v2")
st.sidebar.divider()

page = st.sidebar.radio("", ["Explorer", "Top Findings"], horizontal=True,
                         label_visibility="collapsed")
st.sidebar.divider()

# ── Top Findings page ────────────────────────────────────────
if page == "Top Findings":
    if st.sidebar.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.title("Top Findings")

    findings = analysis.load_findings()

    if findings.empty:
        st.info("No findings yet. Run analyses in the Explorer and save results.")
        st.stop()

    for _, row in findings.iterrows():
        with st.container(border=True):
            a_lbl = analysis.COL_LABELS.get(row["variable_a"], row["variable_a"])
            b_lbl = analysis.COL_LABELS.get(row["variable_b"], row["variable_b"]) \
                    if row["variable_b"] else None
            title_txt = f"{'📌 ' if row['pinned'] else ''}" \
                        f"**{a_lbl}{f'  ×  {b_lbl}' if b_lbl else ''}**"
            meta_txt  = row["analysis_type"]
            if row["lag_days"]:
                meta_txt += f" · lag {int(row['lag_days'])}d"
            if row["sample_size"]:
                meta_txt += f" · {int(row['sample_size'])} days"

            hdr, btn_col = st.columns([8, 1])
            hdr.markdown(f"{title_txt}  \n{meta_txt}")
            if btn_col.button("✕", key=f"del_{row['id']}", help="Delete"):
                analysis.delete_finding(int(row["id"]))
                st.rerun()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",          f"{float(row['r_squared']):.4f}"  if row['r_squared']  else "—")
            c2.metric("p-value",     f"{float(row['p_value']):.4f}"    if row['p_value']    else "—")
            c3.metric("Coefficient", f"{float(row['coefficient']):.4f}" if row['coefficient'] else "—")
            c4.metric("Saved", row["calculated_at"].strftime("%d %b %Y") if row["calculated_at"] else "—")

            if row["r_squared"] and row["p_value"] and row["coefficient"]:
                st.caption(analysis.summary_label(
                    float(row["r_squared"]), float(row["p_value"]), float(row["coefficient"])
                ))

    st.stop()

# ── Explorer page continues below ────────────────────────────

analysis_type = st.sidebar.selectbox("Analysis Type", ANALYSIS_TYPES)

days_map = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90, "All data": 0}
days_label = st.sidebar.selectbox("Data Range", list(days_map.keys()), index=3)
days = days_map[days_label]

df = get_data(days)
st.sidebar.caption(data_summary(df))

if st.sidebar.button("↺ Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# Variable selectors — layout varies by analysis type
if analysis_type in MULTI_PRED:
    pred_labels   = st.sidebar.multiselect("Predictors (Variable A)", ALL_LABELS,
                                            default=ALL_LABELS[:2])
    outcome_label = st.sidebar.selectbox("Outcome (Variable B)", ALL_LABELS)
    predictors    = [ALL_OPTIONS[l] for l in pred_labels]
    outcome       = ALL_OPTIONS[outcome_label]
    var_a = var_b = None

elif analysis_type in SINGLE_VAR:
    var_a_label = st.sidebar.selectbox("Variable", ALL_LABELS)
    var_a       = ALL_OPTIONS[var_a_label]
    var_b       = None
    if analysis_type == "Decomposition":
        period = st.sidebar.selectbox("Seasonality Period (days)", [7, 14, 30])
    elif analysis_type == "Anomaly Detection":
        window    = st.sidebar.slider("Baseline Window (days)", 14, 60, 30)
        threshold = st.sidebar.slider("Threshold (SD)", 1.0, 3.0, 1.5, 0.1)

else:
    var_a_label = st.sidebar.selectbox("Variable A", ALL_LABELS, index=0)
    var_b_label = st.sidebar.selectbox("Variable B", ALL_LABELS, index=1)
    var_a = ALL_OPTIONS[var_a_label]
    var_b = ALL_OPTIONS[var_b_label]
    if analysis_type == "Lagged Correlation":
        lag         = st.sidebar.selectbox("Lag (days — A predicts B after N days)", [0, 1, 2, 3], index=1)
        corr_method = st.sidebar.radio("Method", ["Pearson", "Spearman"],
                                       horizontal=True).lower()
    elif analysis_type == "Rolling Average":
        window      = st.sidebar.selectbox("Rolling Window (days)", [7, 14])
        corr_method = st.sidebar.radio("Method", ["Pearson", "Spearman"],
                                       horizontal=True).lower()
    elif analysis_type in ("Pearson Correlation", "Spearman Correlation"):
        pass  # no extra params

st.sidebar.divider()
run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE — persist result across widget interactions
# ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result      = None
    st.session_state.result_type = None
    st.session_state.result_meta = {}

if run_clicked:
    with st.spinner("Running analysis…"):
        if analysis_type == "Pearson Correlation":
            res = analysis.pearson_correlation(df, var_a, var_b)
        elif analysis_type == "Spearman Correlation":
            res = analysis.spearman_correlation(df, var_a, var_b)
        elif analysis_type == "Lagged Correlation":
            res = analysis.lagged_correlation(df, var_a, var_b, lag, corr_method)
        elif analysis_type == "Rolling Average":
            res = analysis.rolling_avg_correlation(df, var_a, var_b, window, corr_method)
        elif analysis_type == "30-Day Trend (OLS)":
            res = analysis.ols_trend(df, var_a)
        elif analysis_type == "Multiple OLS Regression":
            res = analysis.multiple_ols(df, predictors, outcome)
        elif analysis_type == "Anomaly Detection":
            res = analysis.anomaly_detection(df, var_a, window, threshold)
        elif analysis_type == "Forecast (7-Day)":
            res = analysis.forecast(df, var_a)
        elif analysis_type == "Decomposition":
            res = analysis.decompose(df, var_a, period)
        else:
            res = {"error": "Unknown analysis type"}

    st.session_state.result      = res
    st.session_state.result_type = analysis_type
    st.session_state.result_meta = {
        "var_a": var_a, "var_b": var_b,
        "var_a_label": COL_TO_LABEL.get(var_a, var_a) if var_a else None,
        "var_b_label": COL_TO_LABEL.get(var_b, var_b) if var_b else None,
        "analysis_type": analysis_type,
        "lag": locals().get("lag", 0),
        "window": locals().get("window"),
        "period": locals().get("period"),
        "threshold": locals().get("threshold"),
        "predictors": locals().get("predictors"),
        "outcome": locals().get("outcome"),
        "outcome_label": locals().get("outcome_label"),
    }


# ─────────────────────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────────────────────

result = st.session_state.result
rtype  = st.session_state.result_type
meta   = st.session_state.result_meta

if result is None:
    st.title("Statistical Explorer")
    st.info("Configure your analysis in the sidebar and click **Run Analysis**.")
    st.stop()

if "error" in result:
    st.title("Statistical Explorer")
    st.error(result["error"])
    st.stop()

# Build a readable title
if rtype in MULTI_PRED:
    title = f"Multiple OLS — {meta['outcome_label']}"
elif rtype in SINGLE_VAR:
    title = f"{rtype} — {meta['var_a_label']}"
else:
    title = f"{meta['var_a_label']}  ×  {meta['var_b_label']}"
    if rtype == "Lagged Correlation" and meta["lag"]:
        title += f"  (lag {meta['lag']}d)"
    elif rtype == "Rolling Average":
        title += f"  ({meta['window']}-day rolling)"

st.title(title)

# ─── stat summary bar ────────────────────────────────────────

def stat_bar(r2=None, p=None, coef=None, n=None, label=None, extra_cols=None):
    cols = st.columns(4 + (len(extra_cols) if extra_cols else 0))
    idx = 0
    if r2 is not None:
        cols[idx].metric("R²", f"{r2:.4f}"); idx += 1
    if p is not None:
        cols[idx].metric("p-value", f"{p:.4f}"); idx += 1
    if coef is not None:
        cols[idx].metric("Coefficient", f"{coef:.4f}"); idx += 1
    if n is not None:
        cols[idx].metric("Data Points", n); idx += 1
    if extra_cols:
        for lbl, val in extra_cols:
            cols[idx].metric(lbl, val); idx += 1
    if label:
        st.markdown(f"**{label}**")


# ─── scatter helper ──────────────────────────────────────────

def scatter_with_ols(a: pd.Series, b: pd.Series, coef: float, intercept: float,
                     x_label: str, y_label: str) -> go.Figure:
    x_range = np.linspace(float(a.min()), float(a.max()), 100)
    y_line  = coef * x_range + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a.values, y=b.values, mode="markers",
        text=a.index.strftime("%Y-%m-%d"),
        hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
        marker=dict(color=BLUE, size=7, opacity=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_line, mode="lines",
        line=dict(color=GREEN, dash="dash", width=2),
        showlegend=False,
    ))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, height=450,
                      margin=dict(t=20))
    return fig


# ─── render by analysis type ─────────────────────────────────

if rtype in ("Pearson Correlation", "Spearman Correlation"):
    stat_bar(result["r2"], result["p_value"], result["coefficient"],
             result["n"], result["label"])
    st.plotly_chart(scatter_with_ols(
        result["series_a"], result["series_b"],
        result["coefficient"], result["intercept"],
        meta["var_a_label"], meta["var_b_label"],
    ), use_container_width=True)

elif rtype == "Lagged Correlation":
    lag = meta["lag"]
    stat_bar(result["r2"], result["p_value"], result["coefficient"],
             result["n"], result["label"])
    x_lbl = f"{meta['var_a_label']} (day 0)"
    y_lbl = f"{meta['var_b_label']} (+{lag}d)" if lag else meta["var_b_label"]
    st.plotly_chart(scatter_with_ols(
        result["series_a"], result["series_b"],
        result["coefficient"], result["intercept"],
        x_lbl, y_lbl,
    ), use_container_width=True)

elif rtype == "Rolling Average":
    stat_bar(result["r2"], result["p_value"], result["coefficient"],
             result["n"], result["label"])
    a, b = result["series_a"], result["series_b"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=a.index, y=a.values, name=meta["var_a_label"],
                             line=dict(color=BLUE)), secondary_y=False)
    fig.add_trace(go.Scatter(x=b.index, y=b.values, name=meta["var_b_label"],
                             line=dict(color=ORANGE)), secondary_y=True)
    fig.update_layout(height=450, margin=dict(t=20),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text=meta["var_a_label"], secondary_y=False)
    fig.update_yaxes(title_text=meta["var_b_label"], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

elif rtype == "30-Day Trend (OLS)":
    coef = result["coefficient"]
    direction = "↑" if coef > 0 else "↓"
    stat_bar(result["r2"], result["p_value"], coef, result["n"], result["label"],
             extra_cols=[(f"{direction} per day", f"{abs(coef):.4f}")])
    series, fitted = result["series"], result["fitted"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             mode="lines+markers", name=meta["var_a_label"],
                             line=dict(color=BLUE), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values,
                             mode="lines", name="OLS trend",
                             line=dict(color=GREEN, dash="dash", width=2)))
    fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                      height=450, margin=dict(t=20),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

elif rtype == "Multiple OLS Regression":
    stat_bar(result["r2"], result["p_value"], n=result["n"],
             extra_cols=[("R² (adj)", f"{result['r2_adj']:.4f}")])
    fig = go.Figure()
    mn = min(float(result["actual"].min()), float(result["fitted"].min()))
    mx = max(float(result["actual"].max()), float(result["fitted"].max()))
    fig.add_trace(go.Scatter(
        x=result["actual"].values, y=result["fitted"].values,
        mode="markers", marker=dict(color=BLUE, size=7, opacity=0.8),
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color=GREEN, dash="dash"), showlegend=False))
    outcome_lbl = meta["outcome_label"].replace("[Nut]  ", "").replace("[Bio]  ", "")
    fig.update_layout(xaxis_title=f"Actual {outcome_lbl}",
                      yaxis_title=f"Predicted {outcome_lbl}",
                      height=450, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Coefficients")
    pred_labels_clean = [COL_TO_LABEL.get(p, p).replace("[Nut]  ", "").replace("[Bio]  ", "")
                         for p in meta["predictors"]]
    coef_df = pd.DataFrame({
        "Variable":    pred_labels_clean,
        "Coefficient": [result["coefficients"][p] for p in meta["predictors"]],
        "p-value":     [result["p_values"][p] for p in meta["predictors"]],
        "Significant": ["✓" if result["p_values"][p] < 0.05 else "✗"
                         for p in meta["predictors"]],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

elif rtype == "Anomaly Detection":
    n_anom = result["n_anomalies"]
    st.metric("Anomalies detected", f"{n_anom} day{'s' if n_anom != 1 else ''}")
    st.caption(f">{meta['threshold']} SD from {meta['window']}-day rolling baseline")
    series = result["series"]
    anom   = result["anomalies"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             mode="lines", name=meta["var_a_label"],
                             line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=result["rolling_mean"].index,
                             y=result["rolling_mean"].values,
                             mode="lines", name=f"{meta['window']}-day baseline",
                             line=dict(color=GREEN, dash="dot")))
    if anom.any():
        fig.add_trace(go.Scatter(
            x=series[anom].index, y=series[anom].values,
            mode="markers", name="Anomaly",
            text=series[anom].index.strftime("%Y-%m-%d"),
            hovertemplate="%{text}: %{y:.2f}<extra></extra>",
            marker=dict(color=RED, size=11, symbol="circle-open", line=dict(width=2)),
        ))
    fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                      height=450, margin=dict(t=20),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

elif rtype == "Forecast (7-Day)":
    series = result["series"]
    fc     = result["forecast"]
    hist   = fc[fc["ds"].isin(series.index)]
    proj   = fc[~fc["ds"].isin(series.index)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             mode="lines+markers", name="Historical",
                             line=dict(color=BLUE), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=proj["ds"], y=proj["yhat"],
                             mode="lines+markers", name="Forecast",
                             line=dict(color=ORANGE, dash="dash"),
                             marker=dict(size=6)))
    # Confidence interval as filled band
    x_band = list(proj["ds"]) + list(proj["ds"])[::-1]
    y_band = list(proj["yhat_upper"]) + list(proj["yhat_lower"])[::-1]
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself",
                             fillcolor="rgba(255,127,14,0.15)",
                             line=dict(color="rgba(0,0,0,0)"),
                             name="Confidence interval"))
    fig.update_layout(xaxis_title="Date", yaxis_title=meta["var_a_label"],
                      height=450, margin=dict(t=20),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Prophet forecast based on {result['n']} days of historical data.")

elif rtype == "Decomposition":
    components = [
        ("Observed",  result["observed"],  BLUE),
        ("Trend",     result["trend"],     GREEN),
        ("Seasonal",  result["seasonal"],  ORANGE),
        ("Residual",  result["residual"],  RED),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[c[0] for c in components],
                        vertical_spacing=0.07)
    for i, (name, s, color) in enumerate(components, 1):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(color=color), showlegend=False), row=i, col=1)
    fig.update_layout(height=800, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Additive decomposition · period={meta['period']} days · n={result['n']}")


# ─────────────────────────────────────────────────────────────
# SAVE TO FINDINGS
# ─────────────────────────────────────────────────────────────

saveable = rtype not in ("Anomaly Detection", "Forecast (7-Day)",
                          "Decomposition", "Multiple OLS Regression")

if saveable:
    st.divider()
    col1, col2 = st.columns([1, 4])
    if col1.button("Save to Findings", type="secondary"):
        try:
            analysis.save_finding(
                variable_a    = meta["var_a"],
                variable_b    = meta["var_b"],
                r_squared     = result["r2"],
                p_value       = result["p_value"],
                coefficient   = result["coefficient"],
                lag_days      = meta.get("lag", 0) or 0,
                analysis_type = rtype,
                sample_size   = result["n"],
                pinned        = True,
            )
            col2.success("Saved to findings table.")
        except Exception as e:
            col2.error(f"Save failed: {e}")
