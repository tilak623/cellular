"""
app.py - Streamlit dashboard for the cellular-traffic forecasting benchmark.

The dashboard is a *viewer* for the artifacts produced by ``train_models.py``
(results/test_predictions.csv and results/model_benchmark_results.csv).
It needs no heavy ML libraries, so it deploys quickly and cheaply.

Run locally (either way opens http://localhost:8501):
    python app.py
    streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BENCHMARK_CSV = RESULTS_DIR / "model_benchmark_results.csv"
PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"
PLOT_PNG = RESULTS_DIR / "actual_vs_predicted_72h.png"
METADATA_JSON = RESULTS_DIR / "run_metadata.json"

TARGET_LABEL = "M1 (macro base station)"
COLORS = {"Actual": "#000000", "SARIMA": "#1f77b4", "XGBoost": "#2ca02c", "LSTM": "#d62728"}
MODEL_ICONS = {"SARIMA": "📊", "XGBoost": "🌲", "LSTM": "🧠"}
MODEL_INFO = {
    "SARIMA": ("Statistical baseline (statsmodels). Seasonal ARIMA with a 24-hour season; parameters are "
               "estimated on the training set and the Kalman filter then produces rolling one-step-ahead "
               "forecasts over the test period. Captures the daily cycle but not the weekly pattern."),
    "XGBoost": ("Gradient-boosted trees on engineered features: lag_1, lag_24, lag_168, hour_of_day and "
                "day_of_week. The weekly lag and day-of-week features let it model weekday/weekend behaviour."),
    "LSTM": ("Deep-learning model (PyTorch). A 2-layer LSTM reads the previous 24 hours of MinMax-scaled "
             "traffic and predicts the next hour. Trained with early stopping on a chronological validation tail."),
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



def _running_under_streamlit() -> bool:
    """True when this script is being executed by the Streamlit server."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:  # older Streamlit without suppress_warning
            return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_under_streamlit():
    # `python app.py` -> start the Streamlit server on http://localhost:8501
    print("Launching dashboard at http://localhost:8501 (Ctrl+C to stop)...")
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        "--server.port", "8501", "--server.address", "localhost",
    ]))

st.set_page_config(page_title="Cellular Traffic Forecasting", page_icon="📡", layout="wide")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_predictions(path: str, mtime: float) -> pd.DataFrame:
    """`mtime` is only part of the cache key so the cache refreshes after retraining."""
    df = pd.read_csv(path, index_col="Timestamp", parse_dates=True)
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_benchmark(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def default_window_start(index: pd.DatetimeIndex, horizon: int) -> int:
    """First Monday 00:00 that leaves a full window (matches the paper's 72 h plot)."""
    for pos, ts in enumerate(index):
        if ts.dayofweek == 0 and ts.hour == 0 and pos + horizon <= len(index):
            return pos
    return 0


# --------------------------------------------------------------------------- #
# Load artifacts (or explain how to create them)
# --------------------------------------------------------------------------- #
if not (PREDICTIONS_CSV.is_file() and BENCHMARK_CSV.is_file()):
    st.title("📡 Cellular Network Traffic Forecasting")
    st.error("Benchmark artifacts not found in `results/`. Train the models first:")
    st.code("python train_models.py", language="bash")
    st.caption("This creates results/test_predictions.csv and results/model_benchmark_results.csv. "
               "Then reload this page.")
    st.stop()

preds = load_predictions(str(PREDICTIONS_CSV), PREDICTIONS_CSV.stat().st_mtime)
bench = load_benchmark(str(BENCHMARK_CSV), BENCHMARK_CSV.stat().st_mtime)
model_names = [c for c in preds.columns if c != "Actual"]
n_test = len(preds)

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Forecast controls")

horizon = st.sidebar.slider(
    "Forecast horizon (hours)", min_value=24, max_value=min(168, n_test), value=min(72, n_test), step=24,
    help="Length of the test window shown in the chart and used for the live metrics.",
)

max_start = max(0, n_test - horizon)
if max_start > 0:
    start = st.sidebar.slider(
        "Window start (hours after test start)", min_value=0, max_value=max_start,
        value=default_window_start(preds.index, horizon), step=1,
        help="Slide to move the forecast window across the 1,752-hour test period.",
    )
else:
    start = 0

selected = st.sidebar.multiselect("Models to display", model_names, default=model_names)
if not selected:
    st.sidebar.warning("Select at least one model.")

window = preds.iloc[start:start + horizon]
st.sidebar.caption(f"Window: **{window.index[0]:%Y-%m-%d %H:%M}** → **{window.index[-1]:%Y-%m-%d %H:%M}**")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📡 Cellular Network Traffic Forecasting")
st.caption(
    f"Comparative performance of SARIMA, XGBoost and LSTM on hourly traffic of **{TARGET_LABEL}** · "
    f"test period {preds.index[0]:%Y-%m-%d} → {preds.index[-1]:%Y-%m-%d} ({n_test:,} hourly records)"
)

# Live metrics for the selected window
actual = window["Actual"].to_numpy(dtype=float)
live_rows = []
for name in model_names:
    y_hat = window[name].to_numpy(dtype=float)
    live_rows.append({"Model": name, "RMSE_window": rmse(actual, y_hat), "MAE_window": mae(actual, y_hat)})
live = pd.DataFrame(live_rows)

summary = live.merge(
    bench.rename(columns={"RMSE": "RMSE_full", "MAE": "MAE_full"}),
    on="Model", how="left",
)

# KPI row
best = live.loc[live["RMSE_window"].idxmin()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model (window RMSE)", str(best["Model"]))
c2.metric("RMSE (MB)", f"{best['RMSE_window']:.1f}")
c3.metric("MAE (MB)", f"{best['MAE_window']:.1f}")
c4.metric("Horizon", f"{horizon} h")

all_tabs = st.tabs(
    ["📈 Overview"]
    + [f"{MODEL_ICONS.get(m, '🔹')} {m}" for m in model_names]
    + ["🏁 Benchmark", "ℹ️ About"]
)
tab_forecast = all_tabs[0]
model_tabs = dict(zip(model_names, all_tabs[1:1 + len(model_names)]))
tab_benchmark, tab_about = all_tabs[-2], all_tabs[-1]

# --------------------------------------------------------------------------- #
# Overview tab - all models together
# --------------------------------------------------------------------------- #
with tab_forecast:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window.index, y=window["Actual"], mode="lines", name="Actual",
        line=dict(color=COLORS["Actual"], width=3),
    ))
    for name in selected:
        fig.add_trace(go.Scatter(
            x=window.index, y=window[name], mode="lines", name=name,
            line=dict(color=COLORS.get(name), width=2, dash="dash"),
        ))
    fig.update_layout(
        title=f"Actual vs. predicted traffic - {horizon}-hour window",
        xaxis_title="Timestamp", yaxis_title="Traffic (MB)",
        hovermode="x unified", template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model summary")
    st.caption("Window metrics update live with the sliders; full-test metrics, training time and "
               "latency come from the benchmark run.")
    st.dataframe(
        summary[["Model", "RMSE_window", "MAE_window", "RMSE_full", "MAE_full",
                 "Training_Time_s", "Inference_Latency_ms_per_record"]],
        hide_index=True, use_container_width=True,
        column_config={
            "RMSE_window": st.column_config.NumberColumn("RMSE - window (MB)", format="%.2f"),
            "MAE_window": st.column_config.NumberColumn("MAE - window (MB)", format="%.2f"),
            "RMSE_full": st.column_config.NumberColumn("RMSE - full test (MB)", format="%.2f"),
            "MAE_full": st.column_config.NumberColumn("MAE - full test (MB)", format="%.2f"),
            "Training_Time_s": st.column_config.NumberColumn("Training time (s)", format="%.2f"),
            "Inference_Latency_ms_per_record": st.column_config.NumberColumn(
                "Inference latency (ms/record)", format="%.5f"),
        },
    )

    st.download_button(
        "⬇️ Download window predictions (CSV)",
        data=window.to_csv().encode("utf-8"),
        file_name=f"predictions_{window.index[0]:%Y%m%d_%H%M}_{horizon}h.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Benchmark tab - full test set
# --------------------------------------------------------------------------- #
with tab_benchmark:
    st.subheader("Benchmark on the full test set")
    st.dataframe(
        bench, hide_index=True, use_container_width=True,
        column_config={
            "RMSE": st.column_config.NumberColumn("RMSE (MB)", format="%.2f"),
            "MAE": st.column_config.NumberColumn("MAE (MB)", format="%.2f"),
            "Training_Time_s": st.column_config.NumberColumn("Training time (s)", format="%.2f"),
            "Inference_Latency_ms_per_record": st.column_config.NumberColumn(
                "Inference latency (ms/record)", format="%.5f"),
            "Test_Records": st.column_config.NumberColumn("Test records", format="%d"),
        },
    )

    left, right = st.columns(2)
    err_long = bench.melt(id_vars="Model", value_vars=["RMSE", "MAE"], var_name="Metric", value_name="MB")
    fig_err = px.bar(err_long, x="Model", y="MB", color="Metric", barmode="group",
                     title="Prediction error (lower is better)", template="plotly_white")
    left.plotly_chart(fig_err, use_container_width=True)

    fig_time = px.bar(bench, x="Model", y="Training_Time_s", color="Model", log_y=True,
                      color_discrete_map=COLORS, title="Training time (s, log scale)",
                      template="plotly_white")
    fig_time.update_layout(showlegend=False)
    right.plotly_chart(fig_time, use_container_width=True)

    left2, right2 = st.columns(2)
    fig_lat = px.bar(bench, x="Model", y="Inference_Latency_ms_per_record", color="Model", log_y=True,
                     color_discrete_map=COLORS, title="Inference latency (ms/record, log scale)",
                     template="plotly_white")
    fig_lat.update_layout(showlegend=False)
    left2.plotly_chart(fig_lat, use_container_width=True)

    # Error by hour of day on the full test set
    hourly = []
    for name in model_names:
        err = (preds["Actual"] - preds[name]).abs()
        by_hour = err.groupby(preds.index.hour).mean()
        hourly.append(pd.DataFrame({"Hour": by_hour.index, "MAE": by_hour.values, "Model": name}))
    fig_hour = px.line(pd.concat(hourly), x="Hour", y="MAE", color="Model", markers=True,
                       color_discrete_map=COLORS, title="MAE by hour of day (full test set)",
                       template="plotly_white")
    right2.plotly_chart(fig_hour, use_container_width=True)

    if PLOT_PNG.is_file():
        st.subheader("Static figure used in the paper")
        st.image(str(PLOT_PNG), use_container_width=True)

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
**Dataset** - hourly traffic logs (MB) of a heterogeneous cellular network; this study forecasts macro cell **M1**.

**Split** - chronological 80 % train / 20 % test (no shuffling).

**Models**
- **SARIMA** (statsmodels) - seasonal period 24 h; parameters fitted on the training data only.
- **XGBoost** - features: `lag_1`, `lag_24`, `lag_168`, `hour_of_day`, `day_of_week`.
- **LSTM** (PyTorch) - 24-hour input window, MinMax-scaled (scaler fitted on training data only).

**Evaluation protocol** - all models produce *one-step-ahead* forecasts for the same test hours,
using true past observations as inputs. Predictions are clipped at 0 MB.
Inference latency is the amortised batch time per record.

Use the **model tabs** (SARIMA, XGBoost, LSTM) for per-model diagnostics.

The horizon slider selects how many consecutive test hours are displayed and scored; it does not
re-run the models.
"""
    )


# --------------------------------------------------------------------------- #
# Per-model tabs (SARIMA / XGBoost / LSTM)
# --------------------------------------------------------------------------- #
def load_metadata() -> dict:
    try:
        return json.loads(METADATA_JSON.read_text())
    except Exception:
        return {}


def render_model_tab(name: str) -> None:
    color = COLORS.get(name, "#7f7f7f")
    resid_window = window["Actual"] - window[name]
    resid_full = preds["Actual"] - preds[name]
    live_row = live.loc[live["Model"] == name].iloc[0]
    bench_row = bench.loc[bench["Model"] == name]
    k = f"{name}"  # unique widget/chart key prefix

    st.markdown(f"### {MODEL_ICONS.get(name, '')} {name}")
    if name in MODEL_INFO:
        st.write(MODEL_INFO[name])

    # KPI row: live (window) + benchmark (full test)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"RMSE - {horizon} h window", f"{live_row['RMSE_window']:.1f} MB")
    k2.metric(f"MAE - {horizon} h window", f"{live_row['MAE_window']:.1f} MB")
    if len(bench_row):
        b = bench_row.iloc[0]
        k3.metric("RMSE - full test", f"{b['RMSE']:.1f} MB")
        k4.metric("MAE - full test", f"{b['MAE']:.1f} MB")
        k5.metric("Training time", f"{b['Training_Time_s']:.1f} s")
        k6.metric("Latency", f"{b['Inference_Latency_ms_per_record']:.4f} ms/rec")

    # Forecast for the selected window
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window.index, y=window["Actual"], mode="lines", name="Actual",
                             line=dict(color=COLORS["Actual"], width=3)))
    fig.add_trace(go.Scatter(x=window.index, y=window[name], mode="lines", name=name,
                             line=dict(color=color, width=2, dash="dash")))
    fig.update_layout(title=f"{name}: actual vs. predicted - {horizon}-hour window",
                      xaxis_title="Timestamp", yaxis_title="Traffic (MB)", hovermode="x unified",
                      template="plotly_white", height=430,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=10, r=10, t=70, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"{k}_forecast")

    # Residuals in the window
    fig_res = go.Figure(go.Bar(x=window.index, y=resid_window, marker_color=color, name="Residual"))
    fig_res.update_layout(title="Residuals in window (Actual - Predicted; above 0 = under-forecast)",
                          xaxis_title="Timestamp", yaxis_title="MB", template="plotly_white",
                          height=300, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_res, use_container_width=True, key=f"{k}_residuals")

    # Scatter + histogram (full test set)
    left, right = st.columns(2)
    top = float(max(preds["Actual"].max(), preds[name].max()))
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=preds["Actual"], y=preds[name], mode="markers", name="Hourly points",
                                marker=dict(size=4, opacity=0.45, color=color)))
    fig_sc.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect forecast",
                                line=dict(color="black", dash="dot")))
    fig_sc.update_layout(title="Predicted vs. actual (full test set)", xaxis_title="Actual (MB)",
                         yaxis_title="Predicted (MB)", template="plotly_white", height=380,
                         margin=dict(l=10, r=10, t=60, b=10))
    left.plotly_chart(fig_sc, use_container_width=True, key=f"{k}_scatter")

    fig_hist = go.Figure(go.Histogram(x=resid_full, nbinsx=50, marker_color=color))
    fig_hist.update_layout(title="Residual distribution (full test set)", xaxis_title="Actual - Predicted (MB)",
                           yaxis_title="Hours", template="plotly_white", height=380,
                           margin=dict(l=10, r=10, t=60, b=10))
    right.plotly_chart(fig_hist, use_container_width=True, key=f"{k}_hist")

    # Error by hour of day and by weekday (full test set)
    abs_err = resid_full.abs()
    by_hour = abs_err.groupby(preds.index.hour).mean()
    by_day = abs_err.groupby(preds.index.dayofweek).mean().reindex(range(7))
    left2, right2 = st.columns(2)
    fig_h = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values, marker_color=color))
    fig_h.update_layout(title="MAE by hour of day", xaxis_title="Hour", yaxis_title="MAE (MB)",
                        template="plotly_white", height=330, margin=dict(l=10, r=10, t=60, b=10))
    left2.plotly_chart(fig_h, use_container_width=True, key=f"{k}_by_hour")
    fig_d = go.Figure(go.Bar(x=DAY_NAMES, y=by_day.values, marker_color=color))
    fig_d.update_layout(title="MAE by day of week", xaxis_title="Day", yaxis_title="MAE (MB)",
                        template="plotly_white", height=330, margin=dict(l=10, r=10, t=60, b=10))
    right2.plotly_chart(fig_d, use_container_width=True, key=f"{k}_by_day")

    # Details + download
    meta = load_metadata().get("models", {}).get(name)
    if meta:
        with st.expander("Hyperparameters and saved artifacts"):
            st.json(meta)
    export = window[["Actual", name]].assign(Residual=resid_window)
    st.download_button(
        f"⬇️ Download {name} window predictions (CSV)", data=export.to_csv().encode("utf-8"),
        file_name=f"{name.lower()}_{window.index[0]:%Y%m%d_%H%M}_{horizon}h.csv", mime="text/csv",
        key=f"{k}_download",
    )


for _name, _tab in model_tabs.items():
    with _tab:
        render_model_tab(_name)
