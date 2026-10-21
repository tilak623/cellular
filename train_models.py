"""
train_models.py
===============
End-to-end benchmark of three forecasting families on the HetNet traffic
column ``M1``:

    1. SARIMA   (statsmodels)  - statistical baseline
    2. XGBoost  (xgboost)      - gradient-boosted trees on lag features
    3. LSTM     (PyTorch)      - recurrent network on 24-hour windows

For every model the script records RMSE, MAE, training time (s) and inference
latency (ms / record), then writes:

    results/model_benchmark_results.csv      comparison table
    results/test_predictions.csv             actual + predictions on the test set
    results/actual_vs_predicted_72h.png      72-hour overlay plot (all models)
    results/run_metadata.json                configuration + library versions
    results/training.log                     full log
    exports/sarima_model.pkl, sarima_params.json
    exports/xgboost_model.json
    exports/lstm_model.pt, lstm_scaler.joblib

Evaluation protocol (identical for all models)
----------------------------------------------
* Chronological 80 / 20 split; every model is scored on the same 1,752 test hours.
* One-step-ahead forecasts: each prediction may use all *past* observations
  (true history), but never the value being predicted.
    - SARIMA : parameters fitted on the training set only, then the Kalman filter
               is run over the test period without re-estimation
               (``append(..., refit=False)``) -> rolling 1-step-ahead predictions.
    - XGBoost: lag_1 / lag_24 / lag_168 are true past values.
    - LSTM   : the 24 preceding true hours form the input window.
* Predictions are clipped at 0 MB (traffic cannot be negative).
* Inference latency = wall-clock time to predict the whole test set divided by
  the number of records (batch inference, amortised ms / record).

Usage
-----
    python train_models.py                    # full benchmark
    python train_models.py --epochs 30        # shorter LSTM training
    python train_models.py --skip-sarima      # skip the slowest model
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend (servers / CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_loader import (  # noqa: E402
    FEATURE_COLUMNS,
    LSTM_WINDOW,
    PROJECT_ROOT,
    TARGET_COL,
    TRAIN_RATIO,
    get_split_index,
    load_target_series,
    prepare_lstm_data,
    split_series,
    split_xgb_features,
)

# PyTorch is optional at import time so the other models still run without it.
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

logger = logging.getLogger("train_models")

EXPORT_DIR = PROJECT_ROOT / "exports"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "model_benchmark_results.csv"
PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"
PLOT_PATH = RESULTS_DIR / "actual_vs_predicted_72h.png"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
LOG_PATH = RESULTS_DIR / "training.log"

MODEL_COLORS = {"SARIMA": "#1f77b4", "XGBoost": "#2ca02c", "LSTM": "#d62728"}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
@dataclass
class ModelRun:
    """Everything produced by training + evaluating one model."""
    name: str
    predictions: pd.Series              # indexed by test timestamps, in MB
    train_time_s: float
    latency_ms_per_record: float
    artifacts: List[str] = field(default_factory=list)
    hyperparameters: Dict = field(default_factory=dict)


def setup_logging() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def time_call(fn: Callable, repeats: int = 1):
    """Run ``fn`` ``repeats`` times; return (last_result, mean_seconds)."""
    durations, result = [], None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return result, float(np.mean(durations))


def parse_int_tuple(text: str, expected_len: int, name: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers, got '{text}'") from exc
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} needs {expected_len} integers, got {len(values)}.")
    return values


# --------------------------------------------------------------------------- #
# 1) SARIMA
# --------------------------------------------------------------------------- #
def run_sarima(
    train: pd.Series,
    test: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    maxiter: int,
) -> ModelRun:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    logger.info("SARIMA%s x %s: fitting on %d hours (can take several minutes)...",
                order, seasonal_order, len(train))

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence / non-invertibility chatter
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=maxiter)
    train_time = time.perf_counter() - start
    logger.info("SARIMA fitted in %.1f s (AIC=%.1f)", train_time, fitted.aic)

    # Rolling one-step-ahead forecasts over the test period (no re-estimation).
    def _predict() -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = fitted.append(test, refit=False)
            pred = extended.get_prediction(
                start=len(train), end=len(train) + len(test) - 1, dynamic=False
            ).predicted_mean
        return np.asarray(pred, dtype=float)

    raw_pred, infer_seconds = time_call(_predict, repeats=1)
    if len(raw_pred) != len(test):
        raise RuntimeError(f"SARIMA returned {len(raw_pred)} predictions for {len(test)} test rows.")
    predictions = pd.Series(np.clip(raw_pred, 0.0, None), index=test.index, name="SARIMA")

    artifacts = []
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    params_path = EXPORT_DIR / "sarima_params.json"
    params_path.write_text(json.dumps({
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "params": {str(k): float(v) for k, v in fitted.params.items()},
        "aic": float(fitted.aic),
        "n_train": int(len(train)),
    }, indent=2))
    artifacts.append(str(params_path.relative_to(PROJECT_ROOT)))
    try:
        pkl_path = EXPORT_DIR / "sarima_model.pkl"
        fitted.save(str(pkl_path), remove_data=True)
        artifacts.append(str(pkl_path.relative_to(PROJECT_ROOT)))
    except Exception as exc:  # saving must never break the benchmark
        logger.warning("Could not pickle SARIMA results (%s). Parameters JSON was saved.", exc)

    return ModelRun(
        name="SARIMA",
        predictions=predictions,
        train_time_s=train_time,
        latency_ms_per_record=infer_seconds / len(test) * 1000.0,
        artifacts=artifacts,
        hyperparameters={"order": list(order), "seasonal_order": list(seasonal_order), "maxiter": maxiter},
    )


# --------------------------------------------------------------------------- #
# 2) XGBoost
# --------------------------------------------------------------------------- #
def run_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
) -> ModelRun:
    from xgboost import XGBRegressor

    # Chronological validation tail (last 10 % of train) for early stopping.
    n_val = max(1, int(len(X_train) * 0.10))
    X_fit, y_fit = X_train.iloc[:-n_val], y_train.iloc[:-n_val]
    X_val, y_val = X_train.iloc[-n_val:], y_train.iloc[-n_val:]

    params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )
    logger.info("XGBoost: fitting on %d rows (validation %d rows), features=%s",
                len(X_fit), len(X_val), FEATURE_COLUMNS)

    start = time.perf_counter()
    model = XGBRegressor(**params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - start
    best_iter = getattr(model, "best_iteration", None)
    logger.info("XGBoost fitted in %.2f s (best_iteration=%s)", train_time, best_iter)

    model.predict(X_test.iloc[:1])  # warm-up so one-off setup cost is not timed
    raw_pred, infer_seconds = time_call(lambda: model.predict(X_test), repeats=5)
    predictions = pd.Series(np.clip(np.asarray(raw_pred, dtype=float), 0.0, None),
                            index=X_test.index, name="XGBoost")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / "xgboost_model.json"
    model.save_model(str(path))

    return ModelRun(
        name="XGBoost",
        predictions=predictions,
        train_time_s=train_time,
        latency_ms_per_record=infer_seconds / len(X_test) * 1000.0,
        artifacts=[str(path.relative_to(PROJECT_ROOT))],
        hyperparameters={**{k: v for k, v in params.items() if k != "n_jobs"},
                         "best_iteration": None if best_iter is None else int(best_iter)},
    )


# --------------------------------------------------------------------------- #
# 3) LSTM (PyTorch)
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class LSTMForecaster(nn.Module):
        """Stacked LSTM -> last hidden state -> linear head (single-step output)."""

        def __init__(self, input_size: int = 1, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)           # (batch, window, hidden)
            return self.head(out[:, -1, :])  # (batch, 1)


def run_lstm(
    series: pd.Series,
    train_ratio: float,
    window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> ModelRun:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Run: pip install -r requirements.txt")

    data = prepare_lstm_data(series, train_ratio=train_ratio, window=window)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size, num_layers, dropout = 64, 2, 0.2

    # Chronological validation tail (last 10 % of training windows).
    n_val = max(1, int(len(data.X_train) * 0.10))
    X_fit = torch.from_numpy(data.X_train[:-n_val])
    y_fit = torch.from_numpy(data.y_train[:-n_val])
    X_val = torch.from_numpy(data.X_train[-n_val:]).to(device)
    y_val = torch.from_numpy(data.y_train[-n_val:]).to(device)
    X_test = torch.from_numpy(data.X_test)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=batch_size,
                        shuffle=True, generator=generator)

    model = LSTMForecaster(1, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("LSTM: device=%s | train windows=%d | val windows=%d | window=%d | epochs<=%d",
                device, len(X_fit), n_val, window, epochs)

    best_val, best_state, wait, epochs_run = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(X_fit)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        epochs_run = epoch

        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch == 1 or epoch % 5 == 0:
            logger.info("  epoch %3d | train MSE %.5f | val MSE %.5f", epoch, train_loss, val_loss)
        if wait >= patience:
            logger.info("  early stopping at epoch %d (best val MSE %.5f)", epoch, best_val)
            break
    train_time = time.perf_counter() - start
    logger.info("LSTM trained in %.1f s (%d epochs)", train_time, epochs_run)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict() -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for chunk in torch.split(X_test, 512):
                outputs.append(model(chunk.to(device)).cpu())
        return torch.cat(outputs).numpy()

    _predict()  # warm-up
    scaled_pred, infer_seconds = time_call(_predict, repeats=3)
    raw_pred = data.scaler.inverse_transform(scaled_pred).ravel()
    predictions = pd.Series(np.clip(raw_pred, 0.0, None), index=data.test_index, name="LSTM")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = EXPORT_DIR / "lstm_model.pt"
    scaler_path = EXPORT_DIR / "lstm_scaler.joblib"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"input_size": 1, "hidden_size": hidden_size, "num_layers": num_layers,
                   "dropout": dropout, "window": window},
    }, ckpt_path)
    joblib.dump(data.scaler, scaler_path)

    return ModelRun(
        name="LSTM",
        predictions=predictions,
        train_time_s=train_time,
        latency_ms_per_record=infer_seconds / len(X_test) * 1000.0,
        artifacts=[str(ckpt_path.relative_to(PROJECT_ROOT)), str(scaler_path.relative_to(PROJECT_ROOT))],
        hyperparameters={"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
                         "window": window, "batch_size": batch_size, "learning_rate": learning_rate,
                         "max_epochs": epochs, "epochs_run": epochs_run, "patience": patience,
                         "best_val_mse_scaled": float(best_val)},
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_results_table(runs: List[ModelRun], actual: pd.Series) -> pd.DataFrame:
    rows = []
    for run in runs:
        y_pred = run.predictions.reindex(actual.index)
        if y_pred.isna().any():
            raise ValueError(f"{run.name}: predictions do not cover the full test index.")
        rows.append({
            "Model": run.name,
            "RMSE": round(rmse(actual, y_pred), 4),
            "MAE": round(mae(actual, y_pred), 4),
            "Training_Time_s": round(run.train_time_s, 3),
            "Inference_Latency_ms_per_record": round(run.latency_ms_per_record, 5),
            "Test_Records": int(len(actual)),
        })
    return pd.DataFrame(rows)


def pick_plot_window(test_index: pd.DatetimeIndex, hours: int = 72) -> Tuple[int, int]:
    """First Monday 00:00 in the test set with `hours` of data after it (else: test start)."""
    for pos, ts in enumerate(test_index):
        if ts.dayofweek == 0 and ts.hour == 0 and pos + hours <= len(test_index):
            return pos, pos + hours
    return 0, min(hours, len(test_index))


def plot_actual_vs_predicted(pred_df: pd.DataFrame, runs: List[ModelRun],
                             hours: int = 72, path: Path = PLOT_PATH) -> None:
    lo, hi = pick_plot_window(pred_df.index, hours)
    window = pred_df.iloc[lo:hi]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window.index, window["Actual"], color="black", linewidth=2.4, label="Actual")
    for run in runs:
        ax.plot(window.index, window[run.name], color=MODEL_COLORS.get(run.name),
                linewidth=1.6, linestyle="--", label=run.name)
    ax.set_title(f"Actual vs. Predicted Cellular Traffic ({TARGET_COL}) - {hours}-hour test window")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Traffic (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved plot -> %s (%s to %s)", path.relative_to(PROJECT_ROOT),
                window.index[0], window.index[-1])


def library_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for module in ("sklearn", "statsmodels", "xgboost", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "not installed"
    return versions


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SARIMA, XGBoost and LSTM on cellular traffic.")
    p.add_argument("--data-path", default=None, help="Path to hetnet_traffic_hourly_1year.csv")
    p.add_argument("--target", default=TARGET_COL, help="Target column (default: M1)")
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60, help="Max LSTM epochs (early stopping applies)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8, help="LSTM early-stopping patience")
    p.add_argument("--sarima-order", default="1,0,1", help="p,d,q")
    p.add_argument("--sarima-seasonal", default="1,1,1,24", help="P,D,Q,s")
    p.add_argument("--sarima-maxiter", type=int, default=50)
    p.add_argument("--skip-sarima", action="store_true")
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--skip-lstm", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    set_seed(args.seed)
    logger.info("Starting benchmark | target=%s | seed=%d", args.target, args.seed)

    series = load_target_series(args.data_path, args.target)
    train, test = split_series(series, args.train_ratio)
    logger.info("Chronological split: train=%d (%s -> %s) | test=%d (%s -> %s)",
                len(train), train.index[0], train.index[-1], len(test), test.index[0], test.index[-1])

    runs: List[ModelRun] = []
    jobs: List[Tuple[str, Callable[[], ModelRun]]] = []

    if not args.skip_sarima:
        order = parse_int_tuple(args.sarima_order, 3, "--sarima-order")
        seasonal = parse_int_tuple(args.sarima_seasonal, 4, "--sarima-seasonal")
        jobs.append(("SARIMA", lambda: run_sarima(train, test, order, seasonal, args.sarima_maxiter)))
    if not args.skip_xgboost:
        X_tr, y_tr, X_te, y_te = split_xgb_features(series, args.train_ratio)
        if not X_te.index.equals(test.index):
            raise RuntimeError("XGBoost test index differs from the shared test index.")
        jobs.append(("XGBoost", lambda: run_xgboost(X_tr, y_tr, X_te, args.seed)))
    if not args.skip_lstm:
        jobs.append(("LSTM", lambda: run_lstm(series, args.train_ratio, LSTM_WINDOW, args.epochs,
                                              args.batch_size, args.learning_rate, args.patience, args.seed)))

    for name, job in jobs:
        logger.info("=" * 70)
        try:
            runs.append(job())
        except Exception as exc:  # keep going so one failure doesn't lose the others
            logger.error("%s failed: %s", name, exc)
            logger.error("Traceback:\n%s", traceback.format_exc())

    if not runs:
        logger.error("No model finished successfully - nothing to report.")
        return 1

    # ---- results table --------------------------------------------------- #
    results = build_results_table(runs, test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)

    pred_df = pd.DataFrame({"Actual": test})
    for run in runs:
        pred_df[run.name] = run.predictions.reindex(test.index)
    pred_df.index.name = "Timestamp"
    pred_df.to_csv(PREDICTIONS_CSV)

    plot_actual_vs_predicted(pred_df, runs, hours=72)

    METADATA_PATH.write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "n_train": int(get_split_index(len(series), args.train_ratio)),
        "n_test": int(len(test)),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "libraries": library_versions(),
        "models": {r.name: {"hyperparameters": r.hyperparameters, "artifacts": r.artifacts} for r in runs},
    }, indent=2, default=str))

    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS (test set, %d hourly records)\n%s", len(test), results.to_string(index=False))
    logger.info("Saved: %s | %s | %s", RESULTS_CSV.relative_to(PROJECT_ROOT),
                PREDICTIONS_CSV.relative_to(PROJECT_ROOT), METADATA_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
