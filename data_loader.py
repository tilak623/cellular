"""
data_loader.py
==============
Data pipeline and feature engineering for the cellular-traffic forecasting
benchmark (SARIMA vs. XGBoost vs. LSTM).

Responsibilities
----------------
* Load the raw hourly HetNet CSV and parse the ``Timestamp`` column.
* Select the target series (default: macro cell ``M1``).
* Chronological train / test split (default 80 % / 20 %, NO shuffling).
* XGBoost features: lag_1, lag_24, lag_168, hour_of_day, day_of_week.
* LSTM data: MinMax scaling (fitted on the training portion only) and
  24-step sliding windows for single-step prediction.

All three model families are evaluated on the *same* test timestamps
(``series.index[n_train:]``) so the metrics are directly comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "hetnet_traffic_hourly_1year.csv"

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "M1"
TRAIN_RATIO = 0.80
LSTM_WINDOW = 24
FEATURE_COLUMNS = ["lag_1", "lag_24", "lag_168", "hour_of_day", "day_of_week"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def resolve_data_path(path: Optional[str | Path] = None) -> Path:
    """Return an existing CSV path, trying sensible fallbacks."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [
        DEFAULT_DATA_PATH,
        PROJECT_ROOT / "hetnet_traffic_hourly_1year.csv",
        Path.cwd() / "hetnet_traffic_hourly_1year.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find 'hetnet_traffic_hourly_1year.csv'. Tried:\n  " + tried
    )


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw CSV, parse timestamps and sort chronologically."""
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(f"Column '{TIMESTAMP_COL}' not found in {csv_path.name}.")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=TIMESTAMP_COL)
    df = df.set_index(TIMESTAMP_COL)
    logger.info(
        "Loaded %s: %d rows x %d columns (%s -> %s)",
        csv_path.name, len(df), df.shape[1], df.index.min(), df.index.max(),
    )
    return df


def get_target_series(df: pd.DataFrame, target: str = TARGET_COL) -> pd.Series:
    """
    Select the target column and enforce a strict hourly frequency.

    Missing hours (if any) are inserted and linearly interpolated so that
    lag features and SARIMA see a gap-free series.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)[:10]}...")

    series = df[target].astype("float64")
    series = series.asfreq("h")  # inserts NaN for any missing hour

    n_missing = int(series.isna().sum())
    if n_missing:
        logger.warning("%d missing values in '%s' -> linear interpolation.", n_missing, target)
        series = series.interpolate(method="linear", limit_direction="both")

    series.name = target
    return series


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def get_split_index(n_samples: int, train_ratio: float = TRAIN_RATIO) -> int:
    """Index of the first test sample for a chronological split."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    return int(n_samples * train_ratio)


def split_series(series: pd.Series, train_ratio: float = TRAIN_RATIO) -> Tuple[pd.Series, pd.Series]:
    """Chronological split of a series into (train, test)."""
    n_train = get_split_index(len(series), train_ratio)
    return series.iloc[:n_train], series.iloc[n_train:]


# --------------------------------------------------------------------------- #
# XGBoost features
# --------------------------------------------------------------------------- #
def build_lag_features(series: pd.Series) -> pd.DataFrame:
    """
    Build the supervised-learning frame for XGBoost.

    Columns: ``y`` (target), lag_1, lag_24, lag_168, hour_of_day, day_of_week.
    Every lag looks strictly into the past, so computing the lags on the full
    series and splitting afterwards does not leak test information into training.
    The first 168 rows are dropped because lag_168 is undefined for them.
    """
    feats = pd.DataFrame({"y": series})
    feats["lag_1"] = series.shift(1)
    feats["lag_24"] = series.shift(24)
    feats["lag_168"] = series.shift(168)
    feats["hour_of_day"] = series.index.hour
    feats["day_of_week"] = series.index.dayofweek
    return feats.dropna()


def split_xgb_features(
    series: pd.Series, train_ratio: float = TRAIN_RATIO
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Return ``X_train, y_train, X_test, y_test`` for XGBoost.

    The split timestamp is taken from the *original* series, which guarantees
    that the test set is identical to the one used by SARIMA and the LSTM.
    """
    n_train = get_split_index(len(series), train_ratio)
    split_ts = series.index[n_train]

    feats = build_lag_features(series)
    train = feats.loc[feats.index < split_ts]
    test = feats.loc[feats.index >= split_ts]

    return (
        train[FEATURE_COLUMNS], train["y"],
        test[FEATURE_COLUMNS], test["y"],
    )


# --------------------------------------------------------------------------- #
# LSTM sequences
# --------------------------------------------------------------------------- #
@dataclass
class LSTMData:
    """Container for scaled LSTM windows and the fitted scaler."""
    X_train: np.ndarray          # (n_train_windows, window, 1)
    y_train: np.ndarray          # (n_train_windows, 1)
    X_test: np.ndarray           # (n_test_windows, window, 1)
    y_test: np.ndarray           # (n_test_windows, 1)  (scaled)
    scaler: MinMaxScaler
    test_index: pd.DatetimeIndex
    window: int


def make_windows(scaled: np.ndarray, start: int, stop: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Windows predicting positions ``start .. stop-1``.

    For target position ``i`` the input is ``scaled[i-window : i]`` (24 past
    hours) and the label is ``scaled[i]``  -> multi-step input, single-step output.
    """
    if start < window:
        raise ValueError("start must be >= window so that every window has full history.")
    idx = np.arange(start, stop)
    X = np.stack([scaled[i - window:i] for i in idx]).astype("float32")  # (N, window, 1)
    y = scaled[idx].astype("float32")                                     # (N, 1)
    return X, y


def prepare_lstm_data(
    series: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    window: int = LSTM_WINDOW,
) -> LSTMData:
    """
    Scale with MinMaxScaler (fitted on the training portion ONLY) and build
    24-step sliding windows.

    Test windows may reach back into the last training hours for context
    (this is legitimate: those are past observations), but test values are
    never used for fitting the scaler or the model.
    """
    n = len(series)
    n_train = get_split_index(n, train_ratio)
    values = series.to_numpy(dtype="float64").reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(values[:n_train])
    scaled = scaler.transform(values)

    X_train, y_train = make_windows(scaled, start=window, stop=n_train, window=window)
    X_test, y_test = make_windows(scaled, start=n_train, stop=n, window=window)

    return LSTMData(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        scaler=scaler,
        test_index=series.index[n_train:],
        window=window,
    )


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #
def load_target_series(path: Optional[str | Path] = None, target: str = TARGET_COL) -> pd.Series:
    """Load CSV -> hourly target series in one call."""
    return get_target_series(load_raw_data(path), target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    s = load_target_series()
    tr, te = split_series(s)
    Xtr, ytr, Xte, yte = split_xgb_features(s)
    lstm = prepare_lstm_data(s)
    print(f"Series: {len(s)} | train: {len(tr)} | test: {len(te)}")
    print(f"XGB   : X_train {Xtr.shape} | X_test {Xte.shape}")
    print(f"LSTM  : X_train {lstm.X_train.shape} | X_test {lstm.X_test.shape}")
    assert Xte.index.equals(te.index) and lstm.test_index.equals(te.index)
    print("Test index identical across SARIMA / XGBoost / LSTM: OK")
