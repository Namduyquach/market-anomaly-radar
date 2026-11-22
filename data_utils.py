"""
data_utils.py
Data downloading, feature engineering, non-leaking scaling, and sliding window creation.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def download_financial_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Download historical financial data from yfinance.

    Handles potential MultiIndex headers and ensures required columns exist.
    """
    df = yf.download(
        ticker.strip().upper(),
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if df.empty or len(df) < 30:
        raise ValueError(
            f"Không tìm thấy đủ dữ liệu cho mã '{ticker}' trong khoảng thời gian đã chọn. "
            f"Số phiên tải được: {len(df)}. Vui lòng mở rộng khoảng thời gian hoặc kiểm tra lại mã Ticker."
        )

    # Flatten MultiIndex columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        # Case: MultiIndex with column names at level 0 and ticker at level 1
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_cols.append(col[0])
            else:
                new_cols.append(col)
        df.columns = new_cols

    # Ensure required columns exist
    if "Close" not in df.columns:
        raise ValueError("Dữ liệu không có cột 'Close'.")

    # If Volume is missing or all zero, fill with small positive value to allow vol_change computation
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = 1.0

    # Ensure index is datetime and drop duplicates
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial indicators: log_return and vol_change.

    log_return = ln(Close_t / Close_{t-1}) vol_change = ln((Volume_t + 1) /
    (Volume_{t-1} + 1))
    """
    processed = df.copy()

    # Log returns
    processed["log_return"] = np.log(
        processed["Close"] / processed["Close"].shift(1)
    )

    # Volume change (log ratio with +1 smoothing to avoid log(0))
    vol_curr = processed["Volume"].clip(lower=0) + 1.0
    vol_prev = vol_curr.shift(1)
    processed["vol_change"] = np.log(vol_curr / vol_prev)

    # Drop the first NaN row resulting from shift(1)
    processed = processed.dropna(
        subset=["log_return", "vol_change", "Close"]
    ).copy()

    return processed


def create_sliding_windows(
    data_2d: np.ndarray, window_size: int
) -> np.ndarray:
    """Generate 3D sliding window array of shape (samples, window_size,

    features).
    """
    num_samples = len(data_2d) - window_size + 1
    if num_samples <= 0:
        raise ValueError(
            f"Độ dài dữ liệu ({len(data_2d)}) nhỏ hơn kích thước cửa sổ window_size ({window_size})."
        )

    windows = np.empty(
        (num_samples, window_size, data_2d.shape[1]), dtype=np.float32
    )
    for i in range(num_samples):
        windows[i] = data_2d[i : i + window_size]

    return windows


def prepare_datasets(
    df: pd.DataFrame, window_size: int = 20, train_ratio: float = 0.7
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, pd.DataFrame, StandardScaler]:
    """Split data chronologically, fit StandardScaler ONLY on training set to

    prevent data leakage, and construct 3D sliding windows for both train and
    test.
    """
    feature_cols = ["log_return", "vol_change"]
    n_total = len(df)
    split_idx = int(n_total * train_ratio)

    if split_idx <= window_size + 5:
        raise ValueError(
            f"Tập huấn luyện quá ngắn ({split_idx} phiên) so với window_size ({window_size}). "
            f"Vui lòng tăng khoảng thời gian dữ liệu."
        )

    # Train split
    train_df = df.iloc[:split_idx].copy()

    # Fit scaler strictly on training set only (No Data Leakage!)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(
        train_df[feature_cols].values.astype(np.float32)
    )

    # Create train sliding windows
    X_train = create_sliding_windows(train_features_scaled, window_size)
    train_dates = train_df.index[window_size - 1 :]

    # Test split:
    # We include `window_size - 1` days from train set before split_idx so that the
    # very first test window ends exactly on split_idx (the first day of the test set).
    test_start_idx = max(0, split_idx - (window_size - 1))
    test_features_raw = df.iloc[test_start_idx:][feature_cols].values.astype(
        np.float32
    )

    # Transform test set using the train-fitted scaler
    test_features_scaled = scaler.transform(test_features_raw)

    # Create test sliding windows
    X_test = create_sliding_windows(test_features_scaled, window_size)

    # Align dates for test windows (each window i corresponds to date at index i + window_size - 1)
    test_aligned_df = df.iloc[split_idx:].copy()
    test_dates = test_aligned_df.index

    # Check alignment
    if len(X_test) != len(test_aligned_df):
        # Truncate to matching length if there's any boundary mismatch
        min_len = min(len(X_test), len(test_aligned_df))
        X_test = X_test[:min_len]
        test_aligned_df = test_aligned_df.iloc[:min_len]
        test_dates = test_aligned_df.index

    # Compute Z-score anomaly baseline on log_return
    # Baseline mean and std are derived strictly from train_df
    train_mean = train_df["log_return"].mean()
    train_std = train_df["log_return"].std()

    test_aligned_df["z_score"] = (
        test_aligned_df["log_return"] - train_mean
    ) / (train_std + 1e-8)
    test_aligned_df["z_anomaly"] = test_aligned_df["z_score"].abs() > 2.0

    return X_train, train_dates, X_test, test_aligned_df, scaler
