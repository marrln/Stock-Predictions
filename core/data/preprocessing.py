"""Data preprocessing and feature engineering functions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import TimeSeriesDataset

# Default paths
DEFAULT_SENTIMENT_CSV = "data_stats/daily_sentiment.csv"
DEFAULT_PRICE_DIR = "Stock_price/full_history"


def load_daily_sentiment(path: str = DEFAULT_SENTIMENT_CSV) -> pd.DataFrame:
    """Load daily sentiment data from CSV."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={
        "Daily_Sentiment": "daily_sentiment",
        "n_articles": "n_articles"
    })
    return df


def load_price_for_ticker(ticker: str, price_dir: str = DEFAULT_PRICE_DIR) -> pd.DataFrame:
    """Load price data for a single ticker."""
    p = os.path.join(price_dir, f"{ticker}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    
    df = pd.read_csv(p, parse_dates=["date"])
    return df.rename(columns={"date": "Date"})


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) over a rolling window."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Use simple moving average for initial implementation
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Avoid division by zero
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)  # neutral when undefined
    rsi = rsi.replace([np.inf, -np.inf], 50.0)
    return rsi


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Return upper and lower Bollinger Bands for the series."""
    mid = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, lower


def make_features_for_ticker(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
    sentiment_fill: str = "ffill"
) -> pd.DataFrame:
    """Merge price and sentiment and compute features for a ticker."""
    if sentiment_fill not in ("ffill", "zero"):
        raise ValueError("sentiment_fill must be 'ffill' or 'zero'")
    
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Filter sentiment for this ticker
    sentiment_subset = sentiment_df[sentiment_df["Ticker"] == ticker].copy()
    if not sentiment_subset.empty:
        sentiment_subset["Date"] = pd.to_datetime(sentiment_subset["Date"]).dt.normalize()
    
    # Basic price features
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change()
    df["log_volume"] = np.log1p(df["volume"].astype(float))
    
    # Intraday range (safely handle division by zero)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    df["intraday_range"] = np.where(close > 0, (high - low) / close, 0.0)
    
    # Technical indicators (shifted to avoid lookahead)
    df["volatility_5"] = df["return"].rolling(window=5, min_periods=1).std().shift(1).fillna(0.0)
    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean().shift(1)
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean().shift(1)

    # Additional indicators: RSI, Bollinger Bands, Rate of Change (shifted)
    df["rsi_14"] = calculate_rsi(df["close"], 14).shift(1).fillna(50.0)
    bb_upper, bb_lower = calculate_bollinger_bands(df["close"], window=20, num_std=2)
    df["bb_upper"] = bb_upper.shift(1)
    df["bb_lower"] = bb_lower.shift(1)
    df["roc_10"] = df["close"].pct_change(10).shift(1).fillna(0.0)

    # Momentum and volume spike indicators (shifted to avoid lookahead)
    df["momentum_5"] = (df["close"] / df["close"].shift(5) - 1).shift(1).fillna(0.0)
    df["momentum_10"] = (df["close"] / df["close"].shift(10) - 1).shift(1).fillna(0.0)
    vol_roll = df["volume"].rolling(window=20, min_periods=1).mean().shift(1)
    df["volume_spike"] = (df["volume"] > vol_roll * 1.5).astype(float).fillna(0.0)
    
    # Merge sentiment
    merged = pd.merge(
        df, 
        sentiment_subset[["Date", "daily_sentiment", "n_articles"]], 
        on="Date", 
        how="left"
    )
    
    # Fill missing sentiment
    if sentiment_fill == "ffill":
        merged["daily_sentiment"] = merged["daily_sentiment"].ffill().fillna(0.0)
    else:
        merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    
    # Fill n_articles
    merged["n_articles"] = merged["n_articles"].fillna(0).astype(int)
    
    # Add ticker identifier
    merged["Ticker"] = ticker
    
    return merged


def create_sequences_from_ticker(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
    target_type: str = "return"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct sequences from a single ticker DataFrame."""
    df = df.copy().reset_index(drop=True)
    
    # Compute target
    if target_type == "return":
        df["target"] = df["close"].shift(-1) / df["close"] - 1
    elif target_type == "close":
        df["target"] = df["close"].shift(-1)
    else:
        raise ValueError("target_type must be 'return' or 'close'")
    
    # Drop rows without target
    df = df[df["target"].notna()].reset_index(drop=True)
    
    X_list, y_list, meta_rows = [], [], []
    N = len(df)
    
    for end in range(seq_len - 1, N):
        start = end - (seq_len - 1)
        X_window = df.loc[start:end, feature_cols].values
        
        # Skip windows with NaN values
        if np.isnan(X_window).any():
            continue
        
        target_val = df.loc[end, "target"]
        if np.isnan(target_val):
            continue
        
        X_list.append(X_window)
        y_list.append(target_val)
        
        # Store metadata
        meta_rows.append({
            "Date": df.loc[end, "Date"],
            "Ticker": df.loc[end, "Ticker"],
            "last_return": float(df.loc[end, "return"]) if "return" in df.columns else np.nan,
            "last_close": float(df.loc[end, "close"]) if "close" in df.columns else np.nan,
        })
    
    if not X_list:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=float), \
               np.zeros((0,), dtype=float), \
               pd.DataFrame()
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_rows)
    
    return X, y, meta


def get_ticker_stats(
    tickers: Optional[List[str]] = None,
    price_dir: str = DEFAULT_PRICE_DIR,
    sentiment_csv: str = DEFAULT_SENTIMENT_CSV,
    seq_len: int = 16,
) -> pd.DataFrame:
    """Return per-ticker statistics for selection."""
    if tickers is None:
        tickers = [f.stem for f in Path(price_dir).glob("*.csv")]
    
    sentiment_df = pd.read_csv(sentiment_csv, parse_dates=["Date"]) if os.path.exists(sentiment_csv) else pd.DataFrame()
    
    rows = []
    for ticker in sorted(tickers):
        price_path = Path(price_dir) / f"{ticker}.csv"
        if not price_path.exists():
            continue
        
        price_df = pd.read_csv(price_path, parse_dates=["date"]).rename(columns={"date": "Date"})
        
        stats = {
            "Ticker": ticker,
            "num_days": len(price_df),
            "avg_volume": float(price_df["volume"].mean()) if "volume" in price_df.columns else 0.0,
            "possible_sequences": max(0, len(price_df) - seq_len + 1),
        }
        
        if not sentiment_df.empty:
            ticker_sentiment = sentiment_df[sentiment_df["Ticker"] == ticker]
            stats["n_sentiment_days"] = int(ticker_sentiment["Date"].nunique())
            stats["total_articles"] = int(ticker_sentiment["n_articles"].sum())
        else:
            stats["n_sentiment_days"] = 0
            stats["total_articles"] = 0
        
        rows.append(stats)
    
    return pd.DataFrame(rows).set_index("Ticker")


def build_dataset_all_tickers(
    tickers: Optional[List[str]] = None,
    seq_len: int = 10,
    price_dir: str = DEFAULT_PRICE_DIR,
    sentiment_csv: str = DEFAULT_SENTIMENT_CSV,
    feature_cols: Optional[List[str]] = None,
    target_type: str = "return",
    sentiment_fill: str = "ffill",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], pd.DataFrame]:
    """Build X/y arrays for all specified tickers."""
    # Validate inputs
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    if target_type not in ("return", "close"):
        raise ValueError("target_type must be 'return' or 'close'")
    if not os.path.exists(price_dir):
        raise FileNotFoundError(f"Price directory not found: {price_dir}")
    
    # Load sentiment data
    sentiment_df = load_daily_sentiment(sentiment_csv) if os.path.exists(sentiment_csv) else pd.DataFrame()
    if not sentiment_df.empty:
        sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.normalize()
    
    # Default features
    if feature_cols is None:
        feature_cols = [
            "return", "log_volume", "intraday_range", 
            "volatility_5", "ma_5", "ma_20", 
            # Additional indicators
            "rsi_14", "bb_upper", "bb_lower", "roc_10",
            # Momentum and volume features
            "momentum_5", "momentum_10", "volume_spike",
            "daily_sentiment", "n_articles"
        ]
    
    # If tickers not specified, use all from sentiment
    if tickers is None and not sentiment_df.empty:
        tickers = sorted(sentiment_df["Ticker"].unique())
    elif tickers is None:
        raise ValueError("Must specify tickers when no sentiment data available")
    
    X_chunks, y_chunks, metas = [], [], []
    missing_tickers: List[str] = []
    processed_tickers: List[str] = []

    for ticker in tickers:
        try:
            price_df = load_price_for_ticker(ticker, price_dir)
            feat_df = make_features_for_ticker(price_df, sentiment_df, ticker, sentiment_fill)
            X_t, y_t, meta_t = create_sequences_from_ticker(feat_df, seq_len, feature_cols, target_type)
            
            if X_t.shape[0] > 0:
                X_chunks.append(X_t)
                y_chunks.append(y_t)
                metas.append(meta_t)
                processed_tickers.append(ticker)
        except FileNotFoundError:
            missing_tickers.append(ticker)
            continue

    if missing_tickers:
        print(f"Warning: Missing price files for tickers: {', '.join(missing_tickers)}")
    if processed_tickers:
        print(f"Processed tickers: {', '.join(processed_tickers)}")
    else:
        print("Warning: No tickers produced sequences with the provided settings.")
    
    if not X_chunks:
        return {}, {}, pd.DataFrame()
    
    X_all = np.concatenate(X_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)
    
    info = {
        "feature_cols": feature_cols,
        "target_type": target_type,
        "seq_len": seq_len,
    }
    
    return {"X": X_all, "y": y_all}, info, meta_all


def split_time_based_rolling(
    meta: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_days: int = 750,
    val_days: int = 125,
    test_days: Optional[int] = 125,
    seq_len: int = 30,
    step_days: int = 125,
) -> List[
    Tuple[
        TimeSeriesDataset,
        TimeSeriesDataset,
        Optional[TimeSeriesDataset],
    ]
]:
    """
    Rolling (walk-forward) splits with purging and optional test window.

    Fold layout:
        |-- train --|-- embargo --|-- val --|-- embargo --|-- test --|

    If test_days is None, no test set is created.

    Returns:
        List of (train_ds, val_ds, test_ds_or_None)
    """

    if len(meta) != len(X) or len(X) != len(y):
        raise ValueError("meta, X, and y must have the same length")

    meta = meta.copy()
    meta["Date"] = pd.to_datetime(meta["Date"])  # ensure datetime

    # Sort chronologically (CRITICAL)
    order = np.argsort(meta["Date"].values)

    X = X[order]
    y = y[order]
    meta = meta.iloc[order].reset_index(drop=True)

    # Use unique trading dates as the unit for train/val/test sizes
    unique_dates = meta["Date"].dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    D = len(unique_dates)

    folds = []
    fold_id = 0
    date_start_idx = 0

    while True:
        # Compute date-based indices (inclusive)
        train_end_idx = date_start_idx + train_days - 1
        val_start_idx = train_end_idx + seq_len
        val_end_idx = val_start_idx + val_days - 1

        if test_days is not None:
            test_start_idx = val_end_idx + seq_len
            test_end_idx = test_start_idx + test_days - 1
        else:
            test_start_idx = test_end_idx = None

        # Stop if validation window goes beyond available dates
        if val_end_idx >= D:
            break
        if test_days is not None and test_end_idx >= D:
            break

        # Map date ranges back to sample indices (all samples whose Date falls in range)
        train_dates = unique_dates[date_start_idx : train_end_idx + 1]
        val_dates = unique_dates[val_start_idx : val_end_idx + 1]
        test_dates = (
            unique_dates[test_start_idx : test_end_idx + 1]
            if test_days is not None
            else pd.Index([])
        )

        train_mask = meta["Date"].dt.normalize().isin(train_dates)
        val_mask = meta["Date"].dt.normalize().isin(val_dates)
        test_mask = meta["Date"].dt.normalize().isin(test_dates) if test_days is not None else None

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            # If a window yields no samples (unlikely), advance and continue
            date_start_idx += step_days
            continue

        train_ds = TimeSeriesDataset(
            X[train_mask.values],
            y[train_mask.values],
            meta=meta.loc[train_mask.values].reset_index(drop=True),
        )

        val_ds = TimeSeriesDataset(
            X[val_mask.values],
            y[val_mask.values],
            meta=meta.loc[val_mask.values].reset_index(drop=True),
        )

        if test_days is not None:
            test_ds = TimeSeriesDataset(
                X[test_mask.values],
                y[test_mask.values],
                meta=meta.loc[test_mask.values].reset_index(drop=True),
            )
        else:
            test_ds = None

        folds.append((train_ds, val_ds, test_ds))

        msg = (
            f"Fold {fold_id}: "
            f"Train [{train_ds.meta['Date'].min().date()} -> {train_ds.meta['Date'].max().date()}], "
            f"Val [{val_ds.meta['Date'].min().date()} -> {val_ds.meta['Date'].max().date()}]"
        )

        if test_ds is not None:
            msg += (
                f", Test [{test_ds.meta['Date'].min().date()} -> "
                f"{test_ds.meta['Date'].max().date()}]"
            )

        print(msg)

        fold_id += 1
        date_start_idx += step_days

    if not folds:
        raise ValueError("No rolling folds could be created with the given parameters")

    return folds


def compute_scalers_from_train(
    train_ds: TimeSeriesDataset, 
    per_ticker: bool = False
) -> StandardScaler | Dict[str, StandardScaler]:
    """Fit scaler(s) on train set's X."""
    X = train_ds.X  # (N, seq_len, feat)
    N, S, F = X.shape
    
    if not per_ticker:
        flat = X.reshape(-1, F)
        return StandardScaler().fit(flat)
    
    # Per-ticker scaling
    if train_ds.meta is None:
        raise ValueError("per_ticker=True requires train_ds to have .meta with 'Ticker' column")
    
    scalers = {}
    tickers = train_ds.meta['Ticker'].unique()
    
    for ticker in tickers:
        indices = train_ds.meta.index[train_ds.meta['Ticker'] == ticker].tolist()
        if not indices:
            continue
        
        X_t = X[indices]  # (n_samples_t, seq_len, feat)
        flat_t = X_t.reshape(-1, F)
        
        if flat_t.shape[0] > 0:
            scalers[ticker] = StandardScaler().fit(flat_t)
    
    return scalers


def apply_scaler_to_dataset(
    ds: TimeSeriesDataset, 
    scaler: StandardScaler | Dict[str, StandardScaler]
) -> TimeSeriesDataset:
    """Apply scaler(s) to dataset features."""
    X = ds.X
    N, S, F = X.shape
    
    # Global scaler
    if not isinstance(scaler, dict):
        X_flat = X.reshape(-1, F)
        X_scaled_flat = scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, S, F)
        return TimeSeriesDataset(X_scaled, ds.y, meta=ds.meta)
    
    # Per-ticker scalers
    if ds.meta is None:
        raise ValueError("Per-ticker scalers require dataset to have .meta with 'Ticker' column")
    
    X_scaled = np.empty_like(X)
    ticker_array = ds.meta['Ticker'].values
    
    for ticker in scaler.keys():
        if ticker not in scaler:
            raise ValueError(f"No scaler found for ticker {ticker}")
        
        indices = np.where(ticker_array == ticker)[0]
        if indices.size == 0:
            continue
        
        X_sub = X[indices]
        flat_sub = X_sub.reshape(-1, F)
        scaled_flat = scaler[ticker].transform(flat_sub)
        X_sub_scaled = scaled_flat.reshape(len(indices), S, F)
        X_scaled[indices] = X_sub_scaled
    
    return TimeSeriesDataset(X_scaled, ds.y, meta=ds.meta)