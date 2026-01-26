"""Data preprocessing and feature engineering functions."""
from __future__ import annotations

import os
import pickle
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
    
    # Technical indicators
    df["volatility_5"] = df["return"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    
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
            "daily_sentiment", "n_articles"
        ]
    
    # If tickers not specified, use all from sentiment
    if tickers is None and not sentiment_df.empty:
        tickers = sorted(sentiment_df["Ticker"].unique())
    elif tickers is None:
        raise ValueError("Must specify tickers when no sentiment data available")
    
    X_chunks, y_chunks, metas = [], [], []
    
    for ticker in tickers:
        try:
            price_df = load_price_for_ticker(ticker, price_dir)
            feat_df = make_features_for_ticker(price_df, sentiment_df, ticker, sentiment_fill)
            X_t, y_t, meta_t = create_sequences_from_ticker(feat_df, seq_len, feature_cols, target_type)
            
            if X_t.shape[0] > 0:
                X_chunks.append(X_t)
                y_chunks.append(y_t)
                metas.append(meta_t)
        except FileNotFoundError:
            continue
    
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


def split_time_based(
    meta: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_val_years: Tuple[int, int] = (2018, 2022),
    test_year: int = 2023,
    val_frequency_months: int = 4,
    val_duration_months: int = 1,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Split data into train/val/test temporally."""
    if val_frequency_months <= 0:
        raise ValueError("val_frequency_months must be > 0")
    if val_duration_months >= val_frequency_months:
        raise ValueError("val_duration_months must be less than val_frequency_months")
    
    meta = meta.copy()
    meta["Date"] = pd.to_datetime(meta["Date"])
    years = meta["Date"].dt.year
    
    # Test mask
    test_mask = years == test_year
    
    # Train+val mask
    tv_mask = (years >= train_val_years[0]) & (years <= train_val_years[1])
    
    # Build validation months
    start = pd.Timestamp(year=train_val_years[0], month=1, day=1)
    months_since_start = (meta.loc[tv_mask, "Date"].dt.year - start.year) * 12 + \
                         (meta.loc[tv_mask, "Date"].dt.month - start.month)
    
    val_idx_mask = pd.Series(False, index=meta.index)
    for idx in months_since_start.index:
        ms = months_since_start.loc[idx]
        if (ms % val_frequency_months) < val_duration_months:
            val_idx_mask.loc[idx] = True
    
    train_mask = tv_mask & (~val_idx_mask)
    val_mask = tv_mask & val_idx_mask
    
    # Create datasets
    train_ds = TimeSeriesDataset(
        X[train_mask.values], 
        y[train_mask.values], 
        meta=meta.loc[train_mask.values].reset_index(drop=True)
    )
    
    val_ds = TimeSeriesDataset(
        X[val_mask.values], 
        y[val_mask.values], 
        meta=meta.loc[val_mask.values].reset_index(drop=True)
    )
    
    test_ds = TimeSeriesDataset(
        X[test_mask.values], 
        y[test_mask.values], 
        meta=meta.loc[test_mask.values].reset_index(drop=True)
    )
    
    return train_ds, val_ds, test_ds


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