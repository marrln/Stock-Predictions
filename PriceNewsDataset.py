"""PriceNewsDataset

Provides utilities to build time-series datasets that combine daily stock prices
and daily sentiment (from `data_stats/daily_sentiment.csv`) to feed LSTM models.

Features created (per-ticker):
- return: daily percentage change in close price
- log_volume: log(1 + volume)
- intraday_range: (high - low) / close
- volatility_5: 5-day rolling std of returns
- ma_5: 5-day moving average of close
- ma_20: 20-day moving average of close
- daily_sentiment: sentiment score from news articles
- n_articles: number of articles published that day

Splits:
- Train/Val: years 2018-2022
- Test: year 2023

Validation strategy:
- periodic: reserve `val_duration_months` every `val_frequency_months` months
(e.g., 1 month every 4 months). You can change `val_frequency_months` to 6
for sparser validation.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from pathlib import Path


def _drop_meta_collate(batch):
    """Collate function that drops any meta and returns (X_batch, y_batch).

    Accepts samples that are tuples like `(X, y)` or `(X, y, meta)` and
    returns two tensors: stacked X and stacked y.
    """
    X_list = []
    y_list = []
    for sample in batch:
        if isinstance(sample, dict):
            # rarely used, but support dict samples: prefer 'input'/'x' and 'target'/'y'
            x = sample.get("input", sample.get("x", None))
            y = sample.get("target", sample.get("y", None))
            if x is None or y is None:
                raise ValueError("Batch dict must contain 'input'/'x' and 'target'/'y' keys")
        else:
            if len(sample) < 2:
                raise ValueError("Dataset sample must have at least (X, y)")
            x, y = sample[0], sample[1]
        X_list.append(x)
        y_list.append(y)

    # Use default_collate to turn lists of arrays/tensors into batched tensors
    return default_collate(X_list), default_collate(y_list)


DEFAULT_SENTIMENT_CSV = "data_stats/daily_sentiment.csv"
DEFAULT_PRICE_DIR = "Stock_price/full_history"


class TimeSeriesDataset(Dataset):
    """Simple Dataset wrapping precomputed sequences (X) and targets (y).

    Optionally carries a `meta` DataFrame with per-sample metadata (e.g., Date, Ticker).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: Optional[pd.DataFrame] = None):
        assert len(X) == len(y)
        if meta is not None:
            assert len(meta) == len(X), "meta must have same length as X"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.meta = meta.reset_index(drop=True) if meta is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.meta is None:
            return self.X[idx], self.y[idx]
        # return meta as plain dict to make collating in DataLoader robust
        md = self.meta.loc[idx].to_dict()
        # Convert Timestamp to ISO string for collate compatibility
        if "Date" in md and not pd.isna(md["Date"]):
            md["Date"] = pd.to_datetime(md["Date"]).isoformat()
        return self.X[idx], self.y[idx], md


def load_daily_sentiment(path: str = DEFAULT_SENTIMENT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])  # columns: Ticker, Date, Daily_Sentiment, n_articles
    df = df.rename(columns={"Daily_Sentiment": "daily_sentiment", "n_articles": "n_articles"})
    return df


def load_price_for_ticker(ticker: str, price_dir: str = DEFAULT_PRICE_DIR) -> pd.DataFrame:
    p = os.path.join(price_dir, f"{ticker}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["date"])  # columns: date,volume,open,high,low,close,adj close
    df = df.rename(columns={"date": "Date"})
    return df


def make_features_for_ticker(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, ticker: str, sentiment_fill: str = "ffill") -> pd.DataFrame:
    """Merge price and sentiment and compute features per trading day for a given ticker.

    Output columns: Ticker, Date, close, return, log_volume, intraday_range, 
    volatility_5, ma_5, ma_20, daily_sentiment, n_articles

    sentiment_fill: how to fill missing sentiment values:
      - 'ffill' (default): forward-fill last known sentiment, initial missing -> 0.0
      - 'zero': fill missing with 0.0
    """
    # Validate sentiment_fill early
    if sentiment_fill not in ("ffill", "zero"):
        raise ValueError("sentiment_fill must be 'ffill' or 'zero'")
    df = price_df.copy()
    # normalize and sort dates to ensure consistent merging
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure sentiment dates are normalized when filtered/merged
    s = sentiment_df[sentiment_df["Ticker"] == ticker].copy()
    if not s.empty:
        s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()

    df["Ticker"] = ticker
    # compute pct return (current-day return vs previous close)
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change()
    # We will use next day return as target when creating sequences
    df["log_volume"] = np.log1p(df["volume"].astype(float))
    # handle division by zero safely for intraday range
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    df["intraday_range"] = np.where(close > 0, (high - low) / close, 0.0)

    # compute volatility and moving averages
    df["volatility_5"] = df["return"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # merge sentiment
    merged = pd.merge(df, s[["Date", "daily_sentiment", "n_articles"]], on="Date", how="left")

    # Fill missing sentiment according to policy
    if sentiment_fill == "ffill":
        merged["daily_sentiment"] = merged["daily_sentiment"].ffill().fillna(0.0)
    else:  # zero
        merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)

    # For n_articles, keep 0 when there were no articles that day
    merged["n_articles"] = merged["n_articles"].fillna(0).astype(int)

    # Keep relevant columns (add new features)
    cols = ["Ticker", "Date", "close", "return", "log_volume", "intraday_range", "volatility_5", "ma_5", "ma_20", "daily_sentiment", "n_articles"]
    return merged[cols]


def create_sequences_from_ticker(df: pd.DataFrame, seq_len: int, feature_cols: List[str], target_type: str = "return") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct sequences from a single ticker DataFrame. Sequences are contiguous in terms of index (trading days).

    df must be sorted by Date. The function will compute the target depending on `target_type`:
      - 'return': next-day percentage change of close
      - 'close' : next-day close price
    Returns X (N, seq_len, features), y (N,) and meta (DataFrame with Date of target day and Ticker)
    """
    df = df.copy().reset_index(drop=True)

    # compute target column based on target_type
    if target_type == "return":
        df["target_return"] = df["close"].shift(-1) / df["close"] - 1
        target_col = "target_return"
    elif target_type == "close":
        df["target_close"] = df["close"].shift(-1)
        target_col = "target_close"
    else:
        raise ValueError("target_type must be 'return' or 'close'")

    # drop rows without target
    valid_idx = df[target_col].notna()
    df = df[valid_idx].reset_index(drop=True)

    X_list = []
    y_list = []
    meta_rows = []

    N = len(df)
    for end in range(seq_len - 1, N):
        start = end - (seq_len - 1)
        X_window = df.loc[start : end, feature_cols].values
        # ensure there are no NaNs in X_window
        if np.isnan(X_window).any():
            continue
        target_val = df.loc[end, target_col]
        if np.isnan(target_val):
            continue
        X_list.append(X_window)
        y_list.append(target_val)
        meta_rows.append(df.loc[end, ["Date", "Ticker"]])

    if len(X_list) == 0:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=float), np.zeros((0,), dtype=float), pd.DataFrame()

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    meta = pd.concat(meta_rows, axis=1).T.reset_index(drop=True)
    return X, y, meta


def get_ticker_stats(
    tickers: Optional[List[str]] = None,
    price_dir: str = DEFAULT_PRICE_DIR,
    sentiment_csv: str = DEFAULT_SENTIMENT_CSV,
    seq_len: int = 16,
) -> pd.DataFrame:
    """Return per-ticker stats useful for selection.

    Columns: num_days, avg_volume, possible_sequences, n_sentiment_days, total_articles
    """
    if tickers is None:
        tickers = [fname[:-4] for fname in os.listdir(price_dir) if fname.endswith(".csv")]

    sent = pd.read_csv(sentiment_csv, parse_dates=["Date"]) if os.path.exists(sentiment_csv) else pd.DataFrame()

    rows = []
    for t in sorted(tickers):
        p = os.path.join(price_dir, f"{t}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["date"]).rename(columns={"date": "Date"})
        num_days = len(df)
        avg_vol = float(df["volume"].mean()) if "volume" in df.columns else 0.0
        possible_sequences = max(0, num_days - seq_len + 1)
        if not sent.empty:
            n_sent = int(sent[sent["Ticker"] == t]["Date"].nunique())
            total_articles = int(sent[sent["Ticker"] == t]["n_articles"].sum())
        else:
            n_sent = 0
            total_articles = 0
        rows.append({"Ticker": t, "num_days": num_days, "avg_volume": avg_vol, "possible_sequences": possible_sequences, "n_sentiment_days": n_sent, "total_articles": total_articles})

    df_stats = pd.DataFrame(rows).set_index("Ticker")
    return df_stats


def build_dataset_all_tickers(
    tickers: Optional[List[str]] = None,
    seq_len: int = 10,
    price_dir: str = DEFAULT_PRICE_DIR,
    sentiment_csv: str = DEFAULT_SENTIMENT_CSV,
    feature_cols: Optional[List[str]] = None,
    target_type: str = "return",
    sentiment_fill: str = "ffill",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    """Build X/y arrays for all tickers present (or restricted to `tickers`).

    Parameters:
      - target_type: 'return' (next-day pct change) or 'close' (next-day close price)

    Returns dictionary with keys: X_all (list of arrays concatenated), y_all, and meta dataframe.
    """
    # Validate inputs
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    if target_type not in ("return", "close"):
        raise ValueError("target_type must be 'return' or 'close'")
    if not os.path.exists(price_dir):
        raise FileNotFoundError(f"Price directory not found: {price_dir}")

    sentiment_df = load_daily_sentiment(sentiment_csv) if sentiment_csv and os.path.exists(sentiment_csv) else pd.DataFrame()
    if not sentiment_df.empty:
        sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.normalize()

    if feature_cols is None:
        feature_cols = ["return", "log_volume", "intraday_range", "volatility_5", "ma_5", "ma_20", "daily_sentiment", "n_articles"]

    X_chunks = []
    y_chunks = []
    metas = []

    # If tickers not specified, use unique tickers from sentiment file
    if tickers is None:
        tickers = sorted(sentiment_df["Ticker"].unique())

    for t in tickers:
        try:
            price_df = load_price_for_ticker(t, price_dir)
        except FileNotFoundError:
            # skip missing tickers
            continue
        feat_df = make_features_for_ticker(price_df, sentiment_df, t, sentiment_fill=sentiment_fill)
        X_t, y_t, meta_t = create_sequences_from_ticker(feat_df, seq_len, feature_cols, target_type=target_type)
        if X_t.shape[0] > 0:
            X_chunks.append(X_t)
            y_chunks.append(y_t)
            metas.append(meta_t)
    if len(X_chunks) == 0:
        return {}, {}, pd.DataFrame()

    X_all = np.concatenate(X_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)

    return {"X": X_all, "y": y_all}, {"feature_cols": feature_cols}, meta_all


def split_time_based(
    meta: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_val_years: Tuple[int, int] = (2018, 2022),
    test_year: int = 2023,
    val_frequency_months: int = 4,
    val_duration_months: int = 1,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Split data into train/val/test temporally.

    Validation scheme: within `train_val_years`, reserve `val_duration_months` every `val_frequency_months` months as validation.
    Returns (train_ds, val_ds, test_ds)
    """
    # Validation for parameters
    if val_frequency_months <= 0:
        raise ValueError("val_frequency_months must be > 0")
    if val_duration_months >= val_frequency_months:
        raise ValueError("val_duration_months must be less than val_frequency_months")

    m = meta.copy()
    m["Date"] = pd.to_datetime(m["Date"])  # ensure
    years = m["Date"].dt.year

    # test mask
    test_mask = years == test_year

    # train+val mask
    tv_mask = (years >= train_val_years[0]) & (years <= train_val_years[1])

    # Build validation months
    # We'll compute months since start of first train year
    start = pd.Timestamp(year=train_val_years[0], month=1, day=1)
    # Number of months index for each sample
    months_since_start = (m.loc[tv_mask, "Date"].dt.year - start.year) * 12 + (m.loc[tv_mask, "Date"].dt.month - start.month)

    val_idx_mask = pd.Series(False, index=m.index)
    # For each sample in tv_mask decide if it falls in a validation block
    for idx in months_since_start.index:
        ms = months_since_start.loc[idx]
        # if ms modulo freq falls inside the duration window -> validation
        if (ms % val_frequency_months) < val_duration_months:
            val_idx_mask.loc[idx] = True

    train_mask = tv_mask & (~val_idx_mask)
    val_mask = tv_mask & val_idx_mask

    # slice arrays
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    meta_train = m.loc[train_mask.values, :].reset_index(drop=True)

    X_val = X[val_mask.values]
    y_val = y[val_mask.values]
    meta_val = m.loc[val_mask.values, :].reset_index(drop=True)

    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    meta_test = m.loc[test_mask.values, :].reset_index(drop=True)

    train_ds = TimeSeriesDataset(X_train, y_train, meta=meta_train)
    val_ds = TimeSeriesDataset(X_val, y_val, meta=meta_val)
    test_ds = TimeSeriesDataset(X_test, y_test, meta=meta_test)

    return train_ds, val_ds, test_ds


def compute_scalers_from_train(train_ds: TimeSeriesDataset, per_ticker: bool = False):
    """Fit scaler(s) on train set's X.

    - If `per_ticker` is False (default) returns a single StandardScaler fit on all train samples.
    - If `per_ticker` is True returns a dict mapping ticker -> StandardScaler fit on that ticker's train samples.

    Returns: scaler (StandardScaler or Dict[str, StandardScaler])
    """
    X = train_ds.X  # (N, seq_len, feat)
    N, S, F = X.shape

    if not per_ticker:
        flat = X.reshape(-1, F)
        scaler = StandardScaler().fit(flat)
        return scaler

    # per-ticker
    if train_ds.meta is None:
        raise ValueError("per_ticker=True requires train_ds to have .meta with a 'Ticker' column")

    scalers: Dict[str, StandardScaler] = {}
    tickers = train_ds.meta['Ticker'].unique()
    for t in tickers:
        inds = train_ds.meta.index[train_ds.meta['Ticker'] == t].tolist()
        if len(inds) == 0:
            continue
        X_t = X[inds]  # (n_samples_t, seq_len, feat)
        flat_t = X_t.reshape(-1, F)
        if flat_t.shape[0] == 0:
            continue
        scalers[t] = StandardScaler().fit(flat_t)

    return scalers


def apply_scaler_to_dataset(ds: TimeSeriesDataset, scaler: object) -> TimeSeriesDataset:
    """Apply a scaler (or dict of per-ticker scalers) to dataset features.

    - If `scaler` is a StandardScaler, apply to all samples.
    - If `scaler` is a dict mapping ticker->StandardScaler, dataset must have `meta` with 'Ticker' column.

    Returns a new TimeSeriesDataset with scaled X and same meta.
    """
    X = ds.X
    N, S, F = X.shape

    # global scaler
    if not isinstance(scaler, dict):
        X_flat = X.reshape(-1, F)
        X_scaled_flat = scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, S, F)
        return TimeSeriesDataset(X_scaled, ds.y, meta=ds.meta)

    # per-ticker scalers
    if ds.meta is None:
        raise ValueError("Per-ticker scalers require dataset to have .meta with 'Ticker' column")

    X_scaled = np.empty_like(X)
    tickers = ds.meta['Ticker'].unique()
    ticker_array = ds.meta['Ticker'].values
    for t in tickers:
        if t not in scaler:
            raise ValueError(f"No scaler found for ticker {t}")
        inds = np.where(ticker_array == t)[0]
        if inds.size == 0:
            continue
        X_sub = X[inds]  # (n, seq_len, feat)
        flat_sub = X_sub.reshape(-1, F)
        scaled_flat = scaler[t].transform(flat_sub)
        X_sub_scaled = scaled_flat.reshape(len(inds), S, F)
        X_scaled[inds] = X_sub_scaled

    return TimeSeriesDataset(X_scaled, ds.y, meta=ds.meta)


def make_dataloaders(
    train_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    test_ds: TimeSeriesDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders that always yield `(X, y)` batches.

    By default we use `_drop_meta_collate` which removes any `meta` returned
    by the dataset and returns `(X_batch, y_batch)` tensors.
    """
    if collate_fn is None:
        collate_fn = _drop_meta_collate

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def load_dataloaders(
    save_dir: str = "processed_data/small_ams",
    batch_size: int = 64,
    num_workers: int = 0,
):
    """Load pre-saved datasets from disk and return dataloaders.
    
    Simple function to load already-built datasets. If datasets don't exist,
    raises FileNotFoundError.
    
    Args:
        save_dir: Directory containing train_ds.pt, val_ds.pt, test_ds.pt
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {save_dir}")
    
    train_ds = torch.load(save_path / "train_ds.pt", weights_only=False)
    val_ds = torch.load(save_path / "val_ds.pt", weights_only=False)
    test_ds = torch.load(save_path / "test_ds.pt", weights_only=False)
    
    print(f"Loaded datasets from {save_path}")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    return make_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers)


def build_and_save_datasets(
    tickers: List[str],
    save_dir: str,
    seq_len: int = 8,
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    per_ticker_scaling: bool = False,
):
    """Build datasets from source data and save to disk.
    
    Args:
        tickers: List of ticker symbols to include
        save_dir: Directory to save train_ds.pt, val_ds.pt, test_ds.pt
        seq_len: Sequence length for time series
        sentiment_fill: 'ffill' or 'zero' for missing sentiment
        target_type: 'return' or 'close' for prediction target
        per_ticker_scaling: If True, scale each ticker separately
        
    Returns:
        train_ds, val_ds, test_ds (TimeSeriesDataset objects)
    """
    print(f"Building datasets for {len(tickers)} tickers: {tickers}")
    
    ds_dict, meta_info, meta = build_dataset_all_tickers(
        tickers=tickers,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
    )
    
    if len(ds_dict) == 0:
        raise ValueError(f"No data produced for tickers: {tickers}")
    
    X_all = ds_dict["X"]
    y_all = ds_dict["y"]
    print(f"Total samples: {len(X_all)}")
    
    # Split into train/val/test
    train_ds, val_ds, test_ds = split_time_based(meta, X_all, y_all)
    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Scale features
    if per_ticker_scaling:
        scaler = compute_scalers_from_train(train_ds, per_ticker=True)
    else:
        scaler = compute_scalers_from_train(train_ds, per_ticker=False)
    
    train_ds = apply_scaler_to_dataset(train_ds, scaler)
    val_ds = apply_scaler_to_dataset(val_ds, scaler)
    test_ds = apply_scaler_to_dataset(test_ds, scaler)
    
    # Save to disk
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(train_ds, save_path / "train_ds.pt")
    torch.save(val_ds, save_path / "val_ds.pt")
    torch.save(test_ds, save_path / "test_ds.pt")
    print(f"Saved datasets to {save_path}")
    
    return train_ds, val_ds, test_ds


def load_or_build_datasets(
    tickers: Optional[List[str]] = None,
    seq_len: int = 8,
    batch_size: int = 64,
    num_workers: int = 0,
    save_dir: str = "processed_data/small_ams",
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    force_build: bool = False,
    per_ticker_scaling: bool = False,
):
    """Load pre-saved datasets from `save_dir` or build from source.
    
    Convenience function that tries to load from disk first, then builds if needed.

    Returns: train_loader, val_loader, test_loader
    """
    if not force_build:
        try:
            return load_dataloaders(save_dir, batch_size, num_workers)
        except FileNotFoundError as e:
            print(f"Could not load saved datasets: {e}")
            print("Building datasets from source...")
    
    if tickers is None:
        raise ValueError("Must specify tickers when building datasets")
    
    # Build and save
    build_and_save_datasets(
        tickers=tickers,
        save_dir=save_dir,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
        per_ticker_scaling=per_ticker_scaling,
    )
    
    # Load the newly built datasets
    return load_dataloaders(save_dir, batch_size, num_workers)
