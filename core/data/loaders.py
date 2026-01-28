"""DataLoader creation and management."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import TimeSeriesDataset
from .preprocessing import (
    build_dataset_all_tickers,
    split_time_based_rolling,
    split_time_based_expanding,
    compute_scalers_from_train,
    apply_scaler_to_dataset,
)


def _drop_meta_collate(batch):
    """Collate function that drops metadata and returns (X, y) or (X, y, ticker_idx)."""
    X_list, y_list, ticker_idx_list = [], [], []
    collect_ticker = True
    
    for sample in batch:
        if isinstance(sample, dict):
            x = sample.get("input", sample.get("x", None))
            y = sample.get("target", sample.get("y", None))
            if x is None or y is None:
                raise ValueError("Batch dict must contain 'input'/'x' and 'target'/'y' keys")
            md = sample.get("meta", {})
        else:
            if len(sample) < 2:
                raise ValueError("Dataset sample must have at least (X, y)")
            x, y = sample[0], sample[1]
            md = sample[2] if len(sample) >= 3 else None
        
        X_list.append(x)
        y_list.append(y)
        
        if md is None or "TickerIdx" not in md:
            collect_ticker = False
        else:
            ticker_idx_list.append(int(md.get("TickerIdx")))
    
    X_batch = torch.utils.data._utils.collate.default_collate(X_list)
    y_batch = torch.utils.data._utils.collate.default_collate(y_list)
    
    if collect_ticker and len(ticker_idx_list) == len(X_list):
        ticker_tensor = torch.tensor(ticker_idx_list, dtype=torch.long)
        return X_batch, y_batch, ticker_tensor
    
    return X_batch, y_batch


def make_dataloaders(
    train_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    test_ds: TimeSeriesDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test sets."""
    if collate_fn is None:
        collate_fn = _drop_meta_collate
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def load_dataloaders(
    save_dir: str = "processed_data/small_ams",
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load pre-saved datasets from disk and return dataloaders (DEPRECATED - for single fold only)."""
    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {save_dir}")
    
    try:
        train_ds = torch.load(save_path / "train_ds.pt", weights_only=False)
        val_ds = torch.load(save_path / "val_ds.pt", weights_only=False)
        test_ds = torch.load(save_path / "test_ds.pt", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets from {save_path}: {e}") from e
    
    print(f"Loaded datasets from {save_path}")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    _backfill_ticker_idx(train_ds, val_ds, test_ds)

    for ds in (train_ds, val_ds, test_ds):
        _ensure_meta_aliases(ds)

    return make_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def load_rolling_folds(
    save_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> List[Tuple[DataLoader, DataLoader, Optional[DataLoader]]]:
    """Load rolling folds from disk and return list of dataloader tuples."""
    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {save_dir}")
    
    fold_info_path = save_path / "fold_info.pkl"
    if not fold_info_path.exists():
        raise FileNotFoundError(f"fold_info.pkl not found in {save_dir}")
    
    with open(fold_info_path, "rb") as f:
        fold_info = pickle.load(f)
    
    num_folds = fold_info["num_folds"]
    fold_loaders = []
    
    for fold_idx in range(num_folds):
        fold_dir = save_path / f"fold_{fold_idx}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        
        train_ds = torch.load(fold_dir / "train_ds.pt", weights_only=False)
        val_ds = torch.load(fold_dir / "val_ds.pt", weights_only=False)
        
        test_path = fold_dir / "test_ds.pt"
        test_ds = torch.load(test_path, weights_only=False) if test_path.exists() else None
        
        _backfill_ticker_idx(*([train_ds, val_ds, test_ds] if test_ds else [train_ds, val_ds]))
        for ds in [train_ds, val_ds] + ([test_ds] if test_ds else []):
            _ensure_meta_aliases(ds)
        
        if test_ds is not None:
            train_loader, val_loader, test_loader = make_dataloaders(
                train_ds, val_ds, test_ds, batch_size, num_workers
            )
        else:
            train_loader, val_loader, _ = make_dataloaders(
                train_ds, val_ds, train_ds, batch_size, num_workers
            )
            test_loader = None
        
        fold_loaders.append((train_loader, val_loader, test_loader))
    
    print(f"Loaded {num_folds} folds from {save_path}")
    return fold_loaders


def _backfill_ticker_idx(*datasets: TimeSeriesDataset):
    """Add TickerIdx to datasets that lack it and create lowercase alias 'tickeridx'."""
    all_tickers = []
    for ds in datasets:
        if hasattr(ds, 'meta') and ds.meta is not None and 'Ticker' in ds.meta.columns:
            all_tickers.append(ds.meta[['Ticker']])
    
    if not all_tickers:
        return
    
    all_tickers_df = pd.concat(all_tickers)
    unique_tickers = all_tickers_df['Ticker'].unique()
    ticker_to_idx = {t: i for i, t in enumerate(sorted(unique_tickers))}
    
    for ds in datasets:
        if hasattr(ds, 'meta') and ds.meta is not None and 'Ticker' in ds.meta.columns:
            if 'TickerIdx' not in ds.meta.columns:
                ds.meta['TickerIdx'] = ds.meta['Ticker'].map(ticker_to_idx)
                # also provide lowercase alias for compatibility
                ds.meta['tickeridx'] = ds.meta['TickerIdx']
                ds.ticker_to_idx = ticker_to_idx


def _ensure_meta_aliases(ds: TimeSeriesDataset):
    """Ensure dataset.meta contains lowercase aliases and common capitalized columns for compatibility."""
    if not hasattr(ds, 'meta') or ds.meta is None:
        return
    for col in list(ds.meta.columns):
        low = col.lower()
        if low not in ds.meta.columns:
            ds.meta[low] = ds.meta[col]
    if 'ticker' in ds.meta.columns and 'Ticker' not in ds.meta.columns:
        ds.meta['Ticker'] = ds.meta['ticker']
    if 'date' in ds.meta.columns and 'Date' not in ds.meta.columns:
        ds.meta['Date'] = ds.meta['date']
    if 'tickeridx' in ds.meta.columns and 'TickerIdx' not in ds.meta.columns:
        ds.meta['TickerIdx'] = ds.meta['tickeridx']


def build_and_save_rolling_folds(
    tickers: List[str],
    save_dir: str,
    seq_len: int = 30,
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    target_scaling: bool = True,
    train_days: int = 750,
    val_days: int = 125,
    test_days: Optional[int] = 125,
    step_days: int = 125,
    market_csv: Optional[str] = None,
    mode: str = "rolling",
) -> List[Tuple[TimeSeriesDataset, TimeSeriesDataset, Optional[TimeSeriesDataset]]]:
    """Build rolling or expanding fold datasets from source data and save to disk.
    
    Args:
        mode: 'rolling' (fixed window) or 'expanding' (growing training set)
    """
    fold_type = "expanding" if mode == "expanding" else "rolling"
    print(f"Building {fold_type} fold datasets for {len(tickers)} tickers")
    
    # Automatically disable target scaling for pct_change, return, and log_return targets
    # (they're already normalized and scaling defeats the purpose)
    if target_type in ["pct_change", "return", "log_return"] and target_scaling:
        print(f"⚠️  Disabling target scaling for '{target_type}' (already normalized)")
        target_scaling = False
    
    ds_dict, info, meta = build_dataset_all_tickers(
        tickers=tickers,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
        market_csv=market_csv,
    )
    
    if not ds_dict:
        raise ValueError(f"No data produced for tickers: {tickers}")
    
    X_all, y_all = ds_dict["X"], ds_dict["y"]
    print(f"Total samples: {len(X_all)}")
    
    # Diagnostic: compute number of unique trading days and expected folds (date-based)
    unique_dates = meta['Date'].dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    D = len(unique_dates)
    # Compute expected folds using the same logic as splitter
    window_len = train_days + seq_len + val_days
    if test_days is not None:
        window_len += seq_len + test_days
    expected_folds = 0
    if D > window_len:
        expected_folds = (D - window_len) // step_days + 1
    print(f"Unique trading days: {D}, expected folds (approx): {expected_folds}")

    # Choose split function based on mode
    if mode == "expanding":
        folds = split_time_based_expanding(
            meta, X_all, y_all,
            initial_train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            seq_len=seq_len,
            step_days=step_days,
        )
    else:  # rolling (default)
        folds = split_time_based_rolling(
            meta, X_all, y_all,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            seq_len=seq_len,
            step_days=step_days,
        )
    
    print(f"Created {len(folds)} {fold_type} folds")
    
    processed_folds = []
    feature_cols = info.get("feature_cols", [])
    
    for fold_idx, (train_ds, val_ds, test_ds) in enumerate(folds):
        scaler = compute_scalers_from_train(train_ds, per_ticker=True)
        
        train_ds = apply_scaler_to_dataset(train_ds, scaler)
        val_ds = apply_scaler_to_dataset(val_ds, scaler)
        if test_ds is not None:
            test_ds = apply_scaler_to_dataset(test_ds, scaler)
        
        for ds in [train_ds, val_ds] + ([test_ds] if test_ds else []):
            ds.feature_cols = feature_cols
            ds.target_type = target_type
        
        if target_scaling:
            if test_ds is not None:
                _scale_targets_per_ticker(train_ds, val_ds, test_ds)
            else:
                _scale_targets_per_ticker(train_ds, val_ds)
        
        _add_ticker_indices(*([train_ds, val_ds, test_ds] if test_ds else [train_ds, val_ds]))
        
        processed_folds.append((train_ds, val_ds, test_ds))
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, (train_ds, val_ds, test_ds) in enumerate(processed_folds):
        fold_dir = save_path / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        torch.save(train_ds, fold_dir / "train_ds.pt")
        torch.save(val_ds, fold_dir / "val_ds.pt")
        if test_ds is not None:
            torch.save(test_ds, fold_dir / "test_ds.pt")
    
    with open(save_path / "fold_info.pkl", "wb") as f:
        pickle.dump({
            "num_folds": len(processed_folds),
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "seq_len": seq_len,
            "feature_cols": feature_cols,
        }, f)
    
    print(f"Saved {len(processed_folds)} folds to {save_path}")
    
    return processed_folds


def _scale_targets_per_ticker(
    train_ds: TimeSeriesDataset,
    *other_datasets: TimeSeriesDataset
) -> Dict[str, Any]:
    """Scale targets per ticker using StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    
    y_scalers = {}
    
    for ticker in sorted(train_ds.meta['Ticker'].unique()):
        indices = train_ds.meta.index[train_ds.meta['Ticker'] == ticker].tolist()
        if not indices:
            continue
        
        y_t = train_ds.y[indices].reshape(-1, 1)
        scaler = StandardScaler().fit(y_t)
        y_scalers[ticker] = scaler
        
        train_ds.y[indices] = scaler.transform(y_t).reshape(-1)
    
    for ds in other_datasets:
        if ds is None or ds.meta is None:
            continue
        
        scaled_y = ds.y.copy()
        for i, ticker in enumerate(ds.meta['Ticker'].values):
            scaler = y_scalers.get(ticker)
            if scaler is not None:
                scaled_y[i] = scaler.transform(ds.y[i].reshape(1, -1)).item()
        
        ds.y = scaled_y
    
    for ds in [train_ds] + list(other_datasets):
        if ds is not None:
            ds.target_scaler = y_scalers
    
    return y_scalers


def _add_ticker_indices(*datasets: TimeSeriesDataset):
    """Add TickerIdx column to dataset metadata."""
    # Collect all unique tickers
    all_tickers = set()
    for ds in datasets:
        if hasattr(ds, 'meta') and ds.meta is not None:
            all_tickers.update(ds.meta['Ticker'].unique())
    
    ticker_to_idx = {t: i for i, t in enumerate(sorted(all_tickers))}
    
    # Add TickerIdx to each dataset
    for ds in datasets:
        if hasattr(ds, 'meta') and ds.meta is not None:
            ds.meta["TickerIdx"] = ds.meta["Ticker"].map(ticker_to_idx)
            ds.ticker_to_idx = ticker_to_idx


def load_or_build_rolling_folds(
    tickers: Optional[List[str]] = None,
    seq_len: int = 30,
    batch_size: int = 64,
    num_workers: int = 0,
    save_dir: str = "processed_data/rolling",
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    force_build: bool = False,
    target_scaling: bool = True,
    train_days: int = 750,
    val_days: int = 125,
    test_days: Optional[int] = 125,
    step_days: int = 125,
    market_csv: Optional[str] = None,
    mode: str = "rolling",
) -> List[Tuple[DataLoader, DataLoader, Optional[DataLoader]]]:
    """Load pre-saved rolling/expanding folds or build from source.
    
    Args:
        mode: 'rolling' (fixed window) or 'expanding' (growing training set)
    """
    if not force_build:
        try:
            return load_rolling_folds(save_dir, batch_size, num_workers)
        except FileNotFoundError:
            print("Saved folds not found. Building from source...")
    
    if tickers is None:
        raise ValueError("Must specify tickers when building datasets")
    
    build_and_save_rolling_folds(
        tickers=tickers,
        save_dir=save_dir,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
        target_scaling=target_scaling,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        step_days=step_days,
        market_csv=market_csv,
        mode=mode,
    )
    
    return load_rolling_folds(save_dir, batch_size, num_workers)