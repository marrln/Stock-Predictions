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
    split_time_based,
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
    """Load pre-saved datasets from disk and return dataloaders."""
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
    
    # Backfill TickerIdx mapping for older datasets
    _backfill_ticker_idx(train_ds, val_ds, test_ds)

    # Normalize meta columns for backward compatibility (ensure lowercase aliases exist)
    for ds in (train_ds, val_ds, test_ds):
        _ensure_meta_aliases(ds)

    return make_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


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


def build_and_save_datasets(
    tickers: List[str],
    save_dir: str,
    seq_len: int = 8,
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    target_scaling: bool = True,
    train_val_years: Tuple[int, int] = (2018, 2021),
    test_years: Tuple[int, int] = (2022, 2023),
    val_duration_months: int = 12,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Build datasets from source data and save to disk."""
    print(f"Building datasets for {len(tickers)} tickers")
    
    # Build raw dataset
    ds_dict, info, meta = build_dataset_all_tickers(
        tickers=tickers,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
    )
    
    if not ds_dict:
        raise ValueError(f"No data produced for tickers: {tickers}")
    
    X_all, y_all = ds_dict["X"], ds_dict["y"]
    print(f"Total samples: {len(X_all)}")
    
    # Split temporally
    train_ds, val_ds, test_ds = split_time_based(
        meta, X_all, y_all, 
        train_val_years=train_val_years,
        test_years=test_years,
        val_duration_months=val_duration_months,
    )
    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Scale features (per-ticker)
    scaler = compute_scalers_from_train(train_ds, per_ticker=True)
    
    train_ds = apply_scaler_to_dataset(train_ds, scaler)
    val_ds = apply_scaler_to_dataset(val_ds, scaler)
    test_ds = apply_scaler_to_dataset(test_ds, scaler)
    
    # Add metadata
    feature_cols = info.get("feature_cols", [])
    for ds in [train_ds, val_ds, test_ds]:
        ds.feature_cols = feature_cols
        ds.target_type = target_type
    
    # Scale targets if requested
    y_scalers = None
    if target_scaling:
        y_scalers = _scale_targets_per_ticker(train_ds, val_ds, test_ds)
    
    # Add ticker indices
    _add_ticker_indices(train_ds, val_ds, test_ds)
    
    # Save to disk
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(train_ds, save_path / "train_ds.pt")
    torch.save(val_ds, save_path / "val_ds.pt")
    torch.save(test_ds, save_path / "test_ds.pt")
    
    if y_scalers:
        with open(save_path / "target_scaler.pkl", "wb") as f:
            pickle.dump(y_scalers, f)
        print(f"Saved per-ticker target scalers")
    
    print(f"Saved datasets to {save_path}")
    
    return train_ds, val_ds, test_ds


def _scale_targets_per_ticker(
    train_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    test_ds: TimeSeriesDataset
) -> Dict[str, Any]:
    """Scale targets per ticker using StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    
    y_scalers = {}
    
    # Fit scalers on train targets per ticker
    for ticker in sorted(train_ds.meta['Ticker'].unique()):
        indices = train_ds.meta.index[train_ds.meta['Ticker'] == ticker].tolist()
        if not indices:
            continue
        
        y_t = train_ds.y[indices].reshape(-1, 1)
        scaler = StandardScaler().fit(y_t)
        y_scalers[ticker] = scaler
        
        # Debug: print pre/post scaling stats for this ticker
        try:
            print(f"Target scaling - {ticker}: train mean={y_t.mean():.6f}, std={y_t.std():.6f}")
        except Exception:
            pass
        
        # Transform train targets
        train_ds.y[indices] = scaler.transform(y_t).reshape(-1)
        try:
            print(f"Target scaling - {ticker}: scaled train mean={train_ds.y[indices].mean():.6f}, std={train_ds.y[indices].std():.6f}")
        except Exception:
            pass
    
    # Transform validation and test targets
    for ds in [val_ds, test_ds]:
        if ds.meta is None:
            continue
        
        scaled_y = ds.y.copy()
        for i, ticker in enumerate(ds.meta['Ticker'].values):
            scaler = y_scalers.get(ticker)
            if scaler is not None:
                scaled_y[i] = scaler.transform(ds.y[i].reshape(1, -1)).item()
        
        ds.y = scaled_y
    
    # Attach scalers to datasets
    for ds in [train_ds, val_ds, test_ds]:
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


def load_or_build_datasets(
    tickers: Optional[List[str]] = None,
    seq_len: int = 8,
    batch_size: int = 64,
    num_workers: int = 0,
    save_dir: str = "processed_data/small_ams",
    sentiment_fill: str = "ffill",
    target_type: str = "return",
    force_build: bool = False,
    target_scaling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load pre-saved datasets or build from source."""
    if not force_build:
        try:
            return load_dataloaders(save_dir, batch_size, num_workers)
        except FileNotFoundError:
            print("Saved datasets not found. Building from source...")
    
    if tickers is None:
        raise ValueError("Must specify tickers when building datasets")
    
    # Build and save
    build_and_save_datasets(
        tickers=tickers,
        save_dir=save_dir,
        seq_len=seq_len,
        sentiment_fill=sentiment_fill,
        target_type=target_type,
        target_scaling=target_scaling,
    )
    
    # Load the newly built datasets
    return load_dataloaders(save_dir, batch_size, num_workers)