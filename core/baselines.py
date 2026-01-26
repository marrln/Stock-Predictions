"""Baseline models and evaluators for quick comparisons."""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Tuple
from .data.dataset import TimeSeriesDataset


def evaluate_persistence_on_loader(
    loader: torch.utils.data.DataLoader
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a persistence baseline on a DataLoader.
    
    The persistence baseline predicts next-day return as the most recent
    observed return in the input sequence.
    
    Returns:
        Tuple containing (metrics_dict, predictions, true_values)
    """
    ds = loader.dataset
    
    # Use precomputed metadata if available; support both 'return' and 'close' targets
    if hasattr(ds, "meta") and ds.meta is not None:
        if getattr(ds, 'target_type', 'return') == 'close' and 'last_close' in ds.meta.columns:
            return _evaluate_from_metadata(ds)
        if 'last_return' in ds.meta.columns:
            return _evaluate_from_metadata(ds)

    # Fallback: extract from features (last feature value) when possible
    return _evaluate_from_features(ds, loader)


def _evaluate_from_metadata(ds: TimeSeriesDataset) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate using precomputed metadata.

    Supports both 'return' and 'close' targets when metadata contains
    'last_return' or 'last_close'.
    """
    target_type = getattr(ds, 'target_type', 'return')
    if target_type == 'close' and 'last_close' in ds.meta.columns:
        preds = ds.meta['last_close'].values.astype(float)
        trues = ds.y.astype(float)
        # If the dataset targets are scaled per-ticker, transform preds into
        # the same scaled space before computing metrics.
        ts = getattr(ds, 'target_scaler', None)
        if isinstance(ts, dict):
            scaled_preds = preds.copy()
            from collections import defaultdict
            idxs_by_ticker = defaultdict(list)
            for i, t in enumerate(ds.meta['Ticker']):
                idxs_by_ticker[t].append(i)
            import numpy as _np
            for t, idxs in idxs_by_ticker.items():
                sc = ts.get(t)
                if sc is None:
                    continue
                try:
                    scaled_vals = sc.transform(preds[idxs].reshape(-1,1)).ravel()
                    scaled_preds[idxs] = scaled_vals
                except Exception:
                    # If transform fails, leave raw values (best-effort)
                    continue
            preds = scaled_preds
        elif ts is not None:
            try:
                preds = ts.transform(preds.reshape(-1,1)).ravel()
            except Exception:
                pass

        return compute_metrics(preds, trues), preds, trues

    if 'last_return' in ds.meta.columns:
        preds = ds.meta['last_return'].values.astype(float)
        trues = ds.y.astype(float)
        return compute_metrics(preds, trues), preds, trues

    raise RuntimeError('No suitable metadata (last_return or last_close) found for persistence baseline')


def _evaluate_from_features(
    ds: TimeSeriesDataset, 
    loader: torch.utils.data.DataLoader
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate by extracting features from input sequences.

    For 'return' targets we extract the last timestep's return feature. For
    'close' targets we try to use 'last_close' from meta, otherwise attempt to
    find a 'close' feature in `ds.feature_cols`.
    """
    target_type = getattr(ds, 'target_type', 'return')
    preds_list, trues_list = [], []

    # If target is 'close' prefer metadata
    if target_type == 'close':
        if hasattr(ds, 'meta') and ds.meta is not None and 'last_close' in ds.meta.columns:
            preds = ds.meta['last_close'].values.astype(float)
            trues = ds.y.astype(float)
            return compute_metrics(preds, trues), preds, trues
        # else, try to find 'close' in feature_cols
        close_idx = None
        if hasattr(ds, 'feature_cols') and ds.feature_cols is not None:
            try:
                close_idx = ds.feature_cols.index('close')
            except ValueError:
                close_idx = None
        if close_idx is None:
            raise RuntimeError("Cannot compute persistence baseline for 'close' target: no 'last_close' in meta and no 'close' feature available")

    # Default path: extract return feature
    return_idx = _get_return_feature_index(ds)

    with torch.no_grad():
        for batch in loader:
            xb, yb = _unpack_batch(batch)

            if target_type == 'close' and close_idx is not None:
                last_val = xb[:, -1, close_idx]
            else:
                last_val = xb[:, -1, return_idx]

            preds_list.append(last_val.reshape(-1))
            trues_list.append(yb.reshape(-1))
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    return compute_metrics(preds, trues), preds, trues


def _get_return_feature_index(ds: TimeSeriesDataset) -> int:
    """Get the index of the return feature in the feature columns."""
    if hasattr(ds, "feature_cols") and ds.feature_cols is not None:
        try:
            return ds.feature_cols.index("return")
        except ValueError:
            return 0
    return 0


def _unpack_batch(batch) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack batch into X and y arrays."""
    if isinstance(batch, (list, tuple)):
        xb, yb = batch[0], batch[1]
    else:
        xb, yb = batch["input"], batch["target"]
    
    xb = xb.numpy() if hasattr(xb, "numpy") else np.array(xb)
    yb = yb.numpy() if hasattr(yb, "numpy") else np.array(yb)
    
    return xb, yb


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics between predictions and true values."""
    errors = preds - trues
    squared_errors = errors ** 2
    
    mse = float(squared_errors.mean())
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(mse))
    dir_acc = float((np.sign(preds) == np.sign(trues)).mean())
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "dir_acc": dir_acc
    }