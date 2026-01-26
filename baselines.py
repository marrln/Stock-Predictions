"""Baseline models and evaluators for quick comparisons.

Currently implements:
- persistence baseline (predict last observed return)
"""
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import torch


def evaluate_persistence_on_loader(loader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a persistence baseline on a DataLoader.

    The persistence baseline predicts next-day return as the most recent
    observed return in the input sequence (i.e., `X[:, -1, return_idx]`).

    Returns (metrics_dict, preds, trues) where metrics_dict contains mse, mae, rmse, dir_acc.
    """
    ds = loader.dataset

    # If the dataset has meta and `last_return` recorded, use that raw value
    if hasattr(ds, "meta") and ds.meta is not None and "last_return" in ds.meta.columns:
        preds = ds.meta["last_return"].values.astype(float)
        trues = ds.y.astype(float)
        mse = float(((preds - trues) ** 2).mean())
        mae = float(np.abs(preds - trues).mean())
        rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
        dir_acc = float((np.sign(preds) == np.sign(trues)).mean())
        metrics = {"mse": mse, "mae": mae, "rmse": rmse, "dir_acc": dir_acc}
        return metrics, preds, trues

    # Fallback: extract last return from X (may be scaled) — warning in docs
    preds = []
    trues = []

    # Determine index of 'return' feature if available
    return_idx = 0  # default fallback
    if hasattr(ds, "feature_cols") and ds.feature_cols is not None:
        try:
            return_idx = ds.feature_cols.index("return")
        except ValueError:
            # fallback to 0
            return_idx = 0

    with torch.no_grad():
        for batch in loader:
            xb, yb = batch
            # xb: (B, S, F)
            xb = xb.numpy() if hasattr(xb, "numpy") else np.array(xb)
            yb = yb.numpy() if hasattr(yb, "numpy") else np.array(yb)
            # persistence prediction = last timestep's return feature
            last_ret = xb[:, -1, return_idx]
            preds.append(last_ret.reshape(-1))
            trues.append(yb.reshape(-1))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = float(((preds - trues) ** 2).mean())
    mae = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    dir_acc = float((np.sign(preds) == np.sign(trues)).mean())

    metrics = {"mse": mse, "mae": mae, "rmse": rmse, "dir_acc": dir_acc}
    return metrics, preds, trues
    mse = float(((preds - trues) ** 2).mean())
    mae = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    dir_acc = float((np.sign(preds) == np.sign(trues)).mean())

    metrics = {"mse": mse, "mae": mae, "rmse": rmse, "dir_acc": dir_acc}
    return metrics, preds, trues
