"""Metrics computation functions."""
from __future__ import annotations

import numpy as np
from typing import Dict
import warnings

def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    include_directional: bool = True,
    include_r2: bool = False,
) -> Dict[str, float]:
    """Compute regression metrics with more options."""
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()
    
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have same length. "
            f"Got {len(predictions)} and {len(targets)}"
        )
    
    # Compute errors
    errors = predictions - targets
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)
    
    # Basic metrics
    mse = float(np.mean(squared_errors))
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(mse))
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
    
    # Directional accuracy
    if include_directional:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correct_direction = np.sign(predictions) == np.sign(targets)
            dir_acc = float(np.mean(correct_direction))
        metrics["dir_acc"] = dir_acc
    
    # R-squared
    if include_r2 and len(targets) > 1:
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum(squared_errors)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        metrics["r2"] = float(r2)
    
    return metrics