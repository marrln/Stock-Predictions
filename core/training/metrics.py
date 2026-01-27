"""Metrics computation functions."""
from __future__ import annotations

import numpy as np
from typing import Dict
import warnings

def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    include_directional: bool = True,
    include_r2: bool = True,
    include_sharpe: bool = True,
    ticker_ids: "Optional[np.ndarray]" = None,
    fold_ids: "Optional[np.ndarray]" = None,
    directional_threshold: float = 0.0,
    include_confusion: bool = True,
) -> Dict[str, object]:
    """Compute regression metrics with more options.

    Additional features:
      - r2: Coefficient of determination when include_r2=True
      - sharpe_pred / sharpe_true: mean/std for predictions and targets when include_sharpe=True
      - confusion matrix (TP/TN/FP/FN) when include_confusion=True
      - thresholded directional accuracy (only for targets with abs >= directional_threshold)
      - optional per-ticker and per-fold breakdowns when `ticker_ids` or `fold_ids` supplied

    Returns:
        A dictionary of metrics. May contain nested dicts under 'per_ticker' and 'per_fold'.
    """
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()
    
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have same length. "
            f"Got {len(predictions)} and {len(targets)}"
        )
    n = len(predictions)

    # Compute errors
    errors = predictions - targets
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)
    
    # Basic metrics
    mse = float(np.mean(squared_errors))
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(mse))
    
    metrics: Dict[str, object] = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "n": int(n),
    }
    
    # Directional accuracy (sign-based)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_sign = np.sign(predictions)
        true_sign = np.sign(targets)
        correct_direction = pred_sign == true_sign
        dir_acc = float(np.mean(correct_direction)) if n > 0 else 0.0
    if include_directional:
        metrics["dir_acc"] = dir_acc
    
    # Thresholded directional accuracy (only consider samples with |target| >= threshold)
    if directional_threshold is not None and directional_threshold > 0.0:
        mask = np.abs(targets) >= float(directional_threshold)
        if mask.any():
            dir_acc_thresh = float(np.mean(correct_direction[mask]))
            metrics["dir_acc_thresh"] = dir_acc_thresh
            metrics["dir_acc_thresh_n"] = int(mask.sum())
        else:
            metrics["dir_acc_thresh"] = float("nan")
            metrics["dir_acc_thresh_n"] = 0
    
    # Confusion matrix (binary + vs non-positive)
    if include_confusion:
        # Define positive as > 0, negative as <= 0
        pred_pos = predictions > 0
        true_pos = targets > 0
        tp = int(np.sum(pred_pos & true_pos))
        tn = int(np.sum(~pred_pos & ~true_pos))
        fp = int(np.sum(pred_pos & ~true_pos))
        fn = int(np.sum(~pred_pos & true_pos))
        metrics.update({
            "conf_TP": tp,
            "conf_TN": tn,
            "conf_FP": fp,
            "conf_FN": fn,
        })

    # R-squared
    if include_r2 and len(targets) > 1:
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum(squared_errors)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        metrics["r2"] = float(r2)
    
    # Sharpe-like ratios (non-annualized): pred mean / pred std, target mean / target std
    if include_sharpe:
        def _sharpe(arr: np.ndarray) -> float:
            arr = np.asarray(arr).ravel()
            std = np.std(arr)
            if std == 0 or np.isnan(std):
                return 0.0
            return float(np.mean(arr) / std)
        metrics["sharpe_pred"] = _sharpe(predictions)
        metrics["sharpe_true"] = _sharpe(targets)

    # Per-ticker breakdown
    if ticker_ids is not None:
        ticker_ids = np.asarray(ticker_ids)
        if len(ticker_ids) != n:
            raise ValueError("ticker_ids length must match predictions/targets")
        per_ticker = {}
        for t in np.unique(ticker_ids):
            mask = ticker_ids == t
            if mask.sum() == 0:
                continue
            sub_metrics = compute_regression_metrics(predictions[mask], targets[mask],
                                                     include_directional=include_directional,
                                                     include_r2=include_r2,
                                                     include_sharpe=include_sharpe,
                                                     ticker_ids=None,
                                                     fold_ids=None,
                                                     directional_threshold=directional_threshold,
                                                     include_confusion=include_confusion)
            per_ticker[str(t)] = sub_metrics
        metrics["per_ticker"] = per_ticker

    # Per-fold breakdown
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        if len(fold_ids) != n:
            raise ValueError("fold_ids length must match predictions/targets")
        per_fold = {}
        for f in np.unique(fold_ids):
            mask = fold_ids == f
            if mask.sum() == 0:
                continue
            sub_metrics = compute_regression_metrics(predictions[mask], targets[mask],
                                                     include_directional=include_directional,
                                                     include_r2=include_r2,
                                                     include_sharpe=include_sharpe,
                                                     ticker_ids=None,
                                                     fold_ids=None,
                                                     directional_threshold=directional_threshold,
                                                     include_confusion=include_confusion)
            per_fold[str(f)] = sub_metrics
        metrics["per_fold"] = per_fold

    return metrics