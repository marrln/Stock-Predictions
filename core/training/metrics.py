"""Metrics computation functions."""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional
import warnings

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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


def compute_trading_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 1.0,
    target_type: str = "return"
) -> Dict[str, float]:
    """Compute metrics relevant for trading and stock prediction.
    
    Args:
        predictions: Predicted values
        targets: True values
        threshold: Threshold for filtering significant moves (in same units as targets)
        target_type: Type of target ('return', 'pct_change', 'close', etc.)
        
    Returns:
        Dictionary of trading-relevant metrics
    """
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()
    
    if len(predictions) != len(targets) or len(predictions) == 0:
        return {}
    
    metrics = {}
    
    # Thresholded directional accuracy (only significant moves)
    mask = np.abs(targets) > threshold
    if mask.sum() > 0:
        dir_correct = np.sign(predictions[mask]) == np.sign(targets[mask])
        metrics["dir_acc_thresh"] = float(np.mean(dir_correct))
        metrics["dir_acc_thresh_n"] = int(mask.sum())
    else:
        metrics["dir_acc_thresh"] = np.nan
        metrics["dir_acc_thresh_n"] = 0
    
    # Information Coefficient (correlation)
    if SCIPY_AVAILABLE and len(predictions) > 2:
        try:
            ic_pearson, _ = pearsonr(predictions, targets)
            metrics["ic_pearson"] = float(ic_pearson)
        except:
            metrics["ic_pearson"] = np.nan
        
        try:
            ic_spearman, _ = spearmanr(predictions, targets)
            metrics["ic_spearman"] = float(ic_spearman)
        except:
            metrics["ic_spearman"] = np.nan
    
    # Hit rates (what fraction of predictions within error threshold)
    if target_type in ["return", "pct_change"]:
        # For returns/pct_change, use percentage thresholds
        metrics["hit_rate_1pct"] = float(np.mean(np.abs(predictions - targets) < 1.0))
        metrics["hit_rate_2pct"] = float(np.mean(np.abs(predictions - targets) < 2.0))
    elif target_type == "close":
        # For prices, use percentage of price
        mean_price = np.mean(np.abs(targets))
        if mean_price > 0:
            pct_errors = np.abs(predictions - targets) / mean_price * 100
            metrics["hit_rate_5pct"] = float(np.mean(pct_errors < 5.0))
            metrics["hit_rate_10pct"] = float(np.mean(pct_errors < 10.0))
    
    # Simple trading simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        positions = np.sign(predictions)  # +1 long, -1 short, 0 neutral
        
        # Strategy returns (position * actual return)
        strategy_returns = positions * targets
        
        metrics["strategy_total_return"] = float(np.sum(strategy_returns))
        metrics["strategy_mean_return"] = float(np.mean(strategy_returns))
        
        # Strategy Sharpe (annualized if daily returns)
        std_ret = np.std(strategy_returns)
        if std_ret > 0 and not np.isnan(std_ret):
            sharpe_daily = np.mean(strategy_returns) / std_ret
            metrics["strategy_sharpe_daily"] = float(sharpe_daily)
            # Annualized (assume 252 trading days)
            metrics["strategy_sharpe_annual"] = float(sharpe_daily * np.sqrt(252))
        else:
            metrics["strategy_sharpe_daily"] = 0.0
            metrics["strategy_sharpe_annual"] = 0.0
        
        # Win rate and profit factor
        winning_trades = strategy_returns > 0
        losing_trades = strategy_returns < 0
        
        metrics["strategy_win_rate"] = float(np.mean(winning_trades))
        
        total_profit = np.sum(strategy_returns[winning_trades]) if winning_trades.any() else 0
        total_loss = -np.sum(strategy_returns[losing_trades]) if losing_trades.any() else 0
        
        if total_loss > 0:
            metrics["strategy_profit_factor"] = float(total_profit / total_loss)
        else:
            metrics["strategy_profit_factor"] = float('inf') if total_profit > 0 else 0.0
        
        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics["strategy_max_drawdown"] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    
    return metrics


def get_metrics_for_target_type(target_type: str) -> Dict[str, bool]:
    """Return appropriate metric flags based on target type.
    
    Args:
        target_type: Type of prediction target
        
    Returns:
        Dictionary of boolean flags for which metrics to compute
    """
    # For returns and percent changes: focus on directional and trading metrics
    if target_type in ["return", "pct_change", "log_return"]:
        return {
            "include_directional": True,
            "include_r2": False,  # Not meaningful for returns
            "include_sharpe": False,  # Misleading as currently implemented
            "include_trading_metrics": True,
            "trading_threshold": 1.0,  # 1% threshold for significant moves
        }
    
    # For absolute prices: traditional regression metrics make more sense
    elif target_type in ["close", "price"]:
        return {
            "include_directional": True,  # Still useful
            "include_r2": True,  # More meaningful for prices
            "include_sharpe": False,  # Still not useful
            "include_trading_metrics": False,
            "trading_threshold": 0.0,
        }
    
    # For multi-day returns
    elif target_type in ["return_5d", "return_10d", "return_20d"]:
        return {
            "include_directional": True,
            "include_r2": False,
            "include_sharpe": False,
            "include_trading_metrics": True,
            "trading_threshold": 2.0,  # Higher threshold for multi-day
        }
    
    # Default: conservative set
    else:
        return {
            "include_directional": True,
            "include_r2": True,
            "include_sharpe": False,
            "include_trading_metrics": False,
            "trading_threshold": 0.0,
        }


def compute_metrics_auto(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_type: str = "return",
    ticker_ids: Optional[np.ndarray] = None,
    fold_ids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute appropriate metrics based on target type.
    
    Automatically selects relevant metrics based on what you're predicting.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        target_type: Type of target ('return', 'pct_change', 'close', etc.)
        ticker_ids: Optional ticker identifiers for per-ticker breakdown
        fold_ids: Optional fold identifiers for per-fold breakdown
        
    Returns:
        Dictionary of metrics appropriate for the target type
    """
    # Get appropriate metric flags
    flags = get_metrics_for_target_type(target_type)
    
    # Compute base regression metrics
    metrics = compute_regression_metrics(
        predictions=predictions,
        targets=targets,
        include_directional=flags["include_directional"],
        include_r2=flags["include_r2"],
        include_sharpe=flags["include_sharpe"],
        ticker_ids=ticker_ids,
        fold_ids=fold_ids,
        directional_threshold=flags["trading_threshold"],
        include_confusion=True,
    )
    
    # Add trading metrics if appropriate
    if flags["include_trading_metrics"]:
        trading_metrics = compute_trading_metrics(
            predictions=predictions,
            targets=targets,
            threshold=flags["trading_threshold"],
            target_type=target_type
        )
        metrics.update(trading_metrics)
    
    # Add metadata
    metrics["target_type"] = target_type
    metrics["metrics_note"] = f"Metrics optimized for {target_type} prediction"
    
    return metrics
