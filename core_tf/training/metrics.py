"""
Module for evaluating stock price and direction prediction models.
"""
import numpy as np
from typing import Tuple, Dict

# =============================================================================
# DENORMALIZATION AND UTILITIES
# =============================================================================

def denormalize_predictions(y_normalized: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    """Convert normalized PRICE predictions back to actual prices."""
    return (y_normalized + 1) * y_base


def calculate_direction_from_prices(
    y_pred: np.ndarray, 
    y_true: np.ndarray,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate predicted and actual directions from PRICE predictions.
    
    For PRICE task: compares predicted/actual price with last known price.
    
    Returns
    -------
    pred_direction : np.ndarray
        1 if predicted up, 0 if down
    true_direction : np.ndarray
        1 if actually went up, 0 if down
    """
    last_price = X[:, -1, 0]  # Last normalized price in each sequence
    
    pred_direction = (y_pred > last_price).astype(int)
    true_direction = (y_true > last_price).astype(int)
    
    return pred_direction, true_direction



# =============================================================================
# BASELINE CALCULATIONS
# =============================================================================
def naive_baseline(X: np.ndarray, y_base: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Naive baseline: predict last known price (no change).
    
    Returns actual price predictions.
    """
    last_norm_price = X[:, -1, 0]
    last_actual_price = (last_norm_price + 1) * y_base
    return last_actual_price


def moving_average_baseline(
    X: np.ndarray, 
    y_base: np.ndarray, 
    window: int = 5
) -> np.ndarray:
    """Moving average baseline."""
    ma_norm = X[:, -window:, 0].mean(axis=1)
    return (ma_norm + 1) * y_base


def momentum_baseline(X: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    """Momentum baseline: extrapolate recent trend."""
    # Calculate trend from last 5 points
    recent = X[:, -5:, 0]
    trend = recent[:, -1] - recent[:, 0]  # Change over last 5 days
    daily_trend = trend / 4  # Average daily change
    
    # Predict: last price + trend
    pred_norm = X[:, -1, 0] + daily_trend
    return (pred_norm + 1) * y_base


def direction_baseline_majority(y_train: np.ndarray) -> float:
    """Baseline for direction: always predict majority class."""
    return max((y_train == 1).mean(), (y_train == 0).mean())


def direction_baseline_random() -> float:
    """Baseline for direction: random guess."""
    return 0.5


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_price_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    X: np.ndarray
) -> Dict[str, float]:
    """Comprehensive evaluation for PRICE predictions.
    
    For PRICE task where y = (target_price / base_price) - 1
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Denormalize to actual prices
    y_true_actual = denormalize_predictions(y_true, y_base)
    y_pred_actual = denormalize_predictions(y_pred, y_base)
    
    # Price metrics
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    r2 = r2_score(y_true_actual, y_pred_actual)
    mape = np.mean(np.abs((y_true_actual - y_pred_actual) / y_true_actual)) * 100
    
    # Naive baseline comparison
    y_naive = naive_baseline(X, y_base)
    naive_mape = np.mean(np.abs((y_true_actual - y_naive) / y_true_actual)) * 100
    
    # Direction accuracy (from price predictions)
    pred_dir, true_dir = calculate_direction_from_prices(y_pred, y_true, X)
    direction_accuracy = (pred_dir == true_dir).mean() * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Naive_MAPE': naive_mape,
        'MAPE_vs_Naive': mape - naive_mape,  # Negative = better than naive
        'Direction_Accuracy': direction_accuracy
    }


def evaluate_direction_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Comprehensive evaluation for DIRECTION predictions."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba) * 100
    except:
        auc = 50.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'Confusion_Matrix': cm,
        'vs_Random': accuracy - 50.0 # Positive = better than random
    }


def print_evaluation_results(results: Dict, task: str = 'price'):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({task.upper()})")
    print("=" * 60)
    
    for key, value in results.items():
        if key == 'Confusion_Matrix':
            print(f"\n{key}:")
            print(f"  TN={value[0,0]}, FP={value[0,1]}")
            print(f"  FN={value[1,0]}, TP={value[1,1]}")
        elif isinstance(value, float):
            if 'Accuracy' in key or 'MAPE' in key or key.endswith('_Accuracy') or '_pct' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)
