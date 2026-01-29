"""
Preprocessing - Complete Feature Set
========================================

Supports:
A) Direction Prediction (binary classification)
B) Multi-horizon prediction (1, 5, 10 days ahead)
C) Price Prediction (regression)

"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import json


class PredictionTask(Enum):
    """Type of prediction task."""
    PRICE = "price"           # Predict normalized price (regression)
    DIRECTION = "direction"   # Predict up/down (binary classification)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    seq_len: int = 50
    horizon: int = 1                    # Days ahead to predict (1, 5, 10, etc.)
    task: PredictionTask = PredictionTask.PRICE
    
    # Features
    feature_cols: List[str] = None
    cols_to_norm: List[int] = None      # Columns for relative normalization
    n_articles_scale: float = 10.0      # Scale factor for n_articles
    
    # Direction threshold (for classification)
    direction_threshold: float = 0.0    # 0 = any move, 0.001 = 0.1% minimum move
    
    def __post_init__(self):
        if self.feature_cols is None:
            self.feature_cols = ['close', 'volume', 'daily_sentiment', 'n_articles']
        if self.cols_to_norm is None:
            self.cols_to_norm = [0, 1]  # close, volume
            
    def save_json(self, filepath: str):
        """Save config to JSON."""
        data = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'PreprocessingConfig':
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# CORE PREPROCESSING FUNCTIONS
# =============================================================================

def normalize_window_relative(
    window: np.ndarray, 
    cols_to_norm: List[int],
    n_articles_idx: Optional[int] = None,
    n_articles_scale: float = 10.0
) -> np.ndarray:
    """Normalize window with improved handling.
    
    - Price/volume: relative to first value
    - n_articles: scaled by fixed factor
    - sentiment: unchanged (already in [-1, 1])
    """
    normalized = window.copy()
    
    # Relative normalization for price/volume
    for col_i in cols_to_norm:
        first_value = window[0, col_i]
        if abs(first_value) < 1e-8:
            first_value = 1e-8
        normalized[:, col_i] = (window[:, col_i] / first_value) - 1
    
    # Scale n_articles
    if n_articles_idx is not None:
        normalized[:, n_articles_idx] = window[:, n_articles_idx] / n_articles_scale
    
    return normalized


def create_sequences(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Create sequences with support for all prediction tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and feature columns.
    config : PreprocessingConfig
        Preprocessing configuration.
    
    Returns
    -------
    X : np.ndarray
        Input sequences, shape (n_samples, seq_len-1, n_features).
    y : np.ndarray
        Targets:
        - PRICE: normalized price at horizon
        - DIRECTION: 0 (down) or 1 (up)
    y_base : np.ndarray
        Base values for denormalization.
    metadata : dict
        Additional info (class weights for direction, etc.)
    """
    
    # Find available columns (case-insensitive)
    available_cols = []
    col_mapping = {}
    for col in config.feature_cols:
        matches = df.columns[df.columns.str.lower() == col.lower()]
        if len(matches) > 0:
            available_cols.append(matches[0])
            col_mapping[col.lower()] = len(available_cols) - 1
    
    if len(available_cols) == 0:
        raise ValueError(f"No valid feature columns found. Available: {df.columns.tolist()}")
    
    # Find n_articles index for scaling
    n_articles_idx = col_mapping.get('n_articles')
    
    data = df[available_cols].values.astype(float)
    
    # Handle NaN in sentiment/n_articles
    if 'daily_sentiment' in col_mapping:
        sent_idx = col_mapping['daily_sentiment']
        data[np.isnan(data[:, sent_idx]), sent_idx] = 0
    
    if n_articles_idx is not None:
        data[np.isnan(data[:, n_articles_idx]), n_articles_idx] = 0
    
    X_list, y_list, y_base_list = [], [], []
    
    # Need extra data points for horizon
    max_idx = len(data) - config.seq_len - config.horizon + 1
    
    for i in range(max_idx):
        # Input window
        window = data[i:i + config.seq_len].copy()
        
        # Skip if NaN in price
        if np.isnan(window[:, 0]).any():
            continue
        
        # Base value (first close price)
        base_value = window[0, 0]
        
        # Target: price at horizon steps ahead
        target_idx = i + config.seq_len - 1 + config.horizon
        if target_idx >= len(data):
            continue
        
        target_price = data[target_idx, 0]  # Close price at horizon
        last_price = window[-1, 0]          # Last price in window
        
        # Normalize window
        window_norm = normalize_window_relative(
            window, 
            config.cols_to_norm,
            n_articles_idx,
            config.n_articles_scale
        )
        
        # Calculate target based on task
        if config.task == PredictionTask.PRICE:
            # Normalized price relative to window start
            y_value = (target_price / base_value) - 1
        else: 
            # PredictionTask.DIRECTION
            # Binary: 1 if price went up, 0 if down
            price_change = (target_price - last_price) / last_price
            if abs(price_change) < config.direction_threshold:
                continue  # Skip if change is below threshold
            y_value = 1 if price_change > 0 else 0
            
        
        # X: all timesteps except last (or use full window)
        X_window = window_norm[:-1, :]
        
        X_list.append(X_window)
        y_list.append(y_value)
        y_base_list.append(base_value)
    
    if not X_list:
        raise ValueError("No valid sequences created")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    y_base = np.array(y_base_list, dtype=np.float32)
    
    # Metadata
    metadata = {
        'n_samples': len(X),
        'n_features': X.shape[-1],
        'feature_cols': available_cols,
        'task': config.task.value,
        'horizon': config.horizon
    }
    
    # For direction task, calculate class weights
    if config.task == PredictionTask.DIRECTION:
        n_up = (y == 1).sum()
        n_down = (y == 0).sum()
        total = n_up + n_down
        metadata['class_distribution'] = {'up': int(n_up), 'down': int(n_down)}
        metadata['class_weights'] = {
            0: total / (2 * n_down) if n_down > 0 else 1.0,
            1: total / (2 * n_up) if n_up > 0 else 1.0
        }
        metadata['baseline_accuracy'] = max(n_up, n_down) / total
    
    return X, y, y_base, metadata


def prepare_data(
    csv_file: str,
    config: PreprocessingConfig,
    train_split: float = 0.70,
    val_split: float = 0.85,
    verbose: bool = True
) -> Dict:
    """Complete data preparation pipeline.
    
    Returns
    -------
    dict with keys:
        'train': (X_train, y_train, y_train_base)
        'val': (X_val, y_val, y_val_base)
        'test': (X_test, y_test, y_test_base)
        'metadata': dict with additional info
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Find date column
    date_col = None
    for col in df.columns:
        if col.lower() == 'date':
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    
    # Split by time
    train_end = int(len(df) * train_split)
    val_end = int(len(df) * val_split)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    if verbose:
        print(f"Data split:")
        print(f"  Train: {len(df_train)} days")
        print(f"  Val:   {len(df_val)} days")
        print(f"  Test:  {len(df_test)} days")
        print(f"  Task:  {config.task.value}")
        print(f"  Horizon: {config.horizon} day(s)")
    
    # Create sequences
    X_train, y_train, y_train_base, meta_train = create_sequences(df_train, config)
    X_val, y_val, y_val_base, meta_val = create_sequences(df_val, config)
    X_test, y_test, y_test_base, meta_test = create_sequences(df_test, config)
    
    if verbose:
        print(f"\nSequences created:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        
        if config.task == PredictionTask.DIRECTION:
            print(f"\nClass distribution (train):")
            print(f"  Up:   {meta_train['class_distribution']['up']}")
            print(f"  Down: {meta_train['class_distribution']['down']}")
            print(f"  Baseline accuracy: {meta_train['baseline_accuracy']*100:.1f}%")
    
    return {
        'train': (X_train, y_train, y_train_base),
        'val': (X_val, y_val, y_val_base),
        'test': (X_test, y_test, y_test_base),
        'metadata': {
            'train': meta_train,
            'val': meta_val,
            'test': meta_test,
            'config': config
        }
    }



if __name__ == "__main__":
    print("Preprocessing module loaded.")
    print(f"Available tasks: {[t.value for t in PredictionTask]}")
