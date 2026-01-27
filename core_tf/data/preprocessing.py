"""
Preprocessing of the dataset for LSTM - Relative Normalization
========================================================

Key features from basic preprocessing:
1. Normalize RELATIVE TO FIRST VALUE in each window
2. Target is NORMALIZED PRICE (not return)

"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path


def normalize_window_relative(window: np.ndarray, cols_to_norm: List[int]) -> np.ndarray:
    """Normalize window relative to FIRST value.
    
    Formula: normalized = (value / first_value) - 1
    
    This preserves the relative relationship and trend direction.
    
    Parameters
    ----------
    window : np.ndarray
        Window data, shape (seq_len, n_features).
    cols_to_norm : list of int
        Column indices to normalize (e.g., [0, 1] for close and volume).
    
    Returns
    -------
    normalized_window : np.ndarray
        Normalized window, same shape as input.
    
    Examples
    --------
    >>> window = np.array([[100, 1000], [102, 1050], [105, 900]])
    >>> normalized = normalize_window_relative(window, [0, 1])
    >>> # Column 0: [0, 0.02, 0.05]  (relative to 100)
    >>> # Column 1: [0, 0.05, -0.10] (relative to 1000)
    """
    normalized_window = window.copy()
    
    for col_i in cols_to_norm:
        first_value = window[0, col_i]
        
        # Avoid division by zero
        if first_value == 0:
            first_value = 1
        
        # Normalize: (value / first_value) - 1
        normalized_window[:, col_i] = (window[:, col_i] / first_value) - 1
    
    return normalized_window


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 50,
    feature_cols: List[str] = None,
    cols_to_norm: List[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for  LSTM model.
    
    Key features:
    - Relative normalization to first value
    - Target is NORMALIZED PRICE (not return)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, close, volume, sentiment, etc.
    seq_len : int
        Sequence length (default: 50).
    feature_cols : list of str
        Feature column names (default: ['close', 'volume', 'sentiment']).
    cols_to_norm : list of int
        Column indices to normalize (default: [0, 1] for close and volume).
    normalize : bool
        Whether to normalize data.
    
    Returns
    -------
    X : np.ndarray
        Input sequences, shape (n_samples, seq_len-1, n_features).
        Note: seq_len-1 because last timestep is used as target.
    y : np.ndarray
        Target values (normalized prices), shape (n_samples,).
    y_base : np.ndarray
        Base values for denormalization, shape (n_samples,).
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'close': [100, 102, 105, 103, 108, ...],
    ...     'volume': [1000, 1050, 900, 1100, ...],
    ...     'sentiment': [0.5, 0.6, 0.4, 0.7, ...]
    ... })
    >>> X, y, y_base = create_sequences(df, seq_len=50)
    >>> X.shape  # (n_samples, 49, 3)
    >>> y.shape  # (n_samples,)
    """
    
    # Default feature columns
    if feature_cols is None:
        feature_cols = ['close', 'volume', 'daily_sentiment']
    
    # Default columns to normalize (close and volume, not sentiment)
    if cols_to_norm is None:
        cols_to_norm = [0, 1]  # Normalize close and volume, not sentiment
    
    # Ensure columns exist
    available_cols = []
    for col in feature_cols:
        if col.lower() in df.columns.str.lower():
            # Find exact column name (case-insensitive)
            exact_col = df.columns[df.columns.str.lower() == col.lower()][0]
            available_cols.append(exact_col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame, skipping")
    
    if len(available_cols) == 0:
        raise ValueError(f"No valid feature columns found in DataFrame")
    
    # Extract feature array
    data = df[available_cols].values.astype(float)
    
    X_list, y_list, y_base_list = [], [], []
    
    for i in range(len(data) - seq_len + 1):
        # Get window
        window = data[i:i+seq_len].copy()
        
        # Skip if NaN values
        if np.isnan(window).any():
            continue
        
        # Store base value (first close price) for denormalization
        base_value = window[0, 0]  # First close price
        
        # Normalize window if requested
        if normalize:
            window = normalize_window_relative(window, cols_to_norm)
        
        # Split into input (X) and target (y)
        # X: all timesteps except last (seq_len-1 timesteps)
        # y: normalized price at last timestep
        X_window = window[:-1, :]  # Shape: (seq_len-1, n_features)
        y_value = window[-1, 0]    # Last normalized close price
        
        X_list.append(X_window)
        y_list.append(y_value)
        y_base_list.append(base_value)
    
    if not X_list:
        raise ValueError("No valid sequences created")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    y_base = np.array(y_base_list, dtype=np.float32)
    
    return X, y, y_base


def denormalize_predictions(
    y_normalized: np.ndarray,
    y_base: np.ndarray
) -> np.ndarray:
    """Convert normalized predictions back to actual prices.
    
    Formula: actual_price = (normalized + 1) * base_price
    
    Parameters
    ----------
    y_normalized : np.ndarray
        Normalized predictions, shape (n_samples,).
    y_base : np.ndarray
        Base values (first price in each window), shape (n_samples,).
    
    Returns
    -------
    y_actual : np.ndarray
        Actual price predictions, shape (n_samples,).
    
    Examples
    --------
    >>> y_norm = np.array([0.05, 0.10, -0.02])  # 5%, 10%, -2%
    >>> y_base = np.array([100, 150, 200])
    >>> y_actual = denormalize_predictions(y_norm, y_base)
    >>> # Results: [105, 165, 196]
    """
    return (y_normalized + 1) * y_base


def prepare_data(
    csv_file: str,
    seq_len: int = 50,
    train_split: float = 0.70,
    val_split: float = 0.85,
    feature_cols: List[str] = None,
    cols_to_norm: List[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Complete data preparation pipeline with VALIDATION set.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file with columns: Date, Close, Volume, Scaled_sentiment, etc.
    seq_len : int
        Sequence length (default: 50).
    train_split : float
        Train split ratio (default: 0.70 = 70%).
    val_split : float
        Train + validation split ratio (default: 0.85 = 70% train + 15% val).
    feature_cols : list of str
        Feature columns to use.
    cols_to_norm : list of int
        Column indices to normalize.
    
    Returns
    -------
    X_train : np.ndarray
        Training sequences.
    y_train : np.ndarray
        Training targets (normalized prices).
    y_train_base : np.ndarray
        Base values for train denormalization.
    X_val : np.ndarray
        Validation sequences.
    y_val : np.ndarray
        Validation targets (normalized prices).
    y_val_base : np.ndarray
        Base values for validation denormalization.
    X_test : np.ndarray
        Test sequences.
    y_test : np.ndarray
        Test targets (normalized prices).
    y_test_base : np.ndarray
        Base values for test denormalization.
    """
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split into train/val/test by time
    train_end = int(len(df) * train_split)
    val_end = int(len(df) * val_split)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    if verbose:
        print(f"Data split for ticker {csv_file.split('/')[-1]}:")
        print(f"  Train: {len(df_train):4d} days ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
        print(f"  Val:   {len(df_val):4d} days ({df_val['date'].min().date()} to {df_val['date'].max().date()})")
        print(f"  Test:  {len(df_test):4d} days ({df_test['date'].min().date()} to {df_test['date'].max().date()})")
    
    # Create sequences for train
    X_train, y_train, y_train_base = create_sequences(
        df_train,
        seq_len=seq_len,
        feature_cols=feature_cols,
        cols_to_norm=cols_to_norm,
        normalize=True
    )
    
    # Create sequences for validation
    X_val, y_val, y_val_base = create_sequences(
        df_val,
        seq_len=seq_len,
        feature_cols=feature_cols,
        cols_to_norm=cols_to_norm,
        normalize=True
    )
    
    # Create sequences for test
    X_test, y_test, y_test_base = create_sequences(
        df_test,
        seq_len=seq_len,
        feature_cols=feature_cols,
        cols_to_norm=cols_to_norm,
        normalize=True
    )
    
    if verbose:
        print(f"Sequences created:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")
    
    return (X_train, y_train, y_train_base, 
            X_val, y_val, y_val_base,
            X_test, y_test, y_test_base)



if __name__ == "__main__":
    print("="*60)
    print(" PREPROCESSING - Relative Normalization")
    print("="*60)
    
    # Test with synthetic data
    n_samples = 200
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate realistic price data with trend
    np.random.seed(42)
    base_price = 100
    trend = np.linspace(0, 20, n_samples)
    noise = np.random.randn(n_samples) * 2
    prices = base_price + trend + noise
    
    # Generate volume and sentiment
    volumes = np.random.randint(900000, 1100000, n_samples)
    sentiments = np.random.rand(n_samples) * 2 - 1  # -1 to +1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'close': prices,
        'volume': volumes,
        'daily_sentiment': sentiments
    })
    
    print(f"\nSample data:")
    print(df.head())
    
    # Create sequences
    X, y, y_base = create_sequences(
        df,
        seq_len=50,
        feature_cols=['close', 'volume', 'daily_sentiment'],
        cols_to_norm=[0, 1],
        normalize=True
    )
    
    print(f"\nCreated sequences:")
    print(f"  X shape: {X.shape}")  # (n_samples, 49, 3)
    print(f"  y shape: {y.shape}")  # (n_samples,)
    print(f"  y_base shape: {y_base.shape}")
    
    print(f"\nFirst sequence (first 5 timesteps):")
    print(X[0, :5, :])
    
    print(f"\nTarget for first sequence:")
    print(f"  Normalized: {y[0]:.6f}")
    print(f"  Base price: {y_base[0]:.2f}")
    
    # Denormalize
    y_actual = denormalize_predictions(y[:5], y_base[:5])
    print(f"\nFirst 5 denormalized predictions:")
    for i in range(5):
        print(f"  Sample {i}: norm={y[i]:.6f}, base={y_base[i]:.2f}, actual={y_actual[i]:.2f}")