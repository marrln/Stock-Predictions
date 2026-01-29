''' Training and evaluation functions for financial time series models. '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from pathlib import Path
from typing import List, Dict

from .metrics import (
    evaluate_price_predictions, 
    evaluate_direction_predictions, 
    direction_baseline_majority,
)
from core_tf.data import PreprocessingConfig, prepare_data
from core_tf.models import ModelConfig, FinancialModel



def load_multi_ticker_data(
    data_dir: Path,
    tickers: List[str],
    config: PreprocessingConfig,
    verbose: bool = True
) -> Dict:
    """Load and combine data from multiple tickers."""
    
    X_train_list, y_train_list, y_base_train_list = [], [], []
    X_val_list, y_val_list, y_base_val_list = [], [], []
    X_test_list, y_test_list, y_base_test_list = [], [], []
    
    metadata_list = []
    
    for ticker in tickers:
        csv_path = data_dir / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {ticker}")
            continue
        
        if verbose:
            print(f"\nLoading {ticker}...")
        
        data = prepare_data(str(csv_path), config, verbose=verbose)
        
        X_train, y_train, y_base_train = data['train']
        X_val, y_val, y_base_val = data['val']
        X_test, y_test, y_base_test = data['test']
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        y_base_train_list.append(y_base_train)
        
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        y_base_val_list.append(y_base_val)
        
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        y_base_test_list.append(y_base_test)
        
        metadata_list.append(data['metadata'])
    
    if not X_train_list:
        raise ValueError("No valid data loaded!")
    
    return {
        'train': (
            np.concatenate(X_train_list),
            np.concatenate(y_train_list),
            np.concatenate(y_base_train_list)
        ),
        'val': (
            np.concatenate(X_val_list),
            np.concatenate(y_val_list),
            np.concatenate(y_base_val_list)
        ),
        'test': (
            np.concatenate(X_test_list),
            np.concatenate(y_test_list),
            np.concatenate(y_base_test_list)
        ),
        'metadata': metadata_list[0]  # Use first ticker's metadata
    }


def train_and_evaluate(
    data: Dict,
    model_config: ModelConfig,
    task: str,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 15,
    verbose: int = 1
) -> Dict:
    """Train model and return evaluation results."""
    
    X_train, y_train, y_base_train = data['train']
    X_val, y_val, y_base_val = data['val']
    X_test, y_test, y_base_test = data['test']
    
    # Create model
    model = FinancialModel(model_config)
    
    if verbose:
        model.summary()
    
    # Get class weights for direction task
    class_weight = None
    if task == 'direction':
        n_up = (y_train == 1).sum()
        n_down = (y_train == 0).sum()
        total = n_up + n_down
        class_weight = {
            0: total / (2 * n_down),
            1: total / (2 * n_up)
        }
        if verbose:
            print(f"\nClass weights: {class_weight}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        patience_es=patience,
        verbose=verbose
    )
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    if task == 'direction':
        results = evaluate_direction_predictions(y_test, y_pred)
        
        # Add baseline comparison
        baseline_acc = direction_baseline_majority(y_train) * 100
        results['Baseline_Accuracy'] = baseline_acc
        results['vs_Baseline'] = results['Accuracy'] - baseline_acc
        
    else:  # price or return
        results = evaluate_price_predictions(y_test, y_pred, y_base_test, X_test)
    
    results['epochs_trained'] = len(history.history['loss'])
    
    return results, model, y_pred