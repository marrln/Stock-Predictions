"""Module for plotting stock data and model predictions."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from torchsummary import summary
import torch

def plot_training_history(history: Dict[str, list], save_path: Optional[Path] = None) -> None:
    """Plot training history metrics."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    
    # MAE plot
    axes[0, 1].plot(history['val_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    
    # RMSE plot
    axes[1, 0].plot(history['val_rmse'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE')
    
    # Learning rate plot
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_model_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[Path] = None) -> None:
    """Persist model comparison results to a JSON file."""
    import json
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(results, fh, indent=2)


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[Path] = None) -> None:
    """Create bar plots comparing models across key metrics.

    `results` should be a mapping from model name to a dict containing
    keys: 'mse', 'mae', 'rmse', 'dir_acc' (directional accuracy).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results.keys())
    mse = [results[n]["mse"] for n in names]
    mae = [results[n]["mae"] for n in names]
    rmse = [results[n]["rmse"] for n in names]
    diracc = [results[n].get("dir_acc", 0.0) for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].bar(x - 1.5 * width, mse, width)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_title('MSE')

    axes[1].bar(x - 0.5 * width, mae, width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_title('MAE')

    axes[2].bar(x + 0.5 * width, rmse, width)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_title('RMSE')

    axes[3].bar(x + 1.5 * width, diracc, width)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, rotation=45, ha='right')
    axes[3].set_title('Directional Acc')

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def get_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Get a string summary of the model architecture and parameters."""
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        # Create a dummy input
        dummy_input = torch.randn(1, *input_shape)
        try:
            summary(model, input_shape, device="cpu")
        except ImportError:
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    return f.getvalue()


def plot_ticker_performance(ticker: str, dates: list, true_prices: list, predicted_series: Optional[dict] = None, save_path: Optional[Path] = None) -> None:
    """Plot true vs predicted values for a given ticker.

    `predicted_series` is an optional dict mapping label -> list of predicted values
    aligned with `dates`. If a single list is supplied, it will be plotted as
    `Predicted`.
    """
    import matplotlib.pyplot as plt
    import itertools

    plt.figure(figsize=(10, 6))
    plt.plot(dates, true_prices, label='True', color='black', linewidth=1.5)

    if predicted_series is not None:
        # allow passing a single list accidentally
        if not isinstance(predicted_series, dict):
            predicted_series = {"Predicted": list(predicted_series)}

        color_cycle = itertools.cycle(['orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'brown'])
        for name, series in predicted_series.items():
            plt.plot(dates, series, label=name, color=next(color_cycle), alpha=0.9)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'True vs Predictions for {ticker}')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()