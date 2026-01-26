"""Plotting utilities for visualization."""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plot training history metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss plot
    axes[0, 0].plot(history.get('train_loss', []), label='Train', marker='o', markersize=3)
    axes[0, 0].plot(history.get('val_loss', []), label='Validation', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history.get('val_mae', []), color='tab:orange', marker='^', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE plot
    axes[1, 0].plot(history.get('val_rmse', []), color='tab:green', marker='d', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 1].plot(history.get('lr', []), color='tab:red', marker='v', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    _handle_figure_output(fig, save_path, show)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    """Create bar plots comparing models across key metrics.
    
    Args:
        results: Mapping from model name to metrics dictionary
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
    """
    if not results:
        print("No results to plot")
        return
    
    names = list(results.keys())
    metrics_to_plot = ['mse', 'mae', 'rmse', 'dir_acc']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[n].get(metric, 0.0) for n in names]
        
        bars = axes[idx].bar(names, values, color=plt.cm.Set3(np.arange(len(names))))
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric.upper() if metric != 'dir_acc' else 'Directional Accuracy')
        axes[idx].set_title(metric.upper() if metric != 'dir_acc' else 'Directional Accuracy')
        axes[idx].set_xticks(np.arange(len(names)))
        axes[idx].set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2., 
                height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8
            )
    
    plt.tight_layout()
    _handle_figure_output(fig, save_path, show)


def plot_ticker_performance(
    ticker: str,
    dates: List,
    true_values: List,
    predicted_series: Optional[Union[Dict[str, List], List]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot true vs predicted values for a given ticker.
    
    Args:
        ticker: Ticker symbol
        dates: List of dates
        true_values: List of true values
        predicted_series: Optional predictions (dict or list)
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot true values
    ax.plot(dates, true_values, label='True', color='black', linewidth=2, alpha=0.8)
    
    # Plot predictions if provided
    if predicted_series is not None:
        if isinstance(predicted_series, dict):
            colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_series)))
            for (name, series), color in zip(predicted_series.items(), colors):
                ax.plot(dates, series, label=name, color=color, linewidth=1.5, alpha=0.7)
        else:
            ax.plot(dates, predicted_series, label='Predicted', color='tab:orange', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'True vs Predictions for {ticker}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    _handle_figure_output(fig, save_path, show)


def get_model_summary(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu"
) -> str:
    """Get a string summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        device: Device to run summary on
        
    Returns:
        String containing model summary
    """
    buffer = io.StringIO()
    
    with redirect_stdout(buffer):
        try:
            # Try to use torchsummary
            summary(model, input_shape, device=device)
        except (ImportError, Exception):
            # Fallback to simple summary
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    return buffer.getvalue()


def save_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Path
) -> None:
    """Save model comparison results to a JSON file.
    
    Args:
        results: Model comparison results
        save_path: Path to save the JSON file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _handle_figure_output(
    fig: plt.Figure,
    save_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """Handle figure output (save, show, or close)."""
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)