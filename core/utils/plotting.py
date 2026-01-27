"""Plotting utilities for visualization."""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
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
    figsize: Tuple[int, int] = (12, 6),
    target_type: str = "return",
    show_sign_markers: bool = False,
    show_error_hist: bool = False,
) -> None:
    """Plot true vs predicted values for a given ticker.

    Adds optional sign markers showing where predictions correctly or
    incorrectly predict the sign, and optionally a histogram of prediction errors.

    Args:
        ticker: Ticker symbol
        dates: List of dates
        true_values: List of true values
        predicted_series: Optional predictions (dict or list)
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
        target_type: 'return' or 'price' to determine plot style
        show_sign_markers: If True and target_type is 'return', plot sign-correct/incorrect markers
        show_error_hist: If True, include an error histogram as a second subplot
    """
    # Choose layout depending on whether we show histogram
    if show_error_hist:
        fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.6), gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_hist = None
    
    dates_dt = pd.to_datetime(dates)

    # Helper aggregator for error histogram
    errors = None

    if target_type == "price" or target_type == "close":
        ax.plot(dates_dt, true_values, label='True Price', color='black', linewidth=2, alpha=0.8)

        if predicted_series is not None:
            if isinstance(predicted_series, dict):
                colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_series)))
                for (name, series), color in zip(predicted_series.items(), colors):
                    ax.plot(dates_dt, series, label=name, color=color, linewidth=1.5, alpha=0.7)
                    if isinstance(series, (list, np.ndarray)):
                        errors = np.asarray(series) - np.asarray(true_values)
            else:
                ax.plot(dates_dt, predicted_series, label='Predicted Price', color='tab:orange', linewidth=1.5, alpha=0.7)
                errors = np.asarray(predicted_series) - np.asarray(true_values)

        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} - Price Prediction')

    else:
        ax.plot(dates_dt, true_values, label='True Returns', color='black', linewidth=1.5, alpha=0.8, marker='o', markersize=2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Draw predictions
        if predicted_series is not None:
            if isinstance(predicted_series, dict):
                colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_series)))
                for (name, series), color in zip(predicted_series.items(), colors):
                    ax.plot(dates_dt, series, label=name, color=color, linewidth=1.5, alpha=0.7, marker='s', markersize=2)
                    if isinstance(series, (list, np.ndarray)):
                        errors = np.asarray(series) - np.asarray(true_values)
            else:
                ax.plot(dates_dt, predicted_series, label='Predicted Returns', color='tab:orange', linewidth=1.5, alpha=0.7, marker='s', markersize=2)
                errors = np.asarray(predicted_series) - np.asarray(true_values)

        ax.set_ylabel('Returns')
        ax.set_title(f'{ticker} - Return Prediction')

        # Optional sign markers
        if show_sign_markers and predicted_series is not None:
            # Use the primary predicted_series if dict provided, else the series
            prim_pred = None
            if isinstance(predicted_series, dict):
                # pick first series as representative
                prim_pred = next(iter(predicted_series.values()))
            else:
                prim_pred = predicted_series

            if prim_pred is not None:
                prim_pred = np.asarray(prim_pred)
                tv = np.asarray(true_values)
                correct = np.sign(prim_pred) == np.sign(tv)

                # Plot green markers for correct and red for incorrect
                ax.scatter(dates_dt[correct], tv[correct], marker='o', color='green', s=20, label='Correct Sign')
                ax.scatter(dates_dt[~correct], tv[~correct], marker='x', color='red', s=20, label='Wrong Sign')

    # Error histogram
    if ax_hist is not None and errors is not None:
        ax_hist.hist(errors[~np.isnan(errors)], bins=40, color='gray', alpha=0.8)
        ax_hist.set_xlabel('Prediction Error')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Error Distribution')
        ax_hist.grid(True, alpha=0.3)

    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    _handle_figure_output(fig, save_path, show)


def plot_rolling_predictions(
    ticker: str,
    fold_data: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (16, 8),
    target_type: str = "return"
) -> None:
    """Plot predictions across all rolling folds showing the full timeline.
    
    Args:
        ticker: Ticker symbol
        fold_data: List of dicts with keys: 'dates', 'true_values', 'predictions', 
                   'split' ('train', 'val', 'test')
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
        target_type: 'return' or 'price' to determine plot style
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    split_colors = {
        'train': 'lightblue',
        'val': 'lightcoral',
        'test': 'lightgreen'
    }
    
    all_dates = []
    all_true = []
    all_pred = []
    
    for fold_idx, fold in enumerate(fold_data):
        dates_dt = pd.to_datetime(fold['dates'])
        true_vals = fold['true_values']
        preds = fold['predictions']
        split = fold.get('split', 'val')
        
        all_dates.extend(dates_dt)
        all_true.extend(true_vals)
        all_pred.extend(preds)
        
        color = split_colors.get(split, 'lightgray')
        
        if fold_idx == 0 or fold.get('split') != fold_data[fold_idx-1].get('split'):
            label = f'{split.capitalize()}'
            ax.axvspan(dates_dt.min(), dates_dt.max(), alpha=0.2, color=color, label=label)
        else:
            ax.axvspan(dates_dt.min(), dates_dt.max(), alpha=0.2, color=color)
    
    all_dates_sorted = sorted(set(all_dates))
    
    if target_type == "price" or target_type == "close":
        date_to_true = dict(zip(all_dates, all_true))
        date_to_pred = dict(zip(all_dates, all_pred))
        
        sorted_true = [date_to_true.get(d, np.nan) for d in all_dates_sorted]
        sorted_pred = [date_to_pred.get(d, np.nan) for d in all_dates_sorted]
        
        ax.plot(all_dates_sorted, sorted_true, label='True Price', color='black', linewidth=2, alpha=0.9, zorder=3)
        ax.plot(all_dates_sorted, sorted_pred, label='Predicted Price', color='tab:orange', linewidth=1.5, alpha=0.8, zorder=2)
        
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} - Rolling Window Price Predictions')
        
    else:
        date_to_true = dict(zip(all_dates, all_true))
        date_to_pred = dict(zip(all_dates, all_pred))
        
        sorted_true = [date_to_true.get(d, np.nan) for d in all_dates_sorted]
        sorted_pred = [date_to_pred.get(d, np.nan) for d in all_dates_sorted]
        
        ax.plot(all_dates_sorted, sorted_true, label='True Returns', color='black', linewidth=1.5, alpha=0.9, marker='o', markersize=3, zorder=3)
        ax.plot(all_dates_sorted, sorted_pred, label='Predicted Returns', color='tab:orange', linewidth=1.5, alpha=0.8, marker='s', markersize=3, zorder=2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
        ax.set_ylabel('Returns')
        ax.set_title(f'{ticker} - Rolling Window Return Predictions')
    
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    _handle_figure_output(fig, save_path, show)


def plot_fold_comparison(
    fold_metrics: List[Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """Plot metrics comparison across rolling folds.
    
    Args:
        fold_metrics: List of metric dictionaries for each fold
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
    """
    if not fold_metrics:
        print("No fold metrics to plot")
        return
    
    metrics_to_plot = ['mse', 'mae', 'rmse', 'dir_acc', 'r2', 'sharpe_pred']
    available_metrics = [m for m in metrics_to_plot if m in fold_metrics[0]]
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    fold_ids = list(range(len(fold_metrics)))
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = [fold.get(metric, np.nan) for fold in fold_metrics]
        
        ax.plot(fold_ids, values, marker='o', linewidth=2, markersize=8, color='tab:blue')
        ax.axhline(y=np.nanmean(values), color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {np.nanmean(values):.4f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper().replace('_', ' '))
        ax.set_title(f'{metric.upper().replace("_", " ")} Across Folds')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(fold_ids)
    
    for idx in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[idx])
    
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