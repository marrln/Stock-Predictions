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


def reconstruct_prices_from_pct_change(
    pct_changes: np.ndarray,
    dates: List[str],
    ticker: str,
    price_history_path: str = "Stock_price/full_history"
) -> np.ndarray:
    """Reconstruct price series from percent changes.
    
    Args:
        pct_changes: Array of percent changes (e.g., 2.5 means 2.5% increase)
        dates: List of date strings corresponding to pct_changes
        ticker: Ticker symbol
        price_history_path: Path to directory containing price CSVs
        
    Returns:
        Reconstructed price array
    """
    from pathlib import Path
    
    # Load historical prices
    csv_path = Path(price_history_path) / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"Warning: Cannot reconstruct prices - {csv_path} not found")
        return pct_changes  # Return pct_changes as fallback
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Get the starting price (price at day before first prediction)
    first_date = pd.to_datetime(dates[0])
    mask = df['Date'] < first_date
    
    if mask.sum() == 0:
        print(f"Warning: No historical price before {first_date}")
        return pct_changes
    
    # Use the last available price before prediction period as starting point
    start_price = df.loc[mask, 'Close'].iloc[-1]
    
    # Reconstruct prices: price[t] = price[t-1] * (1 + pct_change[t]/100)
    prices = np.zeros(len(pct_changes))
    prices[0] = start_price * (1 + pct_changes[0] / 100.0)
    
    for i in range(1, len(pct_changes)):
        prices[i] = prices[i-1] * (1 + pct_changes[i] / 100.0)
    
    return prices


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

    elif target_type == "pct_change":
        # Plot percent change similar to returns but with % scale
        ax.plot(dates_dt, true_values, label='True % Change', color='black', linewidth=2, alpha=0.8)

        if predicted_series is not None:
            if isinstance(predicted_series, dict):
                colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_series)))
                for (name, series), color in zip(predicted_series.items(), colors):
                    ax.plot(dates_dt, series, label=name, color=color, linewidth=1.5, alpha=0.7)
                    if isinstance(series, (list, np.ndarray)):
                        errors = np.asarray(series) - np.asarray(true_values)
            else:
                ax.plot(dates_dt, predicted_series, label='Predicted % Change', color='tab:orange', linewidth=1.5, alpha=0.7)
                errors = np.asarray(predicted_series) - np.asarray(true_values)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Percent Change (%)')
        ax.set_title(f'{ticker} - Daily % Change Prediction')
        
        # Add sign markers if requested
        if show_sign_markers and predicted_series is not None:
            if not isinstance(predicted_series, dict):
                pred_arr = np.asarray(predicted_series)
                true_arr = np.asarray(true_values)
                sign_correct = np.sign(pred_arr) == np.sign(true_arr)
                
                correct_dates = [d for d, c in zip(dates_dt, sign_correct) if c]
                correct_vals = [p for p, c in zip(pred_arr, sign_correct) if c]
                incorrect_dates = [d for d, c in zip(dates_dt, sign_correct) if not c]
                incorrect_vals = [p for p, c in zip(pred_arr, sign_correct) if not c]
                
                ax.scatter(correct_dates, correct_vals, color='green', s=15, alpha=0.5, label='Sign Correct', zorder=5)
                ax.scatter(incorrect_dates, incorrect_vals, color='red', s=15, alpha=0.5, label='Sign Incorrect', zorder=5)

    else:
        # Handle log_return vs regular return labeling
        is_log_return = target_type == "log_return"
        label_suffix = "Log Returns" if is_log_return else "Returns"
        
        ax.plot(dates_dt, true_values, label=f'True {label_suffix}', color='black', linewidth=1.5, alpha=0.8, marker='o', markersize=2)
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
                ax.plot(dates_dt, predicted_series, label=f'Predicted {label_suffix}', color='tab:orange', linewidth=1.5, alpha=0.7, marker='s', markersize=2)
                errors = np.asarray(predicted_series) - np.asarray(true_values)

        ax.set_ylabel('Log Returns' if is_log_return else 'Returns')
        ax.set_title(f'{ticker} - {"Log Return" if is_log_return else "Return"} Prediction')

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
    figsize: Tuple[int, int] = (16, 10),
    target_type: str = "return"
) -> None:
    """Plot predictions across all rolling folds showing the full timeline.
    
    Each fold gets its own subplot, stacked vertically, showing train/val/test splits.
    Shows full historical true values as context on all subplots.
    
    Args:
        ticker: Ticker symbol
        fold_data: List of dicts with keys: 'dates', 'true_values', 'predictions', 
                   'split' ('train', 'val', 'test'), 'fold_idx'
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
        target_type: 'return' or 'price' to determine plot style
    """
    # Define colors for different splits
    split_colors = {
        'train': '#E3F2FD',  # Light blue
        'val': '#FFF3E0',     # Light orange
        'test': '#E8F5E9'     # Light green
    }
    
    # Group data by fold
    folds_dict = {}
    for data in fold_data:
        fold_idx = data['fold_idx']
        if fold_idx not in folds_dict:
            folds_dict[fold_idx] = []
        folds_dict[fold_idx].append(data)
    
    n_folds = len(folds_dict)
    if n_folds == 0:
        return
    
    # Collect ALL historical true values across all folds for context
    all_hist_dates = []
    all_hist_true = []
    for data in fold_data:
        all_hist_dates.extend(pd.to_datetime(data['dates']))
        all_hist_true.extend(data['true_values'])
    
    # Create a dict for quick lookup of true values by date
    hist_date_to_true = {}
    for d, v in zip(all_hist_dates, all_hist_true):
        hist_date_to_true[d] = v
    
    all_hist_dates_sorted = sorted(set(all_hist_dates))
    hist_true_sorted = [hist_date_to_true[d] for d in all_hist_dates_sorted]
    
    # Create stacked subplots - one per fold
    fig, axes = plt.subplots(n_folds, 1, figsize=(figsize[0], figsize[1] * n_folds / 4), 
                             sharex=True, sharey=True)
    
    # Handle single fold case
    if n_folds == 1:
        axes = [axes]
    
    # Determine label suffix based on target type
    is_log_return = target_type == "log_return"
    label_suffix = "Log Returns" if is_log_return else "Returns"
    ylabel = 'Log Returns' if is_log_return else 'Returns'
    if target_type == "price" or target_type == "close":
        ylabel = 'Price ($)'
        label_suffix = "Price"
    elif target_type == "pct_change":
        ylabel = 'Percent Change (%)'
        label_suffix = "% Change"
    
    split_labels_shown = set()
    
    # Plot each fold in its own subplot
    for ax_idx, fold_idx in enumerate(sorted(folds_dict.keys())):
        ax = axes[ax_idx]
        fold_splits = folds_dict[fold_idx]
        
        # Sort splits by time order: train, val, test
        split_order = {'train': 0, 'val': 1, 'test': 2}
        fold_splits.sort(key=lambda x: split_order.get(x['split'], 99))
        
        # FIRST: Plot the full historical true values as gray background on ALL folds
        # This shows the complete market context even for dates this fold doesn't predict
        ax.plot(all_hist_dates_sorted, hist_true_sorted, 
               color='lightgray', linewidth=0.6, alpha=0.5, zorder=1, 
               label='Full Historical Context' if fold_idx == 0 else None)
        
        # Collect dates for this fold's data
        fold_dates = []
        fold_true = []
        fold_pred = []
        
        # Draw background regions and collect data for each split
        for data in fold_splits:
            dates_dt = pd.to_datetime(data['dates'])
            split_name = data['split']
            
            fold_dates.extend(dates_dt)
            fold_true.extend(data['true_values'])
            fold_pred.extend(data['predictions'])
            
            # Background shading for split type
            color = split_colors.get(split_name, '#F5F5F5')
            label = None
            if split_name not in split_labels_shown:
                label = f'{split_name.capitalize()} Set'
                split_labels_shown.add(split_name)
            
            ax.axvspan(dates_dt.min(), dates_dt.max(), alpha=0.3, color=color, label=label)
            
            # Add vertical separator between splits
            if split_name != 'train':
                ax.axvline(x=dates_dt.min(), color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Sort data by date for plotting THIS FOLD'S data
        fold_dates_sorted = sorted(set(fold_dates))
        date_to_true = dict(zip(fold_dates, fold_true))
        date_to_pred = dict(zip(fold_dates, fold_pred))
        
        sorted_true = [date_to_true.get(d, np.nan) for d in fold_dates_sorted]
        sorted_pred = [date_to_pred.get(d, np.nan) for d in fold_dates_sorted]
        
        # Plot THIS fold's true values and predictions (on top of historical background)
        if target_type == "price" or target_type == "close":
            ax.plot(fold_dates_sorted, sorted_true, label='True Price (This Fold)' if fold_idx == 0 else None, 
                   color='black', linewidth=1.0, alpha=0.9, zorder=3)
            ax.plot(fold_dates_sorted, sorted_pred, label='Predicted Price' if fold_idx == 0 else None, 
                   color='tab:orange', linewidth=0.8, alpha=0.8, zorder=2)
        elif target_type == "pct_change":
            ax.plot(fold_dates_sorted, sorted_true, label=f'True {label_suffix} (This Fold)' if fold_idx == 0 else None, 
                   color='black', linewidth=1.0, alpha=0.9, zorder=3)
            ax.plot(fold_dates_sorted, sorted_pred, label=f'Predicted {label_suffix}' if fold_idx == 0 else None, 
                   color='tab:orange', linewidth=0.8, alpha=0.8, zorder=2)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        else:
            # Returns or log returns - thinner lines, no markers
            ax.plot(fold_dates_sorted, sorted_true, label=f'True {label_suffix} (This Fold)' if fold_idx == 0 else None, 
                   color='black', linewidth=0.8, alpha=0.9, zorder=3)
            ax.plot(fold_dates_sorted, sorted_pred, label=f'Predicted {label_suffix}' if fold_idx == 0 else None, 
                   color='tab:orange', linewidth=0.8, alpha=0.8, zorder=2)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Formatting
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Set x-axis limits to show full timeline on all subplots
        ax.set_xlim(all_hist_dates_sorted[0], all_hist_dates_sorted[-1])
        
        # Add fold label
        ax.text(0.02, 0.95, f'Fold {fold_idx}', transform=ax.transAxes,
               verticalalignment='top', fontsize=12, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Only show legend on first subplot
        if fold_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Set x-label and rotate ticks on bottom subplot only (sharex handles the rest)
    axes[-1].set_xlabel('Date')
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall title
    title_suffix = "Log Return" if is_log_return else "Return"
    if target_type == "price" or target_type == "close":
        title_suffix = "Price"
    elif target_type == "pct_change":
        title_suffix = "% Change"
    
    fig.suptitle(f'{ticker} - Rolling Fold {title_suffix} Predictions', fontsize=14, weight='bold', y=0.995)
    
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


def plot_predictions_with_price_reconstruction(
    dates: List[str],
    true_pct_changes: np.ndarray,
    pred_pct_changes: np.ndarray,
    ticker: str,
    price_history_path: str = "Stock_price/full_history",
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Plot both percent changes and reconstructed prices side by side.
    
    Args:
        dates: List of date strings
        true_pct_changes: True percent changes
        pred_pct_changes: Predicted percent changes
        ticker: Ticker symbol
        price_history_path: Path to historical price data
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    dates_dt = pd.to_datetime(dates)
    
    # Top plot: Percent changes
    ax1.plot(dates_dt, true_pct_changes, label='True % Change', 
             color='black', linewidth=2, alpha=0.8)
    ax1.plot(dates_dt, pred_pct_changes, label='Predicted % Change', 
             color='tab:orange', linewidth=1.5, alpha=0.7)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Percent Change (%)')
    ax1.set_title(f'{ticker} - Daily % Change Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Reconstructed prices
    true_prices = reconstruct_prices_from_pct_change(
        true_pct_changes, dates, ticker, price_history_path
    )
    pred_prices = reconstruct_prices_from_pct_change(
        pred_pct_changes, dates, ticker, price_history_path
    )
    
    ax2.plot(dates_dt, true_prices, label='True Price (reconstructed)', 
             color='black', linewidth=2, alpha=0.8)
    ax2.plot(dates_dt, pred_prices, label='Predicted Price (reconstructed)', 
             color='tab:orange', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Date')
    ax2.set_title(f'{ticker} - Reconstructed Price from % Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    _handle_figure_output(fig, save_path, show)


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