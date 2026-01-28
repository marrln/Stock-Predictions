"""Model training and evaluation functions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict

import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from ..checkpoint import save_checkpoint
from .metrics import compute_regression_metrics


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack a batch and validate it follows the (X, y) or (X, y, ticker_idx) format."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            xb, yb, ticker_idx = batch
        elif len(batch) == 2:
            xb, yb = batch
            ticker_idx = None
        else:
            raise ValueError(f"Batch must contain 2 or 3 elements, got {len(batch)}")
    else:
        raise TypeError(f"Expected batch as tuple/list, got {type(batch)}")
    
    # Validate tensors
    if not torch.is_tensor(xb) or not torch.is_tensor(yb):
        raise TypeError("Both X and y must be torch.Tensor")
    
    if xb.ndim != 3:
        raise ValueError(f"X must be 3D tensor (batch, seq_len, features), got {xb.shape}")
    
    if yb.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D tensor, got {yb.shape}")
    
    # Ensure batch sizes match
    if yb.shape[0] != xb.shape[0]:
        raise ValueError(f"Batch size mismatch: X {xb.shape[0]}, y {yb.shape[0]}")
    
    # Squeeze y if shape (batch, 1)
    if yb.ndim == 2 and yb.shape[1] == 1:
        yb = yb.squeeze(1)
    
    return xb, yb, ticker_idx


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
    progress_bar: bool = False,
) -> Tuple[float, float]:
    """Run one epoch of training.
    
    Returns:
        Tuple of (average_loss, average_gradient_norm)
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    grad_norms: List[float] = []
    
    # Optional progress bar
    if progress_bar:
        from tqdm import tqdm
        loader = tqdm(loader, desc="Training", leave=False)
    
    for batch in loader:
        xb, yb, ticker_idx = unpack_batch(batch)
        
        # Move to device
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if ticker_idx is not None:
            ticker_idx = ticker_idx.to(device)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb, ticker_idx) if ticker_idx is not None else model(xb)
        loss = loss_fn(preds, yb)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        grad_norm = math.sqrt(total_norm)
        grad_norms.append(grad_norm)
        
        # Gradient clipping
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimization step
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        n_batches += 1
    
    avg_loss = running_loss / max(n_batches, 1)
    avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
    
    return avg_loss, avg_grad_norm


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on a loader.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            xb, yb, ticker_idx = unpack_batch(batch)
            
            xb = xb.to(device)
            yb = yb.to(device)
            if ticker_idx is not None:
                ticker_idx = ticker_idx.to(device)
            
            # Forward pass
            preds = model(xb, ticker_idx) if ticker_idx is not None else model(xb)
            
            # Store predictions and targets
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
            
            # Compute loss
            batch_loss = loss_fn(preds, yb)
            total_loss += batch_loss.item() * xb.size(0)
            total_samples += xb.size(0)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_regression_metrics(all_preds, all_targets, include_r2=True, include_sharpe=True)
    
    return avg_loss, metrics


def compute_unscaled_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_scaler: Any,
) -> Dict[str, float]:
    """Compute metrics on the original (unscaled) target values.
    
    Supports either a single scaler or a dict mapping ticker->scaler.
    """
    model.eval()
    
    # Collect all predictions and targets
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_tickers: List[Optional[str]] = []
    
    # Create ticker index to symbol mapping if available
    ticker_idx_to_symbol = None
    if hasattr(loader.dataset, 'ticker_to_idx'):
        ticker_idx_to_symbol = {v: k for k, v in loader.dataset.ticker_to_idx.items()}
    
    with torch.no_grad():
        for batch in loader:
            xb, yb, ticker_idx = unpack_batch(batch)
            
            xb = xb.to(device)
            if ticker_idx is not None:
                preds = model(xb, ticker_idx.to(device))
            else:
                preds = model(xb)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.numpy())
            
            # Map ticker indices to symbols
            if ticker_idx is not None and ticker_idx_to_symbol is not None:
                batch_tickers = [
                    ticker_idx_to_symbol.get(int(idx), None) 
                    for idx in ticker_idx.cpu().numpy()
                ]
                all_tickers.extend(batch_tickers)
            else:
                all_tickers.extend([None] * len(preds))
    
    # Concatenate results
    preds = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets = np.concatenate(all_targets, axis=0).reshape(-1, 1)
    
    # If a dictionary of scalers is provided we require ticker indices to be present in the batch mapping
    if isinstance(y_scaler, dict):
        if any(t is None for t in all_tickers):
            raise RuntimeError(
                "Per-ticker y_scaler provided but loader batches did not include ticker indices for all samples. "
                "Ensure your DataLoader returns (X, y, ticker_idx) (e.g., use a collate that includes TickerIdx) or that target_scaling uses a single scaler."
            )

        # Group indices by ticker for batch inverse transformation
        from collections import defaultdict
        idx_by_ticker = defaultdict(list)
        for i, t in enumerate(all_tickers):
            idx_by_ticker[t].append(i)

        inv_preds = np.empty((len(preds),), dtype=float)
        inv_targets = np.empty((len(targets),), dtype=float)
        
        for ticker, indices in idx_by_ticker.items():
            scaler = y_scaler.get(ticker)
            
            if scaler is None:
                # No scaler for this ticker, use raw values
                for idx in indices:
                    inv_preds[idx] = preds[idx, 0]
                    inv_targets[idx] = targets[idx, 0]
                continue
            
            # Apply inverse transform
            try:
                batch_preds = preds[indices]
                batch_targets = targets[indices]
                
                inv_batch_preds = scaler.inverse_transform(batch_preds).ravel()
                inv_batch_targets = scaler.inverse_transform(batch_targets).ravel()
                
                for i, idx in enumerate(indices):
                    inv_preds[idx] = inv_batch_preds[i]
                    inv_targets[idx] = inv_batch_targets[i]
            except Exception as e:
                print(f"Warning: Failed to inverse transform for ticker {ticker}: {e}")
                for idx in indices:
                    inv_preds[idx] = preds[idx, 0]
                    inv_targets[idx] = targets[idx, 0]
    else:
        # Single scaler for all data
        try:
            inv_preds = y_scaler.inverse_transform(preds).ravel()
            inv_targets = y_scaler.inverse_transform(targets).ravel()
        except Exception as e:
            print(f"Warning: Failed to inverse transform: {e}")
            inv_preds = preds.ravel()
            inv_targets = targets.ravel()
    
    # Compute metrics on unscaled values
    ds = loader.dataset if hasattr(loader, 'dataset') else None
    target_type = getattr(ds, 'target_type', 'return') if ds is not None else 'return'

    # Handle 'close' (price) targets: directional accuracy and Sharpe should be computed on returns
    if target_type == 'close' and ds is not None and ds.meta is not None and 'last_close' in ds.meta.columns:
        last_close = ds.meta['last_close'].astype(float).values
        # Avoid division by zero
        safe_mask = (last_close != 0) & ~np.isnan(last_close)

        # Compute basic metrics on prices (MAE/RMSE), but compute directional and sharpe on returns
        metrics = compute_regression_metrics(inv_preds, inv_targets, include_r2=True, include_sharpe=False, include_directional=False)

        # Prepare returns for directional and sharpe calculations
        pred_returns = np.empty_like(inv_preds, dtype=float)
        true_returns = np.empty_like(inv_targets, dtype=float)
        pred_returns[~safe_mask] = 0.0
        true_returns[~safe_mask] = 0.0
        pred_returns[safe_mask] = (inv_preds[safe_mask] - last_close[safe_mask]) / last_close[safe_mask]
        true_returns[safe_mask] = (inv_targets[safe_mask] - last_close[safe_mask]) / last_close[safe_mask]

        # Directional accuracy computed on returns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dir_acc_returns = float(np.mean(np.sign(pred_returns) == np.sign(true_returns)))
        metrics['dir_acc'] = dir_acc_returns

        # Sharpe-like ratios on returns
        def _sharpe(arr: np.ndarray) -> float:
            std = np.std(arr)
            if std == 0 or np.isnan(std):
                return 0.0
            return float(np.mean(arr) / std)

        metrics['sharpe_pred'] = _sharpe(pred_returns)
        metrics['sharpe_true'] = _sharpe(true_returns)

        # Per-ticker breakdown on prices + directional metrics computed on returns
        ticker_array = np.array(all_tickers) if any(t is not None for t in all_tickers) else None
        if ticker_array is not None:
            per_ticker = {}
            for t in np.unique(ticker_array):
                mask = ticker_array == t
                if mask.sum() == 0:
                    continue

                # Price-based metrics
                sub_metrics = compute_regression_metrics(
                    inv_preds[mask], inv_targets[mask],
                    include_directional=False, include_r2=True, include_sharpe=False
                )

                # Returns-based directional + sharpe
                lc = last_close[mask]
                safe = (lc != 0) & ~np.isnan(lc)
                if safe.any():
                    pr = np.zeros(mask.sum(), dtype=float)
                    tr = np.zeros(mask.sum(), dtype=float)
                    pr[~safe] = 0.0
                    tr[~safe] = 0.0
                    pr[safe] = (inv_preds[mask][safe] - lc[safe]) / lc[safe]
                    tr[safe] = (inv_targets[mask][safe] - lc[safe]) / lc[safe]
                    sub_metrics['dir_acc'] = float(np.mean(np.sign(pr) == np.sign(tr)))
                    sub_metrics['sharpe_pred'] = _sharpe(pr)
                    sub_metrics['sharpe_true'] = _sharpe(tr)
                else:
                    sub_metrics['dir_acc'] = float('nan')
                    sub_metrics['sharpe_pred'] = 0.0
                    sub_metrics['sharpe_true'] = 0.0

                per_ticker[str(t)] = sub_metrics

            metrics['per_ticker'] = per_ticker

        return metrics

    # Default: returns or other targets — compute metrics directly (predictions & targets already unscaled)
    # Include per-ticker breakdown when ticker information is available
    ticker_array = np.array(all_tickers) if any(t is not None for t in all_tickers) else None

    metrics = compute_regression_metrics(
        inv_preds,
        inv_targets,
        include_r2=True,
        include_sharpe=True,
        ticker_ids=ticker_array
    )

    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[LRScheduler] = None,
    scheduler_type: Optional[str] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_type: str = "adam",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    save_dir: str | Path = "checkpoints",
    ckpt_name: str = "best.pt",
    early_stopping_patience: Optional[int] = 10,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,
    y_scaler: Optional[Any] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List], Path]:
    """Full training loop with validation and checkpointing.
    
    Returns:
        Tuple of (training_history, path_to_best_checkpoint)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract dataset metadata for checkpointing
    train_dataset = train_loader.dataset if hasattr(train_loader, 'dataset') else None
    dataset_info = {}
    if train_dataset is not None:
        dataset_info["target_type"] = getattr(train_dataset, 'target_type', 'unknown')
        dataset_info["feature_cols"] = getattr(train_dataset, 'feature_cols', [])
        dataset_info["n_features"] = train_dataset.X.shape[-1] if hasattr(train_dataset, 'X') else None
    
    # Setup optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            **optimizer_kwargs
        )
    elif optimizer_type.lower() == "sgd":
        momentum = optimizer_kwargs.get("momentum", 0.9)
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum
        )
    else:  # Default to Adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            **optimizer_kwargs
        )
    
    # Setup loss function
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    # Setup scheduler
    if scheduler is None and scheduler_type == "plateau":
        default_kwargs = {"factor": 0.5, "patience": 3, "min_lr": 1e-6}
        if scheduler_kwargs:
            default_kwargs.update(scheduler_kwargs)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=default_kwargs["factor"], 
            patience=default_kwargs["patience"], 
            min_lr=default_kwargs["min_lr"]
        )
        
        if verbose:
            print(f"Using ReduceLROnPlateau with factor={default_kwargs['factor']}, "
                  f"patience={default_kwargs['patience']}")
    
    # Setup checkpointing
    save_dir = Path(save_dir)
    best_ckpt_path = save_dir / ckpt_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training state
    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_dir_acc": [],
        "val_r2": [],
        "val_sharpe_pred": [],
        "lr": [],
        "grad_norm": [],
    }
    
    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss, avg_grad_norm = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip
        )
        
        # Validation phase
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Compute unscaled metrics if scaler is provided
        unscaled_metrics = None
        if y_scaler is not None:
            try:
                unscaled_metrics = compute_unscaled_metrics(
                    model, val_loader, device, y_scaler
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to compute unscaled metrics: {e}")
                unscaled_metrics = None
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(avg_grad_norm)
        
        # Use unscaled metrics if available (prefer unscaled for reporting)
        if unscaled_metrics is not None:
            history["val_mae"].append(unscaled_metrics["mae"])
            history["val_rmse"].append(unscaled_metrics["rmse"])
            history["val_dir_acc"].append(unscaled_metrics.get("dir_acc", 0.0))
            history["val_r2"].append(unscaled_metrics.get("r2", 0.0))
            history["val_sharpe_pred"].append(unscaled_metrics.get("sharpe_pred", 0.0))
        else:
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_dir_acc"].append(val_metrics.get("dir_acc", 0.0))
            history["val_r2"].append(val_metrics.get("r2", 0.0))
            history["val_sharpe_pred"].append(val_metrics.get("sharpe_pred", 0.0))
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Check for improvement
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            
            # Save best checkpoint
            extra = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "train_loss": train_loss,
                "history": history,
            }
            
            # Include model configuration to allow exact reconstruction on load
            extra["model_config"] = {
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "pooling": model.pooling,
                "bidirectional": model.bidirectional,
                "num_tickers": model.num_tickers,
                "ticker_emb_dim": model.ticker_emb_dim,
                "expansion_factor": getattr(model, "expansion_factor", 1),
            }
            
            # Include dataset configuration
            if dataset_info:
                extra["dataset_info"] = dataset_info
            
            if unscaled_metrics is not None:
                extra["val_metrics_unscaled"] = unscaled_metrics
            
            if y_scaler is not None:
                extra["y_scaler_info"] = _serialize_scaler(y_scaler)
            
            if run_metadata is not None:
                extra["run_metadata"] = run_metadata
            
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch=epoch,
                best_val=best_val,
                extra=extra,
            )
        else:
            no_improve += 1
        
        # Log progress
        if verbose:
            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val MAE: {history['val_mae'][-1]:.5f} | "
                f"Val RMSE: {history['val_rmse'][-1]:.5f} | "
                f"Val R2: {history.get('val_r2', [0.0])[-1]:.4f} | "
                f"Val Sharpe: {history.get('val_sharpe_pred', [0.0])[-1]:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            print(log_msg)
        
        # Early stopping
        if early_stopping_patience and no_improve >= early_stopping_patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final checkpoint
    final_extra = {
        "epoch": epochs,
        "val_loss": val_loss,
        "history": history,
    }
    
    # Add model configuration to final checkpoint as well
    final_extra["model_config"] = {
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "pooling": model.pooling,
        "bidirectional": model.bidirectional,
        "num_tickers": model.num_tickers,
        "ticker_emb_dim": model.ticker_emb_dim,
        "expansion_factor": getattr(model, "expansion_factor", 1),
    }
    
    # Include dataset configuration
    if dataset_info:
        final_extra["dataset_info"] = dataset_info
    
    if y_scaler is not None:
        final_extra["y_scaler_info"] = _serialize_scaler(y_scaler)
    
    save_checkpoint(
        save_dir / "final.pt",
        model,
        optimizer,
        epoch=epochs,
        best_val=best_val,
        extra=final_extra,
    )
    
    # Update history with final best values
    history["best_val"] = best_val
    history["best_epoch"] = best_epoch
    
    if verbose:
        print(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val:.5f}")
    
    return history, best_ckpt_path


def _serialize_scaler(scaler: Any) -> Dict[str, Any]:
    """Serialize a scaler object for checkpointing."""
    if isinstance(scaler, dict):
        # Dictionary of per-ticker scalers
        serialized = {}
        for ticker, ticker_scaler in scaler.items():
            if hasattr(ticker_scaler, "mean_") and hasattr(ticker_scaler, "scale_"):
                serialized[ticker] = {
                    "mean": ticker_scaler.mean_.tolist(),
                    "scale": ticker_scaler.scale_.tolist(),
                }
        return {"type": "dict", "scalers": serialized}
    else:
        # Single scaler
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            return {
                "type": "single",
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
    
    return {"type": "unknown"}


def evaluate_on_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Convenience function to evaluate model on any loader.

    If dataset has a `target_scaler` attached (per-ticker or single), compute
    and return metrics in the original (unscaled) target space for better
    interpretability (mae/rmse/dir_acc will reflect unscaled values).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    loss, metrics = evaluate(model, loader, loss_fn, device)

    # If dataset has a target scaler attached, compute unscaled metrics and
    # override reported mae/rmse/dir_acc with unscaled values (keep originals
    # as *_scaled keys)
    try:
        dataset = loader.dataset
        y_scaler = getattr(dataset, "target_scaler", None)
        if y_scaler is not None:
            unscaled = compute_unscaled_metrics(model, loader, device, y_scaler)
            # Preserve scaled values
            metrics["mae_scaled"] = metrics.get("mae")
            metrics["rmse_scaled"] = metrics.get("rmse")
            metrics["mae"] = unscaled.get("mae", metrics.get("mae"))
            metrics["rmse"] = unscaled.get("rmse", metrics.get("rmse"))
            if "dir_acc" in unscaled:
                metrics["dir_acc"] = unscaled.get("dir_acc")
            # Expose unscaled metrics explicitly
            metrics["mae_unscaled"] = metrics["mae"]
            metrics["rmse_unscaled"] = metrics["rmse"]
    except Exception:
        # Best-effort: if unscaled computation fails, keep scaled metrics
        pass
    
    return {"loss": loss, **metrics}


def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and true values."""
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    
    with torch.no_grad():
        for batch in loader:
            xb, yb, ticker_idx = unpack_batch(batch)
            
            xb = xb.to(device)
            if ticker_idx is not None:
                ticker_idx = ticker_idx.to(device)
                preds = model(xb, ticker_idx)
            else:
                preds = model(xb)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.numpy())
    
    predictions = np.concatenate(all_preds, axis=0).ravel()
    targets = np.concatenate(all_targets, axis=0).ravel()
    
    return predictions, targets