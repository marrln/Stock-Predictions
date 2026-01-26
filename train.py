"""Helper module for training the LSTM model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from torch.optim.lr_scheduler import LRScheduler 
import torch
import torch.nn as nn
import torch.optim as optim


def _unpack_and_validate_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack a batch and validate it follows the enforced (X, y) contract.

    Enforced assumptions:
      - batch is a sequence (list/tuple) with at least two elements
      - X is a 3D tensor (B, S, F)
      - y is a 1D or 2D tensor with leading dim B
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch as (X, y); got {type(batch)}")
    if len(batch) < 2:
        raise ValueError("Batch must contain (X, y)")
    xb, yb = batch[0], batch[1]
    if not torch.is_tensor(xb) or not torch.is_tensor(yb):
        raise TypeError("Both X and y must be torch.Tensor")
    if xb.ndim != 3:
        raise ValueError(f"X must be 3D tensor (batch, seq_len, features); got {tuple(xb.shape)}")
    if yb.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D tensor (batch,) or (batch,1); got shape {tuple(yb.shape)}")
    if yb.shape[0] != xb.shape[0]:
        raise ValueError("Batch size mismatch between X and y")
    # Squeeze y if shape (batch,1)
    if yb.ndim == 2 and yb.shape[1] == 1:
        yb = yb.squeeze(1)
    return xb, yb


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
    progress_bar: bool = False,
) -> float:
    """Run one epoch of training and return average loss.

    Assumes DataLoader yields `(X, y)` batches produced by `PriceNewsDataset.make_dataloaders()`.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    # Optional progress bar
    if progress_bar:
        from tqdm import tqdm
        loader = tqdm(loader, desc="Training", leave=False)

    for batch in loader:
        xb, yb = _unpack_and_validate_batch(batch)

        xb = xb.to(device, non_blocking=True)  # non_blocking for speed
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on a loader.
    
    Returns `(avg_loss, metrics)` where metrics include MAE and RMSE
    for regression tasks.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_squared_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            xb, yb = _unpack_and_validate_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            batch_size = xb.size(0)

            # Loss (assuming reduction='mean')
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * batch_size

            # MAE
            mae = torch.abs(preds - yb).mean()
            total_mae += mae.item() * batch_size

            # Squared error for RMSE
            squared_error = torch.sum((preds - yb) ** 2)
            total_squared_error += squared_error.item()

            total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0.0
    avg_rmse = math.sqrt(total_squared_error / total_samples) if total_samples > 0 else 0.0
    
    metrics = {
        "mae": avg_mae,
        "rmse": avg_rmse,
    }
    return avg_loss, metrics


def save_checkpoint(
	path: str | Path,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer] = None,
	epoch: Optional[int] = None,
	best_val: Optional[float] = None,
	extra: Optional[Dict[str, Any]] = None,
) -> None:
	"""Save model (and optionally optimizer) state dict to `path`."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload: Dict[str, Any] = {
		"model_state": model.state_dict(),
		"epoch": epoch,
		"best_val": best_val,
	}
	if optimizer is not None:
		payload["optim_state"] = optimizer.state_dict()
	if extra:
		payload["extra"] = extra
	torch.save(payload, path)


def load_checkpoint(
	path: str | Path,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer] = None,
	map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
	"""Load checkpoint and restore model/optimizer.

	Returns the full checkpoint dict for further inspection.
	"""
	ckpt = torch.load(path, map_location=map_location)
	model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
	if optimizer is not None and "optim_state" in ckpt:
		optimizer.load_state_dict(ckpt["optim_state"])  # type: ignore[arg-type]
	return ckpt


def _compute_unscaled_metrics(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, y_scaler: Any):
    """Compute MSE/MAE/RMSE/dir_acc on the original (unscaled) target values.

    Assumes y_scaler implements inverse_transform(array.reshape(-1,1)).
    Returns a dict with keys 'mse','mae','rmse','dir_acc'.
    """
    import numpy as _np
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().reshape(-1, 1)
            yb_np = yb.numpy().reshape(-1, 1)
            preds.append(out)
            trues.append(yb_np)
    preds = _np.concatenate(preds, axis=0)
    trues = _np.concatenate(trues, axis=0)
    # inverse-transform
    ipreds = y_scaler.inverse_transform(preds).ravel()
    itrues = y_scaler.inverse_transform(trues).ravel()
    mse = float(((ipreds - itrues) ** 2).mean())
    mae = float(_np.abs(ipreds - itrues).mean())
    rmse = float(_np.sqrt(((ipreds - itrues) ** 2).mean()))
    dir_acc = float((_np.sign(ipreds) == _np.sign(itrues)).mean())
    return {"mse": mse, "mae": mae, "rmse": rmse, "dir_acc": dir_acc}


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[LRScheduler] = None,
    scheduler_type: Optional[str] = None,  # 'plateau' to use ReduceLROnPlateau
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_dir: str | Path = "checkpoints",
    ckpt_name: str = "best.pt",
    early_stopping_patience: Optional[int] = 10,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,
    y_scaler: Optional[object] = None,  # If provided, used to report unscaled metrics and saved into checkpoint
) -> Tuple[Dict[str, list], Path]:
    """Full training loop with validation and checkpointing.

    If `scheduler_type == 'plateau'` and no `scheduler` is passed, builds a
    ReduceLROnPlateau scheduler with sensible defaults which can be overridden
    via `scheduler_kwargs`.

    When `y_scaler` is provided (e.g., a fitted sklearn StandardScaler), this
    function will compute and log **unscaled** validation metrics (MSE/MAE/RMSE)
    for interpretability while training still optimizes the (possibly scaled)
    loss used by `loss_fn`.

    Saves the best model (lowest validation loss) to `save_dir/ckpt_name`.
    Returns `(history, best_ckpt_path)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # Build scheduler if requested
    if scheduler is None and scheduler_type == "plateau":
        kwargs = {"factor": 0.5, "patience": 3, "min_lr": 1e-6}
        if scheduler_kwargs:
            kwargs.update(scheduler_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=kwargs["factor"], patience=kwargs["patience"], min_lr=kwargs["min_lr"]
        )
        if verbose:
            print(f"Using ReduceLROnPlateau(factor={kwargs['factor']}, patience={kwargs['patience']}, min_lr={kwargs['min_lr']})")

    save_dir = Path(save_dir)
    best_ckpt_path = save_dir / ckpt_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val = math.inf
    best_epoch = -1
    no_improve = 0
    
    history: Dict[str, list] = {
        "train_loss": [], 
        "val_loss": [], 
        "val_mae": [], 
        "val_rmse": [],
        "lr": []  # Track learning rate
    }
    # Store scalar best values that will be set during training
    history["best_val"] = None  # Will be updated at end
    history["best_epoch"] = None  # Will be updated at end
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip=grad_clip
        )
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # If a target scaler is provided, compute unscaled metrics for interpretability
        unscaled_metrics = None
        if y_scaler is not None:
            try:
                unscaled_metrics = _compute_unscaled_metrics(model, val_loader, device, y_scaler)
                # prefer unscaled MAE/RMSE for history and printing
                history["val_mae"].append(unscaled_metrics["mae"])
                history["val_rmse"].append(unscaled_metrics["rmse"])
            except Exception as e:
                # fallback to scaled if unscaled computation fails
                if "val_mae" not in history or len(history["val_mae"]) < len(history["train_loss"]) + 1:
                    history["val_mae"].append(val_metrics["mae"])
                    history["val_rmse"].append(val_metrics["rmse"])
                print(f"Warning: failed to compute unscaled metrics: {e}")
        else:
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])

        # Also always record loss and train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Scheduler step (handle both regular and ReduceLROnPlateau)
        lr_before = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        lr_after = optimizer.param_groups[0]['lr']
        history["lr"].append(lr_after)
        if scheduler is not None and lr_after < lr_before and verbose:
            print(f"LR reduced: {lr_before:.2e} -> {lr_after:.2e}")
        
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            extra = {
                "val_metrics": val_metrics,
                "train_loss": train_loss,
                "history": history,  # Save history in checkpoint
            }
            if unscaled_metrics is not None:
                extra["val_metrics_unscaled"] = unscaled_metrics
            # If a y_scaler was provided and exposes mean_/scale_, persist its params
            if y_scaler is not None and hasattr(y_scaler, "mean_") and hasattr(y_scaler, "scale_"):
                try:
                    extra["target_scaler"] = {
                        "mean": y_scaler.mean_.tolist(),
                        "scale": y_scaler.scale_.tolist(),
                    }
                except Exception:
                    # best-effort: skip persisting scaler if it cannot be serialized
                    pass

            # Persist simple model config so we can reconstruct the architecture later
            try:
                extra["model_config"] = {
                    "input_size": getattr(model, "input_size", None),
                    "hidden_size": getattr(model, "hidden_size", None),
                    "num_layers": getattr(model, "num_layers", None),
                    "dropout": getattr(model, "dropout", None),
                    "bidirectional": getattr(model, "bidirectional", None),
                }
            except Exception:
                pass
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
        
        if verbose:
            # Prefer unscaled MAE/RMSE when available for interpretability
            if unscaled_metrics is not None:
                disp_mae = unscaled_metrics["mae"]
                disp_rmse = unscaled_metrics["rmse"]
            else:
                disp_mae = val_metrics["mae"]
                disp_rmse = val_metrics["rmse"]
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_mae={disp_mae:.5f} | "
                f"val_rmse={disp_rmse:.5f} | "
                f"lr={history['lr'][-1]:.2e}"
            )
        
        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs "
                      f"(best at {best_epoch}, val={best_val:.5f}).")
            break
    
    # Also save a 'last.pt' for convenience
    extra_last = {"history": history}
    if y_scaler is not None and hasattr(y_scaler, "mean_") and hasattr(y_scaler, "scale_"):
        try:
            extra_last["target_scaler"] = {
                "mean": y_scaler.mean_.tolist(),
                "scale": y_scaler.scale_.tolist(),
            }
        except Exception:
            pass

    save_checkpoint(
        save_dir / "last.pt",
        model,
        optimizer,
        epoch=epochs if epoch >= epochs else epoch,
        best_val=best_val,
        extra=extra_last,
    )
    
    # Update history with final best values
    history["best_val"] = best_val
    history["best_epoch"] = best_epoch
    
    if verbose and best_epoch != -1:
        print(f"Training complete. Best epoch: {best_epoch}, "
              f"Best validation loss: {best_val:.5f}")
    
    return history, best_ckpt_path


def evaluate_on_loader(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	device: Optional[torch.device] = None,
	loss_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
	"""Convenience wrapper to get final metrics on any loader (e.g., test)."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if loss_fn is None:
		loss_fn = nn.MSELoss()
	loss, metrics = evaluate(model, loader, loss_fn, device)
	return {"loss": loss, **metrics}



