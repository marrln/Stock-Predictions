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


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[LRScheduler] = None,  # Fixed type
    save_dir: str | Path = "checkpoints",
    ckpt_name: str = "best.pt",
    early_stopping_patience: Optional[int] = 5,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,  # Add verbosity control
) -> Tuple[Dict[str, list], Path]:
    """Full training loop with validation and checkpointing.
    
    Saves the best model (lowest validation loss) to `save_dir/ckpt_name`.
    Returns `(history, best_ckpt_path)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
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
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip=grad_clip
        )
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr"].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step (handle both regular and ReduceLROnPlateau)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                best_ckpt_path, 
                model, 
                optimizer, 
                epoch=epoch, 
                best_val=best_val,
                extra={
                    "val_metrics": val_metrics, 
                    "train_loss": train_loss,
                    "history": history  # Save history in checkpoint
                }
            )
        else:
            no_improve += 1
        
        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_mae={val_metrics['mae']:.5f} | "
                f"val_rmse={val_metrics['rmse']:.5f} | "
                f"lr={history['lr'][-1]:.2e}"
            )
        
        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs "
                      f"(best at {best_epoch}, val={best_val:.5f}).")
            break
    
    # Also save a 'last.pt' for convenience
    save_checkpoint(
        save_dir / "last.pt", 
        model, 
        optimizer, 
        epoch=epochs if epoch >= epochs else epoch,
        best_val=best_val,
        extra={"history": history}
    )
    
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



