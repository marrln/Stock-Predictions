"""Helper module for training the LSTM model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	loss_fn: nn.Module,
	device: torch.device,
	grad_clip: Optional[float] = 1.0,
) -> float:
	"""Run one epoch of training and return average loss."""
	model.train()
	running_loss = 0.0
	n_batches = 0

	for batch in loader:
		# Expect batches of the form (X, y) or (X, y, meta)
		if isinstance(batch, (list, tuple)):
			xb, yb = batch[0], batch[1]
		else:
			xb, yb = batch  # type: ignore
		xb = xb.to(device)
		yb = yb.to(device)

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
	running_loss = 0.0
	n_batches = 0
	mae_sum = 0.0
	rmse_sum = 0.0

	with torch.no_grad():
		for batch in loader:
			# Expect batches of the form (X, y) or (X, y, meta)
			if isinstance(batch, (list, tuple)):
				xb, yb = batch[0], batch[1]
			else:
				xb, yb = batch  # type: ignore
			xb = xb.to(device)
			yb = yb.to(device)

			preds = model(xb)
			loss = loss_fn(preds, yb)

			running_loss += loss.item()
			n_batches += 1

			# Basic regression metrics
			err = (preds - yb).abs()
			mae_sum += err.mean().item()
			rmse_sum += torch.sqrt(torch.mean((preds - yb) ** 2)).item()

	avg_loss = running_loss / max(n_batches, 1)
	metrics = {
		"mae": mae_sum / max(n_batches, 1),
		"rmse": rmse_sum / max(n_batches, 1),
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
	scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
	save_dir: str | Path = "checkpoints",
	ckpt_name: str = "best.pt",
	early_stopping_patience: Optional[int] = 5,
	grad_clip: Optional[float] = 1.0,
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

	history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(
			model, train_loader, optimizer, loss_fn, device, grad_clip=grad_clip
		)
		val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["val_mae"].append(val_metrics["mae"])  # type: ignore[index]
		history["val_rmse"].append(val_metrics["rmse"])  # type: ignore[index]

		if scheduler is not None:
			scheduler.step()

		improved = val_loss < best_val
		if improved:
			best_val = val_loss
			best_epoch = epoch
			no_improve = 0
			save_checkpoint(
				best_ckpt_path, model, optimizer, epoch=epoch, best_val=best_val,
				extra={"val_metrics": val_metrics, "train_loss": train_loss}
			)
		else:
			no_improve += 1

		print(
			f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} "
			f"| val_mae={val_metrics['mae']:.5f} | val_rmse={val_metrics['rmse']:.5f}"
		)

		if early_stopping_patience is not None and no_improve >= early_stopping_patience:
			print(f"Early stopping after {epoch} epochs (best at {best_epoch}, val={best_val:.5f}).")
			break

	# Also save a 'last.pt' for convenience
	save_checkpoint(save_dir / "last.pt", model, optimizer, epoch=epoch, best_val=best_val)
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



