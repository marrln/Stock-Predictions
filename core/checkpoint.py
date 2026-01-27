"""Checkpoint utilities for naming and organization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch


def make_checkpoint_name(config: Dict[str, Any], prefix: str = "best") -> str:
    """Generate a descriptive checkpoint filename from hyperparameters.
    
    Example: best_h128_l2_d0.2_lr0.001_b64_optadam_losshuber_hd1.0_ptscale.pt
    """
    parts = [prefix]
    
    # Core architecture parameters
    parts.append(f"h{config.get('hidden_size', 128)}")
    parts.append(f"l{config.get('num_layers', 2)}")
    parts.append(f"d{config.get('dropout', 0.2)}")
    parts.append(f"ef{config.get('expansion_factor', 4)}")
    
    # Training parameters
    parts.append(f"lr{config.get('lr', 0.001)}")
    parts.append(f"b{config.get('batch_size', 64)}")
    
    # Optional pooling
    if config.get('pooling') and config['pooling'] != 'last':
        parts.append(f"p{config['pooling']}")
    
    # Optimizer
    if config.get('optimizer'):
        parts.append(f"opt{config['optimizer']}")
    
    # Loss function
    if config.get('loss') and config['loss'] != 'mse':
        parts.append(f"loss{config['loss']}")
        if config['loss'] == 'huber' and 'huber_delta' in config:
            parts.append(f"hd{config['huber_delta']}")
    
    # Target scaling
    if config.get('target_scaling', True):
        parts.append("ptscale")
    
    return "_".join(parts) + ".pt"


def make_save_dir(config: Dict[str, Any], base_dir: str = "experiments") -> Path:
    """Create experiment directory with hyperparameter configuration."""
    dir_name = make_checkpoint_name(config, prefix="").strip("_").replace(".pt", "")
    save_dir = Path(base_dir) / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)  # Handle non-serializable types
    
    return save_dir


def load_config_from_dir(exp_dir: Path) -> Dict[str, Any]:
    """Load config.json from experiment directory."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        return json.load(f)


def find_all_experiments(base_dir: str = "experiments") -> List[Tuple[Path, Dict, Path]]:
    """Find all experiment directories with checkpoints.

    This searches recursively for checkpoint files (e.g., `best.pt`) so it will
    discover experiments organized under fold directories (e.g., `fold_0/<exp>/best.pt`).

    Returns:
        List of (exp_dir, config, best_ckpt_path) tuples
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    experiments = []

    # Search recursively for best.pt, last.pt or any .pt file
    ckpt_candidates = list(base_path.rglob("best.pt"))
    # If no explicit best.pt files, look for last.pt
    if not ckpt_candidates:
        ckpt_candidates = list(base_path.rglob("last.pt"))
    # If still none, fall back to any .pt files
    if not ckpt_candidates:
        ckpt_candidates = list(base_path.rglob("*.pt"))

    # Deduplicate and sort
    ckpt_candidates = sorted(set(ckpt_candidates))

    for ckpt_path in ckpt_candidates:
        exp_dir = ckpt_path.parent
        # Skip comparison output dir
        if "comparison" in exp_dir.parts:
            continue

        config = load_config_from_dir(exp_dir)
        experiments.append((exp_dir, config, ckpt_path))

    return experiments


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_val: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    
    if extra:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint and restore model/optimizer."""
    checkpoint = torch.load(path, map_location=map_location)
    
    model.load_state_dict(checkpoint["model_state"])
    
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    return checkpoint