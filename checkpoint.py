"""Checkpoint utilities for naming and organization."""

from pathlib import Path
from typing import Dict, Any
import json


def make_checkpoint_name(config: Dict[str, Any], prefix: str = "best") -> str:
    """Generate a descriptive checkpoint filename from hyperparameters.
    
    Example: best_h128_l2_d0.2_lr0.001_b64.pt
    """
    parts = [prefix]
    parts.append(f"h{config['hidden_size']}")
    parts.append(f"l{config['num_layers']}")
    parts.append(f"d{config['dropout']}")
    parts.append(f"lr{config['lr']}")
    parts.append(f"b{config['batch_size']}")
    if config.get('pooling') and config['pooling'] != 'last':
        parts.append(f"p{config['pooling']}")
    return "_".join(parts) + ".pt"


def make_save_dir(config: Dict[str, Any], base_dir: str = "experiments") -> Path:
    """Create experiment directory with hyperparameter configuration.
    
    Example: experiments/h128_l2_d0.2_lr0.001_b64/
    """
    dir_name = make_checkpoint_name(config, prefix="").strip("_").replace(".pt", "")
    save_dir = Path(base_dir) / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to JSON for reproducibility
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return save_dir


def get_all_tickers(price_dir: str = "Stock_price/full_history"):
    """Extract all ticker symbols from CSV filenames in the price directory."""
    price_path = Path(price_dir)
    csv_files = list(price_path.glob("*.csv"))
    tickers = sorted([f.stem for f in csv_files if f.stem.upper() == f.stem])
    return tickers


def load_config_from_dir(exp_dir: Path) -> Dict[str, Any]:
    """Load config.json from experiment directory."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def find_all_experiments(base_dir: str = "experiments") -> list:
    """Find all experiment directories with checkpoints.
    
    Returns list of (exp_dir, config, best_ckpt_path) tuples.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    experiments = []
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "comparison":
            continue
        
        best_ckpt = exp_dir / "best.pt"
        if not best_ckpt.exists():
            continue
        
        config = load_config_from_dir(exp_dir)
        experiments.append((exp_dir, config, best_ckpt))
    
    return experiments
