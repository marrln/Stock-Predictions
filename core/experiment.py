"""Experiment configuration and management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import torch.nn as nn


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    # Data settings
    tickers: List[str] = field(default_factory=list)
    seq_len: int = 8
    target_type: str = "return"
    sentiment_fill: str = "ffill"
    batch_size: int = 64
    target_scaling: bool = True
    
    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    pooling: str = "last"
    bidirectional: bool = False
    use_ticker_embedding: bool = True
    ticker_emb_dim: int = 16
    
    # Training hyperparameters
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    huber_delta: float = 1.0
    epochs: int = 100
    early_stopping_patience: int = 20
    grad_clip: float = 1.0
    
    # Scheduler
    scheduler_type: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    # Experiment management
    experiment_name: Optional[str] = None
    save_dir: str = "experiments"
    data_dir: str = "processed_data"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Create config from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function based on config."""
        if self.loss == "mse":
            return nn.MSELoss()
        elif self.loss == "huber":
            return nn.HuberLoss(delta=self.huber_delta)
        elif self.loss == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")


@dataclass
class ExperimentResult:
    """Results from a training experiment."""
    config: ExperimentConfig
    history: Dict[str, List[float]]
    checkpoint_path: Path
    best_val_loss: float
    best_epoch: int
    val_metrics: Dict[str, float]
    
    def save(self, path: Path) -> None:
        """Save experiment results to JSON."""
        data = {
            "config": self.config.to_dict(),
            "history": self.history,
            "checkpoint_path": str(self.checkpoint_path),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "val_metrics": self.val_metrics
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def create_hyperparameter_grid() -> List[ExperimentConfig]:
    """Create a grid of hyperparameters to sweep."""
    base_config = ExperimentConfig()
    
    # Define variations
    hidden_sizes = [64, 128, 256]
    num_layers_list = [1, 2, 3]
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [1e-3, 5e-4, 1e-4]
    pooling_methods = ["last", "mean", "max"]
    
    configs = []
    
    for hs in hidden_sizes:
        for nl in num_layers_list:
            for dr in dropout_rates:
                for lr in learning_rates:
                    for pool in pooling_methods:
                        config = ExperimentConfig(
                            hidden_size=hs,
                            num_layers=nl,
                            dropout=dr,
                            lr=lr,
                            pooling=pool,
                            experiment_name=f"h{hs}_l{nl}_d{dr}_lr{lr}_p{pool}"
                        )
                        configs.append(config)
    
    return configs


def create_quick_grid() -> List[ExperimentConfig]:
    """Create a smaller grid for quick experiments."""
    return [
        ExperimentConfig(
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
            lr=1e-3,
            pooling="last",
            epochs=30,
            early_stopping_patience=10,
            experiment_name="quick_small"
        ),
        ExperimentConfig(
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            lr=1e-3,
            pooling="mean",
            epochs=30,
            early_stopping_patience=10,
            experiment_name="quick_medium"
        ),
    ]