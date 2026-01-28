"""Experiment configuration and management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import torch.nn as nn

from core.training.losses import DirectionalLoss, SignPenaltyLoss, AsymmetricLoss, QuantileLoss


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    tickers: List[str] = field(default_factory=list)
    seq_len: int = 30
    target_type: str = "return"
    sentiment_fill: str = "ffill"
    batch_size: int = 64
    target_scaling: bool = True
    market_csv: Optional[str] = "data_stats/SPY.csv"  # SPY ETF price data for market features
    
    train_days: int = 750
    val_days: int = 125
    test_days: Optional[int] = 125
    step_days: int = 125
    fold_mode: str = "rolling"  # 'rolling' (fixed window) or 'expanding' (growing train)
    
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    pooling: str = "last"
    bidirectional: bool = False
    use_ticker_embedding: bool = True
    ticker_emb_dim: int = 16
    expansion_factor: int = 4
    
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    huber_delta: float = 1.0
    direction_weight: float = 2.0  # For directional loss
    direction_penalty: float = 0.0  # Additional penalty for wrong direction
    sign_penalty_alpha: float = 1.0  # For sign penalty loss
    quantile: float = 0.5  # For quantile loss
    epochs: int = 100
    early_stopping_patience: int = 20
    grad_clip: float = 1.0
    
    scheduler_type: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    experiment_name: Optional[str] = None
    save_dir: str = "experiments"
    data_dir: str = "processed_data"
    
    # Plotting configuration
    max_plot_tickers: int = 7  # Max number of tickers to plot per fold (0 = plot all)
    
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
        elif self.loss == "directional":
            return DirectionalLoss(
                base_loss='mse',
                direction_weight=self.direction_weight,
                direction_penalty=self.direction_penalty
            )
        elif self.loss == "directional_mae":
            return DirectionalLoss(
                base_loss='mae',
                direction_weight=self.direction_weight,
                direction_penalty=self.direction_penalty
            )
        elif self.loss == "sign_penalty":
            return SignPenaltyLoss(alpha=self.sign_penalty_alpha)
        elif self.loss == "quantile":
            return QuantileLoss(quantile=self.quantile)
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
        # Extract R2 and Sharpe from val_metrics or fallback to history
        val_r2 = self.val_metrics.get("r2") if isinstance(self.val_metrics, dict) else None
        val_sharpe = self.val_metrics.get("sharpe_pred") if isinstance(self.val_metrics, dict) else None

        if val_r2 is None:
            vr = self.history.get("val_r2")
            val_r2 = vr[-1] if vr else None
        if val_sharpe is None:
            vs = self.history.get("val_sharpe_pred")
            val_sharpe = vs[-1] if vs else None

        data = {
            "config": self.config.to_dict(),
            "history": self.history,
            "checkpoint_path": str(self.checkpoint_path),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "val_metrics": self.val_metrics,
            "val_r2": val_r2,
            "val_sharpe_pred": val_sharpe,
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
    expansion_factors = [2, 4, 8]
    
    configs = []
    
    for hs in hidden_sizes:
        for nl in num_layers_list:
            for dr in dropout_rates:
                for lr in learning_rates:
                    for pool in pooling_methods:
                        for ef in expansion_factors:
                            config = ExperimentConfig(
                                hidden_size=hs,
                                num_layers=nl,
                                dropout=dr,
                                lr=lr,
                                pooling=pool,
                                expansion_factor=ef,
                                experiment_name=f"h{hs}_l{nl}_d{dr}_lr{lr}_p{pool}_ef{ef}"
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
            expansion_factor=2,
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
            expansion_factor=4,
            epochs=30,
            early_stopping_patience=10,
            experiment_name="quick_medium"
        ),
    ]