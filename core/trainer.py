"""Main trainer class for simplified workflow."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn as nn

from .data.loaders import load_or_build_datasets, build_and_save_datasets
from .models.lstm import PriceNewsLSTMReg
from .experiment import ExperimentConfig, ExperimentResult
from .training.trainer import train_model
from .checkpoint import make_save_dir
from .utils.plotting import plot_training_history


class LSTMTrainer:
    """Simplified trainer for LSTM models."""
    
    def __init__(self, config: ExperimentConfig, device: Optional[str] = None):
        self.config = config
        self.device = self._setup_device(device)
        self.loss_fn = config.get_loss_function()
        
        # Will be set during setup
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def setup_data(self, force_rebuild: bool = False) -> None:
        """Setup data loaders."""
        # Create data directory name
        data_dir_name = f"{len(self.config.tickers)}tickers_seq{self.config.seq_len}"
        data_dir = Path(self.config.data_dir) / data_dir_name
        
        print(f"Setting up data from {data_dir}")
        
        self.train_loader, self.val_loader, self.test_loader = load_or_build_datasets(
            tickers=self.config.tickers,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
            save_dir=str(data_dir),
            sentiment_fill=self.config.sentiment_fill,
            target_type=self.config.target_type,
            force_build=force_rebuild,
            target_scaling=self.config.target_scaling,
        )
        
        print(f"Data loaded: Train={len(self.train_loader.dataset)}, "
              f"Val={len(self.val_loader.dataset)}, Test={len(self.test_loader.dataset)}")
    
    def setup_model(self) -> None:
        """Initialize the model."""
        if self.train_loader is None:
            raise RuntimeError("Must call setup_data() before setup_model()")
        
        # Get input size from dataset
        input_size = self.train_loader.dataset.X.shape[-1]
        
        # Determine if we should use ticker embeddings
        if self.config.use_ticker_embedding and hasattr(self.train_loader.dataset, 'ticker_to_idx'):
            num_tickers = len(self.train_loader.dataset.ticker_to_idx)
        else:
            num_tickers = None
            self.config.ticker_emb_dim = 0
        
        print(f"Creating model with input_size={input_size}, "
              f"hidden_size={self.config.hidden_size}, "
              f"num_tickers={num_tickers}")
        
        self.model = PriceNewsLSTMReg(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            pooling=self.config.pooling,
            bidirectional=self.config.bidirectional,
            num_tickers=num_tickers,
            ticker_emb_dim=self.config.ticker_emb_dim,
        )
        
        print(f"Model has {self.model.total_parameters:,} parameters "
              f"({self.model.trainable_parameters:,} trainable)")
    
    def train(self, verbose: bool = True) -> ExperimentResult:
        """Train the model and return results."""
        if self.model is None:
            self.setup_model()
        
        # Create save directory
        save_dir = make_save_dir(self.config.to_dict(), base_dir=self.config.save_dir)
        
        # Get target scaler if available
        y_scaler = None
        if hasattr(self.train_loader.dataset, 'target_scaler'):
            y_scaler = self.train_loader.dataset.target_scaler
        
        # Prepare run metadata
        run_metadata = {
            "config": self.config.to_dict(),
            "data_info": {
                "n_train": len(self.train_loader.dataset),
                "n_val": len(self.val_loader.dataset),
                "n_test": len(self.test_loader.dataset),
                "feature_cols": getattr(self.train_loader.dataset, 'feature_cols', []),
            }
        }
        
        # Train the model
        history, best_ckpt_path = train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config.epochs,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            loss_fn=self.loss_fn,
            scheduler_type=self.config.scheduler_type,
            scheduler_kwargs={
                "factor": self.config.scheduler_factor,
                "patience": self.config.scheduler_patience,
                "min_lr": self.config.scheduler_min_lr,
            },
            optimizer_type=self.config.optimizer,
            save_dir=str(save_dir),
            ckpt_name="best.pt",
            early_stopping_patience=self.config.early_stopping_patience,
            grad_clip=self.config.grad_clip,
            verbose=verbose,
            y_scaler=y_scaler,
            run_metadata=run_metadata,
        )
        
        # Plot training history
        plot_training_history(history, save_path=save_dir / "training_history.png")
        
        # Create result object
        result = ExperimentResult(
            config=self.config,
            history=history,
            checkpoint_path=best_ckpt_path,
            best_val_loss=history.get("best_val", float("inf")),
            best_epoch=history.get("best_epoch", -1),
            val_metrics=history.get("val_metrics", {}),
        )
        
        # Save result
        result.save(save_dir / "results.json")
        
        return result
    
    def evaluate(self, loader_type: str = "test") -> Dict[str, float]:
        """Evaluate model on a specific loader."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        from .training.trainer import evaluate_on_loader
        
        if loader_type == "train":
            loader = self.train_loader
        elif loader_type == "val":
            loader = self.val_loader
        elif loader_type == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        return evaluate_on_loader(
            self.model, loader, device=self.device, loss_fn=self.loss_fn
        )


def hyperparameter_sweep(
    configs: List[ExperimentConfig],
    tickers: List[str],
    quick_mode: bool = False,
    output_dir: str = "experiments/sweep_results",
) -> List[ExperimentResult]:
    """Run hyperparameter sweep across multiple configurations."""
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(tqdm(configs, desc="Hyperparameter sweep")):
        print(f"\n{'='*80}")
        print(f"Training configuration {i+1}/{len(configs)}")
        print(f"Config: {config.experiment_name or f'config_{i}'}")
        print(f"{'='*80}")
        
        # Update tickers for this config
        config.tickers = tickers
        
        try:
            # Create trainer
            trainer = LSTMTrainer(config)
            
            # Setup data (force rebuild only for first config)
            trainer.setup_data(force_rebuild=(i == 0))
            
            # Train
            result = trainer.train(verbose=True)
            results.append(result)
            
            # Save individual result
            result.save(output_path / f"result_{i}.json")
            
            # Quick mode: break after 2 configs
            if quick_mode and i >= 1:
                print("Quick mode: Stopping after 2 configurations")
                break
                
        except Exception as e:
            print(f"Error training config {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results