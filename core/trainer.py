"""Main trainer class for simplified workflow."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from .data.loaders import load_or_build_rolling_folds
from .models.lstm import PriceNewsLSTMReg
from .experiment import ExperimentConfig, ExperimentResult
from .training.trainer import train_model
from .checkpoint import make_save_dir
from .utils.plotting import plot_training_history


class LSTMTrainer:
    """Simplified trainer for LSTM models with rolling window CV."""
    
    def __init__(self, config: ExperimentConfig, device: Optional[str] = None):
        self.config = config
        self.device = self._setup_device(device)
        self.loss_fn = config.get_loss_function()
        
        self.folds = None
        self.model = None
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def setup_data(self, force_rebuild: bool = False) -> None:
        """Setup rolling fold data loaders."""
        data_dir_name = f"{len(self.config.tickers)}tickers_seq{self.config.seq_len}_rolling"
        data_dir = Path(self.config.data_dir) / data_dir_name
        
        print(f"Setting up rolling folds from {data_dir}")
        
        self.folds = load_or_build_rolling_folds(
            tickers=self.config.tickers,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
            save_dir=str(data_dir),
            sentiment_fill=self.config.sentiment_fill,
            target_type=self.config.target_type,
            force_build=force_rebuild,
            target_scaling=self.config.target_scaling,
            train_days=self.config.train_days,
            val_days=self.config.val_days,
            test_days=self.config.test_days,
            step_days=self.config.step_days,
        )
        
        print(f"Loaded {len(self.folds)} rolling folds")
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(self.folds):
            print(f"  Fold {fold_idx}: Train={len(train_loader.dataset)}, "
                  f"Val={len(val_loader.dataset)}, "
                  f"Test={len(test_loader.dataset) if test_loader else 0}")
    
    def setup_model(self, fold_idx: int = 0) -> None:
        """Initialize the model based on a specific fold."""
        if self.folds is None:
            raise RuntimeError("Must call setup_data() before setup_model()")
        
        train_loader, _, _ = self.folds[fold_idx]
        input_size = train_loader.dataset.X.shape[-1]
        
        if self.config.use_ticker_embedding and hasattr(train_loader.dataset, 'ticker_to_idx'):
            num_tickers = len(train_loader.dataset.ticker_to_idx)
        else:
            num_tickers = None
            self.config.ticker_emb_dim = 0
        
        print(f"Creating model with input_size={input_size}, "
              f"hidden_size={self.config.hidden_size}, "
              f"expansion_factor={self.config.expansion_factor}, "
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
            expansion_factor=self.config.expansion_factor,
        )
        
        print(f"Model has {self.model.total_parameters:,} parameters "
              f"({self.model.trainable_parameters:,} trainable)")
    
    def train(self, verbose: bool = True) -> List[ExperimentResult]:
        """Train the model on all folds and return results."""
        if self.folds is None:
            raise RuntimeError("Must call setup_data() before train()")
        
        results = []
        
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(self.folds):
            print(f"\n{'='*80}")
            print(f"Training Fold {fold_idx + 1}/{len(self.folds)}")
            print(f"{'='*80}")
            
            self.setup_model(fold_idx)
            
            save_dir = make_save_dir(
                self.config.to_dict(), 
                base_dir=Path(self.config.save_dir) / f"fold_{fold_idx}"
            )
            
            y_scaler = None
            if hasattr(train_loader.dataset, 'target_scaler'):
                y_scaler = train_loader.dataset.target_scaler
            
            run_metadata = {
                "config": self.config.to_dict(),
                "fold_idx": fold_idx,
                "data_info": {
                    "n_train": len(train_loader.dataset),
                    "n_val": len(val_loader.dataset),
                    "n_test": len(test_loader.dataset) if test_loader else 0,
                    "feature_cols": getattr(train_loader.dataset, 'feature_cols', []),
                }
            }
            
            history, best_ckpt_path = train_model(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
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
            
            plot_training_history(history, save_path=save_dir / "training_history.png")
            
            best_epoch = history.get("best_epoch", -1)
            best_val_metrics = {
                "loss": history.get("best_val", float("inf")),
                "mae": history["val_mae"][best_epoch - 1] if best_epoch > 0 and len(history.get("val_mae", [])) >= best_epoch else float("nan"),
                "rmse": history["val_rmse"][best_epoch - 1] if best_epoch > 0 and len(history.get("val_rmse", [])) >= best_epoch else float("nan"),
                "dir_acc": history["val_dir_acc"][best_epoch - 1] if best_epoch > 0 and len(history.get("val_dir_acc", [])) >= best_epoch else float("nan"),
                "r2": history["val_r2"][best_epoch - 1] if best_epoch > 0 and len(history.get("val_r2", [])) >= best_epoch else float("nan"),
                "sharpe_pred": history["val_sharpe_pred"][best_epoch - 1] if best_epoch > 0 and len(history.get("val_sharpe_pred", [])) >= best_epoch else float("nan"),
            }
            
            result = ExperimentResult(
                config=self.config,
                history=history,
                checkpoint_path=best_ckpt_path,
                best_val_loss=history.get("best_val", float("inf")),
                best_epoch=best_epoch,
                val_metrics=best_val_metrics,
            )
            
            result.save(save_dir / "results.json")
            results.append(result)
        
        return results
    
    def evaluate(self, fold_idx: int = 0, loader_type: str = "test") -> Dict[str, float]:
        """Evaluate model on a specific fold and loader."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        from .training.trainer import evaluate_on_loader
        
        train_loader, val_loader, test_loader = self.folds[fold_idx]
        
        if loader_type == "train":
            loader = train_loader
        elif loader_type == "val":
            loader = val_loader
        elif loader_type == "test":
            if test_loader is None:
                raise ValueError(f"Fold {fold_idx} has no test set")
            loader = test_loader
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
            trainer = LSTMTrainer(config)
            
            trainer.setup_data(force_rebuild=(i == 0))
            
            fold_results = trainer.train(verbose=True)
            results.extend(fold_results)
            
            for fold_idx, result in enumerate(fold_results):
                result.save(output_path / f"result_{i}_fold_{fold_idx}.json")
            
            if quick_mode and i >= 1:
                print("Quick mode: Stopping after 2 configurations")
                break
                
        except Exception as e:
            print(f"Error training config {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results