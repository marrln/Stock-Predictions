#!/usr/bin/env python3
"""
Training script for stock price prediction with LSTM.

Usage:
    python 6_train_models.py --tickers "AAPL MSFT GOOG" --quick
    python 6_train_models.py --config configs/my_config.json
"""

import argparse
from pathlib import Path
from typing import List

from core.experiment import (
    ExperimentConfig, 
    create_hyperparameter_grid, 
    create_quick_grid
)
from core.trainer import LSTMTrainer, hyperparameter_sweep
from core.data.preprocessing import get_ticker_stats


def parse_tickers(tickers_input: str) -> List[str]:
    """Parse tickers from comma/space separated string."""
    if ',' in tickers_input:
        return [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    else:
        return [t.strip().upper() for t in tickers_input.split() if t.strip()]


def get_all_tickers(price_dir: str = "Stock_price/full_history") -> List[str]:
    """Get all tickers from price directory."""
    p = Path(price_dir)
    if not p.exists():
        raise FileNotFoundError(f"Price directory not found: {price_dir}")
    
    tickers = sorted([f.stem for f in p.glob("*.csv") if f.stem.upper() == f.stem])
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM models for stock price prediction"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--tickers", type=str, default="", help="Comma-separated ticker symbols (e.g., AAPL,MSFT,GOOGL)")
    data_group.add_argument("--all-tickers", action="store_true", help="Use all available tickers")
    data_group.add_argument("--max-tickers", type=int, default=None, help="Maximum number of tickers to use")
    data_group.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    data_group.add_argument("--target", choices=["return", "close"], default="return", help="Target to predict")
    data_group.add_argument("--train-days", type=int, default=900, help="Training window size in days")
    data_group.add_argument("--val-days", type=int, default=125, help="Validation window size in days")
    data_group.add_argument("--test-days", type=int, default=125, help="Test window size in days (None to disable)")
    data_group.add_argument("--step-days", type=int, default=125, help="Step size for rolling window")
    data_group.add_argument("--no-test", action="store_true", help="Disable test window in rolling splits")
    
    # Training mode
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument("--single", action="store_true", help="Train a single model (default)")
    mode_group.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    mode_group.add_argument("--quick", action="store_true", help="Quick mode (fewer configs, fewer epochs)")
    mode_group.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Model arguments (for single mode)
    model_group = parser.add_argument_group("Model (for single mode)")
    model_group.add_argument("--hidden-size", type=int, default=32)
    model_group.add_argument("--num-layers", type=int, default=2)
    model_group.add_argument("--dropout", type=float, default=0.2)
    model_group.add_argument("--pooling", choices=["last", "mean", "max"], default="last")
    model_group.add_argument("--expansion-factor", type=int, default=4, help="Feature expansion factor applied to input features before LSTM")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_group.add_argument("--batch-size", type=int, default=32)
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--loss", choices=["mse", "huber", "l1"], default="huber")
    train_group.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    train_group.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None, help="Device to use (auto-detected if not specified)")
    
    # Experiment management
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument("--name", type=str, help="Experiment name")
    exp_group.add_argument("--save-dir", type=str, default="experiments", help="Directory to save experiments")
    
    args = parser.parse_args()
    
    # Determine tickers
    if args.all_tickers:
        tickers = get_all_tickers()
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
        print(f"Using {len(tickers)} tickers")
    elif args.tickers:
        tickers = parse_tickers(args.tickers)
        print(f"Using specified tickers: {tickers}")
    else:
        # Default to a few major stocks
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "IBM", "NVDA"]
        print(f"Using default tickers: {tickers}")
    
    # Show ticker statistics
    if tickers:
        stats = get_ticker_stats(tickers=tickers, seq_len=args.seq_len)
        print("\nTicker statistics:")
        print(stats[["num_days", "avg_volume", "possible_sequences"]].to_string())
    
    # Create or load configuration
    if args.config:
        # Load from file
        config = ExperimentConfig.from_json(Path(args.config))
        print(f"Loaded configuration from {args.config}")
    else:
        # Create configuration
        if args.single or (not args.single and not args.sweep):
            # Single model training
            config = ExperimentConfig(
                tickers=tickers,
                seq_len=args.seq_len,
                target_type=args.target,
                train_days=args.train_days,
                val_days=args.val_days,
                test_days=None if args.no_test else args.test_days,
                step_days=args.step_days,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pooling=args.pooling,
                expansion_factor=args.expansion_factor,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                loss=args.loss,
                optimizer=args.optimizer,
                experiment_name=args.name or f"single_{args.hidden_size}_{args.num_layers}",
                save_dir=args.save_dir,
            )
        else:
            # Will create multiple configs for sweep
            config = None
    
    # Run training
    if args.sweep or (config is None):
        # Hyperparameter sweep
        print("\n" + "="*80)
        print("Running hyperparameter sweep")
        print("="*80)
        
        if args.quick:
            configs = create_quick_grid()
        else:
            configs = create_hyperparameter_grid()
        
        # Update tickers for all configs
        for cfg in configs:
            cfg.tickers = tickers
        
        print(f"Testing {len(configs)} configurations")
        
        results = hyperparameter_sweep(
            configs=configs,
            tickers=tickers,
            quick_mode=args.quick,
            output_dir=f"{args.save_dir}/sweep"
        )
        
        print(f"\nCompleted {len(results)}/{len(configs)} configurations successfully")
        
    else:
        print("\n" + "="*80)
        print("Training single model")
        print("="*80)
        
        trainer = LSTMTrainer(config, device=args.device)
        trainer.setup_data()
        results = trainer.train(verbose=True)
        
        print(f"\nTrained on {len(results)} folds")
        
        for fold_idx, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx} Results:")
            print(f"{'='*60}")
            
            val_metrics = result.val_metrics
            dir_acc = val_metrics.get('dir_acc', float('nan'))
            r2 = val_metrics.get('r2', float('nan'))
            sharpe = val_metrics.get('sharpe_pred', float('nan'))
            
            loss = val_metrics.get("loss", float("nan"))
            mae = val_metrics.get("mae", float("nan"))
            rmse = val_metrics.get("rmse", float("nan"))

            print(
                f"VAL: Loss={loss:.4f}, "
                f"MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, "
                f"DirAcc={dir_acc:.2%}, "
                f"R2={r2:.2%}, "
                f"Sharpe={sharpe:.4f}"
            )
            print(f"Best epoch: {result.best_epoch}")
            print(f"Model saved to: {result.checkpoint_path}")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()