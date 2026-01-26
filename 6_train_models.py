"""6_train_models.py

Full hyperparameter sweep training script.
Trains multiple LSTM configurations on all or specified tickers.
"""

import torch
import argparse
from pathlib import Path
from core.PriceNewsDataset import build_and_save_datasets, load_dataloaders
from core.Model import PriceNewsLSTMReg
from core.train import train_model
from core.checkpoint import get_all_tickers, make_save_dir
from core.plotter import plot_training_history


def train_with_config(
    config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    verbose: bool = True,
):
    """Train a model with the given configuration."""
    save_dir = make_save_dir(config)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training config: h{config['hidden_size']}_l{config['num_layers']}_d{config['dropout']}_lr{config['lr']}")
        print(f"{'='*80}")
    
    # Build model
    input_size = train_loader.dataset.X.shape[-1]
    model = PriceNewsLSTMReg(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config.get("pooling", "last"),
        bidirectional=config.get("bidirectional", False),
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Get y_scaler if available
    y_scaler = None
    if hasattr(train_loader.dataset, 'target_scaler') and train_loader.dataset.target_scaler is not None:
        y_scaler = train_loader.dataset.target_scaler
    
    # Train
    history, ckpt_path = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.get("epochs", 100),
        lr=config["lr"],
        grad_clip=config.get("grad_clip", 5.0),
        early_stopping_patience=config.get("early_stopping_patience", 20),
        verbose=verbose,
        scheduler_type=config.get("scheduler_type", "ReduceLROnPlateau"),
        scheduler_kwargs=config.get("scheduler_kwargs", {"factor": 0.7, "patience": 5, "min_lr": 1e-5}),
        save_dir=save_dir,
        ckpt_name="best.pt",
        y_scaler=y_scaler,
    )
    
    # Plot training history
    if verbose:
        print(f"Best epoch: {history.get('best_epoch', 'N/A')}, Best val: {history.get('best_val', 'N/A'):.6f}")
    plot_training_history(history, save_path=save_dir / "training_history.png")
    
    return history, ckpt_path



def main():
    parser = argparse.ArgumentParser(description="Full hyperparameter sweep training")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to use (default: use all)")
    parser.add_argument("--max_tickers", type=int, default=None, help="Max number of tickers to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--early_stopping", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer configs, fewer epochs")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers
        print(f"Using specified tickers: {tickers}")
    else:
        tickers = get_all_tickers()
        print(f"Found {len(tickers)} tickers")
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
            print(f"Using first {len(tickers)} tickers")
    
    # Build/load datasets
    data_dir = f"processed_data/{len(tickers)}tickers_seq{args.seq_len}"
    print(f"\nBuilding/loading datasets in {data_dir}...")
    
    try:
        train_loader, val_loader, test_loader = load_dataloaders(data_dir, args.batch_size, num_workers=0)
        print(f"Loaded existing datasets")
    except (FileNotFoundError, RuntimeError):
        print(f"Building new datasets...")
        build_and_save_datasets(
            tickers, 
            data_dir, 
            seq_len=args.seq_len, 
            target_scaling=True
        )
        train_loader, val_loader, test_loader = load_dataloaders(data_dir, args.batch_size, num_workers=0)
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Define configurations
    if args.quick:
        # Quick mode: 2 configs, fewer epochs
        configs = [
            {
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "last",
                "epochs": min(30, args.epochs),
                "early_stopping_patience": 10,
            },
            {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "mean",
                "epochs": min(30, args.epochs),
                "early_stopping_patience": 10,
            },
        ]
    else:
        # Full sweep
        configs = [
            # Small model
            {
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "last",
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
            },
            # Medium model with mean pooling
            {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "mean",
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
            },
            # Larger model
            {
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "mean",
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
            },
            # Lower learning rate
            {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "lr": 5e-4,
                "batch_size": args.batch_size,
                "pooling": "mean",
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
            },
            # Higher dropout
            {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "lr": 1e-3,
                "batch_size": args.batch_size,
                "pooling": "mean",
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
            },
        ]
    
    print(f"\nRunning hyperparameter sweep with {len(configs)} configurations\n")
    
    # Train all configs
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"{'#'*80}")
        
        try:
            history, ckpt_path = train_with_config(config, train_loader, val_loader, device)
            results.append((config, history, ckpt_path))
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Trained {len(results)}/{len(configs)} models successfully")
    print(f"Results saved in experiments/ directory")
    print("\nNext steps:")
    print("  1. python 7_evaluate_models.py    # Find best model")
    print("  2. python 8_compare_models.py     # Compare with baselines")


if __name__ == "__main__":
    main()
