"""6_train_model.py

Training script for stock price prediction LSTM model (regression-only).
Loads preprocessed datasets and trains an LSTM to predict next-day return.
"""

import torch
import argparse
from PriceNewsDataset import load_dataloaders, build_and_save_datasets
from Model import PriceNewsLSTMReg


DATA_DIR = "processed_data/small_ams"  # Default datasets dir
TICKERS = ["AAPL", "MSFT", "AMZN"]  # Used only when building datasets from scratch
SEQ_LEN = 8


def get_dataloaders(batch_size: int, num_workers: int):
    """Load dataloaders from saved datasets. If datasets are not found, build and save them first."""
    try:
        return load_dataloaders(DATA_DIR, batch_size, num_workers)
    except FileNotFoundError:
        print(f"Datasets not found in {DATA_DIR}")
        print("Building datasets... (this may take a few minutes)")
        build_and_save_datasets(TICKERS, DATA_DIR, seq_len=SEQ_LEN)
        return load_dataloaders(DATA_DIR, batch_size, num_workers)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression LSTM on Price+News datasets")
    # Data & Loader
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Directory of saved datasets")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    # Model
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    # Optimization
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load loaders
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # Inspect a batch
    xb, yb, *meta = next(iter(train_loader))
    print(f"\nBatch shapes: X={xb.shape}, y={yb.shape}")
    print(f"Features: {xb.shape[-1]} dimensions")
    print(f"Sequence length: {xb.shape[1]} timesteps")
    
    
    # Model
    model = PriceNewsLSTMReg(
        input_size=xb.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(model)