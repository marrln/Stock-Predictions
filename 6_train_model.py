"""6_train_model.py

Training script for stock price prediction LSTM model.
Loads preprocessed datasets and prepares them for training.
"""

import torch
from PriceNewsDataset import load_dataloaders, build_and_save_datasets


# Configuration
DATA_DIR = "processed_data/small_ams"  # Where datasets are saved
BATCH_SIZE = 64
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

# Tickers to use (only needed if building datasets from scratch)
TICKERS = ["AAPL", "MSFT", "AMZN"]
SEQ_LEN = 8


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """Load dataloaders from saved datasets.
    
    If datasets don't exist, uncomment the build_datasets() call below to create them.
    """
    try:
        return load_dataloaders(DATA_DIR, batch_size, num_workers)
    except FileNotFoundError:
        print(f"Datasets not found in {DATA_DIR}")
        print("Building datasets... (this may take a few minutes)")
        build_and_save_datasets(TICKERS, DATA_DIR, seq_len=SEQ_LEN)
        return load_dataloaders(DATA_DIR, batch_size, num_workers)


if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Inspect a batch
    xb, yb, *meta = next(iter(train_loader))
    print(f"\nBatch shapes: X={xb.shape}, y={yb.shape}")
    print(f"Features: {xb.shape[-1]} dimensions")
    print(f"Sequence length: {xb.shape[1]} timesteps")
    
    # TODO: Add LSTM model definition and training loop here
