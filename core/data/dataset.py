"""Dataset classes for time series data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional


class TimeSeriesDataset(Dataset):
    """Simple Dataset wrapping precomputed sequences (X) and targets (y).
    
    Optionally carries a `meta` DataFrame with per-sample metadata.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        meta: Optional[pd.DataFrame] = None,
        feature_cols: Optional[list] = None,
        target_type: str = "return"
    ):
        assert len(X) == len(y), "X and y must have same length"
        if meta is not None:
            assert len(meta) == len(X), "meta must have same length as X"
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.meta = meta.reset_index(drop=True) if meta is not None else None
        self.feature_cols = feature_cols
        self.target_type = target_type

        # Normalize meta columns for backward compatibility with older scripts.
        # Create lowercase aliases (e.g., 'Ticker' -> 'ticker') and ensure
        # commonly-used capitalized keys ('Ticker','Date','TickerIdx') exist
        # when only lowercase versions are present.
        if self.meta is not None:
            # Create lowercase copies for any column that lacks a lowercase alias
            for col in list(self.meta.columns):
                low = col.lower()
                if low not in self.meta.columns:
                    self.meta[low] = self.meta[col]

            # Ensure commonly referenced capitalized columns exist when only lowercase versions are present
            if 'ticker' in self.meta.columns and 'Ticker' not in self.meta.columns:
                self.meta['Ticker'] = self.meta['ticker']
            if 'date' in self.meta.columns and 'Date' not in self.meta.columns:
                self.meta['Date'] = self.meta['date']
            if 'tickeridx' in self.meta.columns and 'TickerIdx' not in self.meta.columns:
                self.meta['TickerIdx'] = self.meta['tickeridx']
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        if self.meta is None:
            return self.X[idx], self.y[idx]
        
        # Return meta as plain dict for DataLoader compatibility
        md = self.meta.loc[idx].to_dict()
        
        # Convert Timestamp to ISO string
        if "Date" in md and not pd.isna(md["Date"]):
            md["Date"] = pd.to_datetime(md["Date"]).isoformat()
        
        return self.X[idx], self.y[idx], md
    
    @property
    def shape(self) -> tuple:
        """Return dataset shape (n_samples, seq_len, n_features)."""
        return self.X.shape