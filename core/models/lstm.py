"""LSTM model for price-news regression task."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class PriceNewsLSTMReg(nn.Module):
    """Regression-only LSTM model for predicting next-day returns.
    
    Parameters:
        input_size: Number of features per timestep
        hidden_size: LSTM hidden units
        num_layers: Number of stacked LSTM layers
        dropout: Dropout applied between LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        pooling: How to aggregate LSTM outputs ('last', 'mean', 'max')
        num_tickers: Number of unique tickers for embedding (optional)
        ticker_emb_dim: Dimension of ticker embedding
        expansion_factor: Factor to expand input features before LSTM
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pooling: str = "last",
        num_tickers: Optional[int] = None,
        ticker_emb_dim: int = 16,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.expansion_factor = expansion_factor
        
        self.num_tickers = num_tickers
        self.ticker_emb_dim = ticker_emb_dim if num_tickers is not None else 0
        
        # Feature expansion layer (skip if expansion_factor == 1)
        if expansion_factor > 1:
            expanded_size = input_size * expansion_factor
            self.feature_expansion = nn.Sequential(
                nn.Linear(input_size, expanded_size),
                nn.BatchNorm1d(expanded_size),
                nn.Tanh(),  # Tanh preserves negative values (better for financial data than ReLU)
                nn.Dropout(dropout * 0.5),
            )
            lstm_input_size = expanded_size
        else:
            self.feature_expansion = None
            lstm_input_size = input_size
        
        # LSTM layer (receives either raw or expanded features)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Output dimension after pooling
        out_dim = hidden_size * (2 if bidirectional else 1)
        
        # Ticker embedding (optional)
        if self.num_tickers is not None and self.num_tickers > 0:
            self.ticker_emb = nn.Embedding(self.num_tickers, self.ticker_emb_dim)
            reg_in_dim = out_dim + self.ticker_emb_dim
        else:
            self.ticker_emb = None
            reg_in_dim = out_dim
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(reg_in_dim, reg_in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reg_in_dim, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # Initialize expansion layer weights if present
        if self.feature_expansion is not None:
            nn.init.xavier_uniform_(self.feature_expansion[0].weight)  # Changed from kaiming (ReLU) to xavier (Tanh)
    
    def forward(
        self, 
        x: torch.Tensor, 
        ticker_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            ticker_idx: Optional tensor of ticker indices for embedding
            
        Returns:
            Predicted values of shape (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply feature expansion if present
        if self.feature_expansion is not None:
            x_reshaped = x.view(-1, self.input_size)  # (batch_size * seq_len, input_size)
            x_expanded = self.feature_expansion(x_reshaped)
            x_expanded = x_expanded.view(batch_size, seq_len, -1)
            lstm_input = x_expanded
        else:
            lstm_input = x
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)  # (batch_size, seq_len, hidden_size * directions)
        
        # Pooling
        if self.pooling == 'last':
            features = lstm_out[:, -1, :]
        elif self.pooling == 'mean':
            features = lstm_out.mean(dim=1)
        elif self.pooling == 'max':
            features, _ = lstm_out.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Add ticker embedding if provided
        if self.ticker_emb is not None and ticker_idx is not None:
            emb = self.ticker_emb(ticker_idx)
            features = torch.cat([features, emb], dim=-1)
        
        # Regression output
        output = self.reg_head(features)
        return output.squeeze(-1)
    
    @property
    def total_parameters(self) -> int:
        """Total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def trainable_parameters(self) -> int:
        """Number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)