"""LSTM_model

Regression-only LSTM for stock sequences (predicts next-day return).

Works with dataloaders produced by PriceNewsDataset.py.

Usage example:
	from PriceNewsDataset import load_dataloaders
	from LSTM_model import PriceNewsLSTMReg

	train_loader, val_loader, test_loader = load_dataloaders()
	input_size = train_loader.dataset.X.shape[-1]
	model = PriceNewsLSTMReg(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PriceNewsLSTMReg(nn.Module):
	"""Regression-only LSTM model.

	Predicts a single continuous value (e.g., next-day return).

	Parameters:
		input_size: number of features per timestep
		hidden_size: LSTM hidden units
		num_layers: number of stacked LSTM layers
		dropout: dropout applied between LSTM layers
	"""

	def __init__(
		self,
		input_size: int,
		hidden_size: int = 128,
		num_layers: int = 2,
		dropout: float = 0.2,
	) -> None:
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout

		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
			bidirectional=False,
		)

		out_dim = hidden_size
		self.reg_head = nn.Sequential(
			nn.Linear(out_dim, out_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(out_dim, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.lstm(x)
		last = out[:, -1, :]
		reg = self.reg_head(last)
		return reg.squeeze(-1)
