"""LSTM_model for price-news regression task.
Usage example:
	model = PriceNewsLSTMReg(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PriceNewsLSTMReg(nn.Module):
	"""Regression-only LSTM model.
	Predicts a single continuous value (e.g., next-day return).
	Parameters:
		input_size: number of features per timestep
		hidden_size: LSTM hidden units
		num_layers: number of stacked LSTM layers
		dropout: dropout applied between LSTM layers
		bidirectional: whether to use a bidirectional LSTM
		pooling: how to aggregate LSTM outputs ('last'|'mean'|'max')
	"""

	def __init__(
		self,
		input_size: int,
		hidden_size: int = 128,
		num_layers: int = 2,
		dropout: float = 0.2,
		bidirectional: bool = False,
		pooling: str = "last",
	) -> None:
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.bidirectional = bidirectional
		self.pooling = pooling

		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
			bidirectional=bidirectional,
		)

		out_dim = hidden_size * (2 if bidirectional else 1)
		self.reg_head = nn.Sequential(
			nn.Linear(out_dim, out_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(out_dim, 1),
		)

		# Weight initialization for better convergence
		for n, p in self.named_parameters():
			if 'weight_ih' in n:
				nn.init.xavier_uniform_(p)
			elif 'weight_hh' in n:
				nn.init.orthogonal_(p)
			elif 'bias' in n:
				nn.init.constant_(p, 0.0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.lstm(x)

		if self.pooling == 'last':
			feat = out[:, -1, :]
		elif self.pooling == 'mean':
			feat = out.mean(dim=1)
		elif self.pooling == 'max':
			feat, _ = out.max(dim=1)
		else:
			raise ValueError("pooling must be one of 'last', 'mean', 'max'")

		reg = self.reg_head(feat)
		return reg.squeeze(-1)
