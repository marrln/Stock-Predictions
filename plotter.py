"""Module for plotting stock data and model predictions."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from torchsummary import summary
import torch

def plot_training_history(history: Dict[str, list], save_path: Optional[Path] = None) -> None:
    """Plot training history metrics."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training and Validation Loss')
    
    # MAE plot
    axes[0, 1].plot(history['val_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    
    # RMSE plot
    axes[1, 0].plot(history['val_rmse'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE')
    
    # Learning rate plot
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def get_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Get a string summary of the model architecture and parameters."""
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        # Create a dummy input
        dummy_input = torch.randn(1, *input_shape)
        try:
            summary(model, input_shape, device="cpu")
        except ImportError:
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    return f.getvalue()