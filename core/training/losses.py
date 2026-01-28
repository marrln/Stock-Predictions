"""
Custom loss functions for financial forecasting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Loss that penalizes wrong-direction predictions more heavily than wrong-magnitude.
    
    Commonly used in financial forecasting where getting the sign right (up/down)
    is more important than predicting the exact magnitude.
    
    Args:
        base_loss: Base loss function ('mse' or 'mae')
        direction_weight: How much more to penalize wrong-direction predictions (default: 2.0)
        direction_penalty: Additional penalty for wrong direction (default: 0.0)
    """
    def __init__(self, base_loss='mse', direction_weight=2.0, direction_penalty=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.direction_weight = direction_weight
        self.direction_penalty = direction_penalty
        
    def forward(self, pred, target):
        # Compute base loss (MSE or MAE)
        if self.base_loss == 'mse':
            loss = (pred - target) ** 2
        elif self.base_loss == 'mae':
            loss = torch.abs(pred - target)
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")
        
        # Check if prediction and target have same sign (same direction)
        same_direction = (pred * target) > 0
        
        # Weight losses: penalize wrong-direction predictions more heavily
        weights = torch.where(
            same_direction, 
            torch.ones_like(loss),
            self.direction_weight * torch.ones_like(loss)
        )
        
        weighted_loss = loss * weights
        
        # Optional: Add extra penalty for wrong direction
        if self.direction_penalty > 0:
            direction_penalty_term = torch.where(
                same_direction,
                torch.zeros_like(loss),
                self.direction_penalty * torch.ones_like(loss)
            )
            weighted_loss = weighted_loss + direction_penalty_term
        
        return weighted_loss.mean()


class SignPenaltyLoss(nn.Module):
    """
    MSE loss with explicit sign penalty.
    
    Loss = MSE + alpha * sign_penalty
    where sign_penalty = 1 if pred and target have different signs, 0 otherwise
    
    Args:
        alpha: Weight for sign penalty (default: 1.0)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # MSE component
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Sign penalty: 1 if different signs, 0 if same
        sign_penalty = ((pred * target) < 0).float()
        
        # Combined loss
        loss = mse + self.alpha * sign_penalty
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes overestimation and underestimation differently.
    
    Useful when the cost of false positives differs from false negatives
    (e.g., predicting large gain when stock actually drops).
    
    Args:
        alpha: Weight for overestimation error (pred > target) (default: 1.0)
        beta: Weight for underestimation error (pred < target) (default: 1.0)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        error = pred - target
        
        # Overestimation (pred > target): use alpha weight
        # Underestimation (pred < target): use beta weight
        overestimation_loss = self.alpha * torch.relu(error) ** 2
        underestimation_loss = self.beta * torch.relu(-error) ** 2
        
        return (overestimation_loss + underestimation_loss).mean()


class QuantileLoss(nn.Module):
    """
    Quantile (pinball) loss for predicting specific quantiles.
    
    Args:
        quantile: The quantile to predict (default: 0.5 for median)
    """
    def __init__(self, quantile=0.5):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1"
        self.quantile = quantile
        
    def forward(self, pred, target):
        error = target - pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return loss.mean()
