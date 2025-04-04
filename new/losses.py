import torch
import torch.nn as nn
import numpy as np

class DerivativeLoss(nn.Module):
    """
    Loss function that combines absolute error with derivative error.
    
    This loss encourages both accurate predictions and smooth transitions
    between consecutive predictions.
    """
    def __init__(self, alpha=0.5):
        """
        Args:
            alpha (float): Weight for derivative loss component (0-1)
                         Higher values give more weight to matching derivatives
        """
        super(DerivativeLoss, self).__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.prev_pred = None
        self.prev_target = None
    
    def forward(self, pred, target):
        """
        Calculate combined loss of MAE and derivative matching
        
        Args:
            pred (torch.Tensor): Predicted values [batch_size, features]
            target (torch.Tensor): Target values [batch_size, features]
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Calculate standard MAE loss
        mae_loss = self.mae(pred, target)
        
        # For the first batch, we can't calculate derivatives
        if self.prev_pred is None or self.prev_target is None:
            self.prev_pred = pred.detach().clone()
            self.prev_target = target.detach().clone()
            return mae_loss
        
        # Check if batch sizes match
        if pred.size(0) != self.prev_pred.size(0):
            # If batch sizes don't match, just use MAE loss and update previous values
            self.prev_pred = pred.detach().clone()
            self.prev_target = target.detach().clone()
            return mae_loss
        
        # Calculate derivative loss using current and previous predictions
        pred_diff = pred - self.prev_pred
        target_diff = target - self.prev_target
        
        derivative_loss = self.mae(pred_diff, target_diff)
        
        # Update previous values for next iteration
        self.prev_pred = pred.detach().clone()
        self.prev_target = target.detach().clone()
        
        # Combine the losses
        combined_loss = (1 - self.alpha) * mae_loss + self.alpha * derivative_loss
        
        return combined_loss

class WeightedMAELoss(nn.Module):
    """MAE loss with optional feature weights"""
    def __init__(self, feature_weights=None):
        """
        Args:
            feature_weights (torch.Tensor, optional): Weights for each feature
        """
        super(WeightedMAELoss, self).__init__()
        self.feature_weights = feature_weights
    
    def forward(self, pred, target):
        """
        Calculate weighted MAE loss
        
        Args:
            pred (torch.Tensor): Predicted values [batch, sequence_len, features]
            target (torch.Tensor): Target values [batch, sequence_len, features]
            
        Returns:
            torch.Tensor: Weighted MAE loss
        """
        mae = torch.abs(pred - target)
        
        if self.feature_weights is not None:
            # Apply feature weights
            mae = mae * self.feature_weights.to(pred.device)
        
        return torch.mean(mae)

class CombinedLoss(nn.Module):
    """Combines multiple loss functions with weights"""
    def __init__(self, losses, weights=None):
        """
        Args:
            losses (list): List of loss function instances
            weights (list, optional): List of weights for each loss function
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)
    
    def forward(self, pred, target):
        """
        Calculate weighted combination of losses
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Combined loss value
        """
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss

def create_derivative_loss(alpha=0.5):
    """
    Factory function to create a derivative loss instance
    
    Args:
        alpha (float): Weight for derivative component
        
    Returns:
        DerivativeLoss: Loss function instance
    """
    return DerivativeLoss(alpha=alpha)

def create_weighted_mae_loss(feature_weights=None):
    """
    Factory function to create a weighted MAE loss instance
    
    Args:
        feature_weights (torch.Tensor, optional): Weights for each feature
        
    Returns:
        WeightedMAELoss: Loss function instance
    """
    return WeightedMAELoss(feature_weights=feature_weights)

def create_combined_loss(alpha_derivative=0.3, feature_weights=None):
    """
    Factory function to create a combined loss with derivative and weighted MAE
    
    Args:
        alpha_derivative (float): Weight for derivative component
        feature_weights (torch.Tensor, optional): Weights for each feature
        
    Returns:
        CombinedLoss: Combined loss function instance
    """
    derivative_loss = DerivativeLoss(alpha=alpha_derivative)
    weighted_mae = WeightedMAELoss(feature_weights=feature_weights)
    
    return CombinedLoss(
        losses=[derivative_loss, weighted_mae],
        weights=[0.5, 0.5]
    ) 