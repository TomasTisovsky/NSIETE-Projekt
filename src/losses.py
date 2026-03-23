"""
Loss function with forward and backward passes.
"""

import numpy as np
from .base import Module


class BCELoss(Module):
    """Binary Cross Entropy loss - numerically stable version."""
    
    def __init__(self, reduce: str = "mean"):
        super(BCELoss, self).__init__()
        if reduce == "mean":
            self.reduce_fn = np.mean
        elif reduce == "sum":
            self.reduce_fn = np.sum
        elif reduce is None:
            self.reduce_fn = lambda x: x
        else:
            raise ValueError(f"reduce must be 'mean', 'sum', or None, got {reduce}")
        
        self.eps = 1e-15
    
    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute binary cross entropy loss.
        Numerically stable with clipping.
        """
        # Clip predictions to prevent log(0)
        input_clipped = np.clip(input, self.eps, 1 - self.eps)
        
        loss = -(target * np.log(input_clipped) + (1 - target) * np.log(1 - input_clipped))
        return self.reduce_fn(loss)
    
    def backward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute gradient of BCE loss."""
        input_clipped = np.clip(input, self.eps, 1 - self.eps)
        dz = (input_clipped - target) / (input_clipped * (1 - input_clipped))
        return dz
