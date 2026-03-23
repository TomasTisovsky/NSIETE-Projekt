"""
Layer implementations - Linear (Fully Connected) layer.
Uses numpy only for all computations.
"""

import numpy as np
from .base import Module


class Linear(Module):
    """
    Linear layer (Fully Connected, Dense, Single Layer Perceptron).
    
    Args:
        in_features: number of input features
        out_features: number of output features
    """
    
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(out_features, in_features))
        self.b = np.zeros((out_features, 1))
        
        # Gradient buffers
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self.fw_inputs = None
        self.m = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass: output = W @ input + b
        
        Args:
            input: shape (in_features, batch_size)
        
        Returns:
            output: shape (out_features, batch_size)
        """
        self.fw_inputs = input
        self.m = input.shape[1] if len(input.shape) > 1 else 1
        net = np.matmul(self.W, input) + self.b
        return net
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients and propagate to previous layer.
        
        Args:
            dz: gradient w.r.t. output, shape (out_features, batch_size)
        
        Returns:
            dx: gradient w.r.t. input, shape (in_features, batch_size)
        """
        # Compute gradients
        self.dW = (1.0 / self.m) * np.matmul(dz, self.fw_inputs.T)
        self.db = (1.0 / self.m) * np.sum(dz, axis=1, keepdims=True)
        
        # Propagate gradient to previous layer
        dx = np.matmul(self.W.T, dz)
        return dx
    
    def get_optimizer_context(self):
        """Return parameters and gradients for optimizer."""
        return [[self.W, self.dW], [self.b, self.db]]
    
    def set_optimizer_context(self, params):
        """Update parameters from optimizer."""
        self.W, self.b = params
