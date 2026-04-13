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
        if input.ndim != 2:
            raise ValueError(f"Linear.forward expects 2D input, got shape {input.shape}")
        if input.shape[0] != self.in_features:
            raise ValueError(
                f"Linear.forward expected input with {self.in_features} features, got {input.shape[0]}"
            )

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
        if dz.ndim != 2:
            raise ValueError(f"Linear.backward expects 2D gradient, got shape {dz.shape}")
        if dz.shape[0] != self.out_features:
            raise ValueError(
                f"Linear.backward expected gradient with {self.out_features} outputs, got {dz.shape[0]}"
            )

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


class Dropout(Module):
    """
    Dropout regularization layer.

    During training it drops activations with probability p and rescales
    surviving activations by 1 / (1 - p).
    """

    def __init__(self, p: float = 0.2):
        super(Dropout, self).__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.mask = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Apply dropout mask in training mode, identity in eval mode."""
        if not self.training or self.p == 0.0:
            self.mask = None
            return input

        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*input.shape) < keep_prob).astype(input.dtype) / keep_prob
        return input * self.mask

    def backward(self, da: np.ndarray) -> np.ndarray:
        """Propagate gradients through the same dropout mask."""
        if not self.training or self.p == 0.0 or self.mask is None:
            return da
        return da * self.mask
