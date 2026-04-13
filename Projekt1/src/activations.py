"""
Activation functions with forward and backward passes.
"""

import numpy as np
from .base import Module


class Sigmoid(Module):
    """Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))"""
    
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.fw_input = None
        self.output = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.fw_input = input
        self.output = 1.0 / (1.0 + np.exp(-input))
        return self.output
    
    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: da * sigmoid(x) * (1 - sigmoid(x))"""
        dz = da * self.output * (1 - self.output)
        return dz


class Tanh(Module):
    """Hyperbolic tangent activation."""
    
    def __init__(self):
        super(Tanh, self).__init__()
        self.fw_input = None
        self.output = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.fw_input = input
        self.output = (np.exp(2 * input) - 1) / (np.exp(2 * input) + 1)
        return self.output
    
    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: da * (1 - tanh(x)^2)"""
        dz = da * (1 - self.output ** 2)
        return dz


class ReLU(Module):
    """Rectified Linear Unit: max(0, x)"""
    
    def __init__(self):
        super(ReLU, self).__init__()
        self.fw_input = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.fw_input = input
        return np.maximum(input, 0)
    
    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: da * (fw_input > 0)"""
        dz = da * (self.fw_input > 0)
        return dz


class LeakyReLU(Module):
    """Leaky ReLU activation: max(alpha*x, x)"""
    
    def __init__(self, alpha: float = 0.01):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.fw_input = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.fw_input = input
        return np.maximum(self.alpha * input, input)
    
    def backward(self, da: np.ndarray) -> np.ndarray:
        """Backward pass: da * (alpha if x < 0 else 1)"""
        dz = da * np.where(self.fw_input < 0, self.alpha, 1.0)
        return dz
