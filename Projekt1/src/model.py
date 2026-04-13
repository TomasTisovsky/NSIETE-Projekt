"""
Model class - container for sequential neural network layers.
"""

import numpy as np
from .base import Module


class Model(Module):
    """
    Sequential model that applies layers in order.
    Supports forward and backward propagation.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass - apply all modules in order.
        
        Args:
            input: input data
        
        Returns:
            output after all layers
        """
        for name, module in self.modules.items():
            input = module(input)
        return input
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        Backward pass - propagate gradients in reverse order.
        
        Args:
            dz: upstream gradient
        
        Returns:
            gradient w.r.t. input
        """
        for name, module in reversed(list(self.modules.items())):
            dz = module.backward(dz)
        return dz
    
    def get_trainable_layers(self):
        """Return list of layers that have trainable parameters."""
        trainable = []
        for name, module in self.modules.items():
            if hasattr(module, 'get_optimizer_context') and module.get_optimizer_context() is not None:
                trainable.append((name, module))
        return trainable
