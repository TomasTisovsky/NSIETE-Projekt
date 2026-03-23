"""
Base Module class for neural network building blocks.
Similar to PyTorch's Module - serves as the elementary building block.
"""

import numpy as np
from collections import OrderedDict


class Module:
    """Base class for all neural network modules (layers, activations, losses)."""
    
    def __init__(self):
        self.modules = OrderedDict()
        self.training = True
    
    def add_module(self, module, name: str):
        """Add a submodule to this module."""
        if hasattr(self, name) and name not in self.modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError("module name can't contain '.'")
        elif name == '':
            raise KeyError("module name can't be empty string")
        self.modules[name] = module
    
    def forward(self, *args, **kwargs) -> np.ndarray:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        """Backward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        """Allow calling module as a function."""
        return self.forward(*args, **kwargs)

    def train(self):
        """Set module and all submodules to training mode."""
        self.training = True
        for module in self.modules.values():
            if hasattr(module, "train"):
                module.train()
        return self

    def eval(self):
        """Set module and all submodules to evaluation mode."""
        self.training = False
        for module in self.modules.values():
            if hasattr(module, "eval"):
                module.eval()
        return self
    
    def get_optimizer_context(self):
        """Return parameter gradients for optimizer. To be implemented by layer classes."""
        return None
    
    def set_optimizer_context(self, params):
        """Set updated parameters from optimizer. To be implemented by layer classes."""
        pass
