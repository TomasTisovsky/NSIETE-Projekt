"""
Optimizers: SGD, SGDMomentum, RMSprop, Adam.
All implemented using numpy only.
"""

import numpy as np


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self):
        pass
    
    def step(self, model):
        """Update model parameters."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, lr: float):
        super(SGD, self).__init__()
        self.lr = lr
    
    def step(self, model):
        """Update parameters using SGD."""
        for name, layer in model.modules.items():
            if hasattr(layer, 'get_optimizer_context'):
                params = layer.get_optimizer_context()
                if params is not None:
                    [[W, dW], [b, db]] = params
                    
                    # Update parameters
                    W = W - self.lr * dW
                    b = b - self.lr * db
                    
                    layer.set_optimizer_context([W, b])


class SGDMomentum(Optimizer):
    """SGD with Momentum optimizer."""
    
    def __init__(self, lr: float, beta: float = 0.9):
        super(SGDMomentum, self).__init__()
        self.lr = lr
        self.beta = beta
        self.context = {}
    
    def step(self, model):
        """Update parameters using SGD with momentum."""
        for name, layer in model.modules.items():
            if hasattr(layer, 'get_optimizer_context'):
                params = layer.get_optimizer_context()
                if params is not None:
                    [[W, dW], [b, db]] = params
                    
                    # Initialize momentum buffers
                    if name not in self.context:
                        self.context[name] = {
                            "vW": np.zeros_like(W),
                            "vb": np.zeros_like(b)
                        }
                    
                    # Update velocity (exponential moving average)
                    self.context[name]["vW"] = self.beta * self.context[name]["vW"] + (1 - self.beta) * dW
                    self.context[name]["vb"] = self.beta * self.context[name]["vb"] + (1 - self.beta) * db
                    
                    # Update parameters
                    W = W - self.lr * self.context[name]["vW"]
                    b = b - self.lr * self.context[name]["vb"]
                    
                    layer.set_optimizer_context([W, b])


class RMSprop(Optimizer):
    """RMSprop optimizer (Root Mean Square Propagation)."""
    
    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        super(RMSprop, self).__init__()
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.context = {}
    
    def step(self, model):
        """Update parameters using RMSprop."""
        for name, layer in model.modules.items():
            if hasattr(layer, 'get_optimizer_context'):
                params = layer.get_optimizer_context()
                if params is not None:
                    [[W, dW], [b, db]] = params
                    
                    # Initialize cache buffers
                    if name not in self.context:
                        self.context[name] = {
                            "sW": np.zeros_like(W),
                            "sb": np.zeros_like(b)
                        }
                    
                    # Update cache (squared gradient exponential moving average)
                    self.context[name]["sW"] = self.beta * self.context[name]["sW"] + (1 - self.beta) * (dW ** 2)
                    self.context[name]["sb"] = self.beta * self.context[name]["sb"] + (1 - self.beta) * (db ** 2)
                    
                    # Update parameters
                    W = W - self.lr * dW / (np.sqrt(self.context[name]["sW"]) + self.eps)
                    b = b - self.lr * db / (np.sqrt(self.context[name]["sb"]) + self.eps)
                    
                    layer.set_optimizer_context([W, b])


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""
    
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super(Adam, self).__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.context = {}
        self.t = 0
    
    def step(self, model):
        """Update parameters using Adam."""
        self.t += 1
        
        for name, layer in model.modules.items():
            if hasattr(layer, 'get_optimizer_context'):
                params = layer.get_optimizer_context()
                if params is not None:
                    [[W, dW], [b, db]] = params
                    
                    # Initialize moment buffers
                    if name not in self.context:
                        self.context[name] = {
                            "mW": np.zeros_like(W),  # 1st moment
                            "mb": np.zeros_like(b),
                            "vW": np.zeros_like(W),  # 2nd moment
                            "vb": np.zeros_like(b)
                        }
                    
                    # Update biased moments
                    self.context[name]["mW"] = self.beta1 * self.context[name]["mW"] + (1 - self.beta1) * dW
                    self.context[name]["mb"] = self.beta1 * self.context[name]["mb"] + (1 - self.beta1) * db
                    self.context[name]["vW"] = self.beta2 * self.context[name]["vW"] + (1 - self.beta2) * (dW ** 2)
                    self.context[name]["vb"] = self.beta2 * self.context[name]["vb"] + (1 - self.beta2) * (db ** 2)
                    
                    # Bias correction
                    mW_hat = self.context[name]["mW"] / (1 - self.beta1 ** self.t)
                    mb_hat = self.context[name]["mb"] / (1 - self.beta1 ** self.t)
                    vW_hat = self.context[name]["vW"] / (1 - self.beta2 ** self.t)
                    vb_hat = self.context[name]["vb"] / (1 - self.beta2 ** self.t)
                    
                    # Update parameters
                    W = W - self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
                    b = b - self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
                    
                    layer.set_optimizer_context([W, b])
