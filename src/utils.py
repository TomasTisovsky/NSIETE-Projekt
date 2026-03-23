"""
Utility functions.
"""

import numpy as np
import json
from pathlib import Path


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def save_config(config: dict, path: str):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_model_from_config(config: dict):
    """
    Create model from configuration dictionary.
    
    Args:
        config: dict with 'architecture' key containing list of layer configs
    
    Returns:
        Model instance
    """
    from .model import Model
    from .layers import Linear
    from .activations import Sigmoid, Tanh, ReLU, LeakyReLU
    
    activations_map = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
    }
    
    model = Model()
    arch = config.get('architecture', [])
    
    for i, layer_config in enumerate(arch):
        layer_type = layer_config.get('type')
        
        if layer_type == 'linear':
            in_features = layer_config.get('in_features')
            out_features = layer_config.get('out_features')
            model.add_module(Linear(in_features, out_features), f'linear_{i}')
        
        elif layer_type == 'activation':
            activation_name = layer_config.get('name', 'relu')
            activation_class = activations_map.get(activation_name, ReLU)
            
            if activation_name == 'leaky_relu':
                alpha = layer_config.get('alpha', 0.01)
                model.add_module(activation_class(alpha), f'{activation_name}_{i}')
            else:
                model.add_module(activation_class(), f'{activation_name}_{i}')
    
    return model


def print_model_summary(model):
    """Print model architecture summary."""
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    
    total_params = 0
    
    for name, module in model.modules.items():
        module_type = module.__class__.__name__
        print(f"{name:30s} | {module_type}")
        
        # Count parameters
        if hasattr(module, 'W'):
            n_params = module.W.size + module.b.size
            total_params += n_params
            print(f"{'':30s} | Params: {n_params} "
                  f"(W: {module.W.shape}, b: {module.b.shape})")
    
    print("="*60)
    print(f"Total trainable parameters: {total_params}")
    print("="*60 + "\n")
