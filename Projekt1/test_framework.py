#!/usr/bin/env python
"""Quick test of the MLP framework."""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.model import Model
from src.layers import Linear
from src.activations import ReLU, Sigmoid
from src.losses import BCELoss
from src.optimizers import Adam
from src.utils import print_model_summary, set_seed

set_seed(42)

print("="*60)
print("MLP Framework Test")
print("="*60)

# 1. Create model
print("\n1. Building model...")
model = Model()
model.add_module(Linear(10, 32), 'linear1')
model.add_module(ReLU(), 'relu1')
model.add_module(Linear(32, 16), 'linear2')
model.add_module(ReLU(), 'relu2')
model.add_module(Linear(16, 1), 'linear3')
model.add_module(Sigmoid(), 'sigmoid')

print_model_summary(model)

# 2. Test forward pass
print("\n2. Testing forward pass...")
X = np.random.randn(10, 8).astype(np.float32)  # 10 features, 8 batch size
y = np.random.randint(0, 2, (1, 8)).astype(np.float32)  # Binary targets

output = model.forward(X)
print(f"   Input shape: {X.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
assert output.shape == (1, 8), f"Expected (1, 8), got {output.shape}"

# 3. Test loss
print("\n3. Testing loss function...")
loss_fn = BCELoss()
loss = loss_fn.forward(output, y)
print(f"   BCELoss: {loss:.4f}")

# 4. Test backward pass
print("\n4. Testing backward pass...")
dA = loss_fn.backward(output, y)
dx = model.backward(dA)
print(f"   Gradient shape: {dx.shape}")
assert dx.shape == X.shape, f"Expected {X.shape}, got {dx.shape}"

# 5. Test optimizer
print("\n5. Testing optimizer...")
optimizer = Adam(lr=0.001)
optimizer.step(model)
print(f"   ✓ Adam optimizer step completed")

# 6. Check parameters updated
print("\n6. Verifying parameter updates...")
output_after = model.forward(X)
print(f"   Output changed after optimizer step: {not np.allclose(output, output_after)}")

print("\n" + "="*60)
print("✓ All framework tests passed!")
print("="*60)
