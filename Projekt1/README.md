# MLP for MAGIC Gamma Telescope Dataset

A from-scratch implementation of a Multi-Layer Perceptron neural network for the MAGIC Gamma Telescope binary classification dataset.

## Key Features

- **No deep learning frameworks**: Pure numpy implementation
- **Educational style**: Clean, modular code inspired by PyTorch's API
- **Full training pipeline**: Layers, activations, losses, optimizers, training loop
- **Multiple optimizers**: SGD, SGDMomentum, RMSprop, Adam
- **Improvement technique included**: Dropout regularization layer
- **Comprehensive metrics**: Accuracy, precision, recall, F1, ROC curves
- **Experiment tracking**: Multi-run comparison saved to CSV

## Project Structure

```
project/
├── src/
│   ├── base.py           # Base Module class
│   ├── layers.py         # Linear layer
│   ├── activations.py    # Sigmoid, Tanh, ReLU, LeakyReLU
│   ├── losses.py         # MSELoss, BCELoss
│   ├── model.py          # Sequential Model container
│   ├── optimizers.py     # SGD, SGDMomentum, RMSprop, Adam
│   ├── preprocessing.py  # Data loading and preprocessing
│   ├── metrics.py        # Evaluation metrics
│   ├── trainer.py        # Training loop
│   ├── experiments.py    # Multi-experiment runner and CSV logging
│   ├── utils.py          # Helper utilities
│   └── visualization.py  # Plotting utilities
├── notebooks/
│   ├── eda-telescope.ipynb
│   └── training_and_experiments.ipynb
├── data/
├── results/
│   ├── plots/
│   └── logs/
├── requirements.txt
└── README.md
```

## Dataset

**MAGIC Gamma Telescope**: Binary classification dataset to distinguish gamma rays from cosmic background noise.
- Features: 10 numeric attributes from telescope observations
- Target: 'g' (gamma) or 'h' (hadron)
- Classes: Mapped to 1 (gamma) and 0 (hadron)

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

The dataset is automatically loaded from UCI ML Repository in the notebooks.

### 3. Run training notebook

```bash
jupyter notebook notebooks/training_and_experiments.ipynb

### 4. Run tracked experiments

The notebook section "Multi-Experiment Tracking" runs predefined configurations
and writes outputs to `results/experiments.csv`.
```

## Model Architecture

Default baseline model:
```
Input (10 features)
  ↓
Linear(10 → 32)
  ↓
ReLU
  ↓
Linear(32 → 16)
  ↓
ReLU
  ↓
Linear(16 → 1)
  ↓
Sigmoid
  ↓
Output (0 or 1)
```

## Implementation Details

### Module System
Each layer/activation/loss is a `Module` with:
- `forward()`: forward pass computation
- `backward()`: gradient computation
- Caching of intermediate values for backward pass

### Optimizers
All optimizers work with the module system:
- Track parameter gradients and optimizer state
- Support momentum, RMSprop, and Adam variants
- Update parameters via `layer.set_optimizer_context()`

### Training Loop
Standard mini-batch SGD:
1. Forward pass through all layers
2. Compute loss
3. Backward pass to compute gradients
4. Optimizer step to update parameters
5. Repeat with new mini-batch

## Example Usage

```python
from src.model import Model
from src.layers import Linear
from src.activations import ReLU, Sigmoid
from src.losses import BCELoss
from src.optimizers import Adam
from src.trainer import Trainer
from src.preprocessing import load_magic_dataset, preprocess_data

# Load and preprocess data
X, y = load_magic_dataset()
data = preprocess_data(X, y)

# Build model
model = Model()
model.add_module(Linear(10, 32), 'linear1')
model.add_module(ReLU(), 'relu1')
model.add_module(Linear(32, 16), 'linear2')
model.add_module(ReLU(), 'relu2')
model.add_module(Linear(16, 1), 'linear3')
model.add_module(Sigmoid(), 'sigmoid')

# Setup training
loss_fn = BCELoss()
optimizer = Adam(lr=0.001)
trainer = Trainer(model, loss_fn, optimizer, batch_size=32)

# Train
history = trainer.fit(
    data['train']['X'], data['train']['y'],
    data['val']['X'], data['val']['y'],
    epochs=100
)

# Evaluate
_, test_metrics = trainer.evaluate(data['test']['X'], data['test']['y'])
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1: {test_metrics['f1']:.4f}")
```

## Code Style

- No TensorFlow, PyTorch, Keras, or JAX
- Pure numpy for all computations
- Numpy-only, no external ML frameworks
- Follows university course patterns (Module, Linear, forward/backward)
- Educational implementation, not production-grade