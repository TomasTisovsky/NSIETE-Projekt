"""
Data preprocessing utilities for MAGIC Telescope dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class PreprocessingConfig:
    """Configuration for data loading and preprocessing."""
    
    # UCI ML Repository ID for MAGIC Telescope
    UCI_DATASET_ID = 159
    
    # Class mapping
    CLASS_MAPPING = {'g': 1, 'h': 0}  # gamma -> 1, hadron -> 0
    
    # Train/val/test split
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    RANDOM_STATE = 42


def load_magic_dataset():
    """
    Load MAGIC Telescope dataset from UCI ML Repository.
    
    The dataset is automatically downloaded from:
    https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
    
    Returns:
        features: ndarray of shape (n_samples, n_features)
        targets: ndarray of shape (n_samples,)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Loading MAGIC Gamma Telescope dataset from UCI ML Repository...")
        
        # Fetch dataset
        magic = fetch_ucirepo(id=PreprocessingConfig.UCI_DATASET_ID)
        
        X = magic.data.features.values.astype(np.float32)
        y = magic.data.targets.iloc[:, 0].values
        
        # Map class labels: 'g' -> 1 (gamma), 'h' -> 0 (hadron)
        y_mapped = np.array([PreprocessingConfig.CLASS_MAPPING.get(val, val) for val in y], dtype=np.float32)
        
        print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y_mapped
    
    except ImportError:
        raise ImportError("ucimlrepo package required. Install with: pip install ucimlrepo")


def preprocess_data(X, y, normalize=True, random_state=None):
    """
    Split data into train/val/test and normalize.
    
    Args:
        X: features
        y: targets
        normalize: whether to standardize features
        random_state: for reproducibility
    
    Returns:
        dict with 'train', 'val', 'test' splits and 'scaler'
    """
    if random_state is None:
        random_state = PreprocessingConfig.RANDOM_STATE
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=PreprocessingConfig.TRAIN_RATIO,
        random_state=random_state,
        stratify=y
    )
    
    # Split temp into val and test
    val_ratio = PreprocessingConfig.VAL_RATIO / (PreprocessingConfig.VAL_RATIO + PreprocessingConfig.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Normalize using train statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_val, 'y': y_val},
        'test': {'X': X_test, 'y': y_test},
        'scaler': scaler
    }


def create_batches(X, y, batch_size, shuffle=True, random_state=None):
    """
    Create mini-batches from data.
    
    Args:
        X: features, shape (n_samples, n_features)
        y: targets, shape (n_samples,)
        batch_size: batch size
        shuffle: whether to shuffle batches
        random_state: for reproducibility
    
    Yields:
        (X_batch, y_batch) where shapes are (n_features, batch_size) and (1, batch_size)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle and random_state is not None:
        np.random.seed(random_state)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        X_batch = X[batch_indices].T  # (n_features, batch_size)
        y_batch = y[batch_indices].reshape(1, -1)  # (1, batch_size)
        yield X_batch.astype(np.float32), y_batch.astype(np.float32)
