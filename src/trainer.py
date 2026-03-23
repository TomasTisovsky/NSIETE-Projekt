"""
Training loop utilities.
"""

import numpy as np
from .metrics import evaluate_model, threshold_predictions


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, loss_fn, optimizer, batch_size=32):
        """
        Initialize trainer.
        
        Args:
            model: neural network model
            loss_fn: loss function
            optimizer: optimizer
            batch_size: batch size
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def train_epoch(self, X_train, y_train, shuffle=True):
        """
        Train for one epoch.
        
        Args:
            X_train: training features, shape (n_samples, n_features)
            y_train: training targets, shape (n_samples,)
            shuffle: whether to shuffle batches
        
        Returns:
            average loss for epoch
        """
        self.model.train()
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        epoch_loss = []
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X_train[batch_indices].T  # (n_features, batch_size)
            y_batch = y_train[batch_indices].reshape(1, -1)  # (1, batch_size)
            
            # Forward pass
            y_pred = self.model.forward(X_batch)
            
            # Compute loss
            loss = self.loss_fn.forward(y_pred, y_batch)
            epoch_loss.append(float(loss))
            
            # Backward pass
            dA = self.loss_fn.backward(y_pred, y_batch)
            self.model.backward(dA)
            
            # Optimizer step
            self.optimizer.step(self.model)
        
        return np.mean(epoch_loss)
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate model on data.
        
        Args:
            X: features, shape (n_samples, n_features)
            y: targets, shape (n_samples,)
            threshold: decision threshold
        
        Returns:
            tuple (loss, metrics_dict)
        """
        self.model.eval()
        X_T = X.T  # (n_features, n_samples)
        y_T = y.reshape(1, -1)  # (1, n_samples)
        
        # Forward pass
        y_pred = self.model.forward(X_T)
        
        # Loss
        loss = self.loss_fn.forward(y_pred, y_T)
        
        # Metrics
        metrics = evaluate_model(y.flatten(), y_pred, threshold=threshold)
        
        return float(loss), metrics
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, early_stopping=True, patience=10):
        """
        Train model.
        
        Args:
            X_train: training features
            y_train: training targets
            X_val: validation features
            y_val: validation targets
            epochs: number of epochs
            early_stopping: whether to use early stopping
            patience: patience for early stopping
        
        Returns:
            training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, shuffle=True)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.evaluate(X_val, y_val)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Train metrics
            _, train_metrics = self.evaluate(X_train, y_train)
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {train_loss:.4f} - "
                      f"val_loss: {val_loss:.4f} - "
                      f"acc: {train_metrics['accuracy']:.4f} - "
                      f"val_acc: {val_metrics['accuracy']:.4f}")
        
        return self.history
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            probabilities, shape (n_samples,)
        """
        self.model.eval()
        X_T = X.T  # (n_features, n_samples)
        y_pred = self.model.forward(X_T)
        return y_pred.flatten()
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Args:
            X: features, shape (n_samples, n_features)
            threshold: decision threshold
        
        Returns:
            class labels, shape (n_samples,)
        """
        y_proba = self.predict_proba(X)
        return threshold_predictions(y_proba.reshape(1, -1), threshold)
