"""Improved LSTM model - predicts normalized prices (not returns)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


@dataclass
class LSTMConfig:
    """Configuration for LSTM model.
        seq_len: Length of input sequences (including target timestep).
        n_features: Number of input features per timestep.
        lstm_units: List of integers specifying number of units in each LSTM layer.
        dropout: Dropout rate between layers.
        recurrent_dropout: Dropout rate for recurrent connections.
        l2_reg: L2 regularization factor.
        learning_rate: Learning rate for optimizer.
        optimizer: Optimizer type ('adam', 'sgd', etc.).
        loss: Loss function ('mse', 'mae', 'huber', etc.).
    """
    seq_len: int = 50
    n_features: int = 3 
    lstm_units: list[int] = None
    dropout: float = 0.2
    recurrent_dropout: float = 0.0 
    l2_reg: float = 0.0
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'mse'
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.lstm_units is None:
            # Consistent architecture (4 layers, 100 units each)
            self.lstm_units = [100, 100, 100, 100]

    def save_json(self, filepath: str, indent: int = 2):
        """Save the configuration to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path to write the JSON file.
        indent : int
            Indentation level for pretty-printing JSON (default: 2).
        """
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    @classmethod
    def load_json(cls, filepath: str = 'config.json') -> "LSTMConfig":
        """Load configuration from a JSON file and return `LSTMConfig`.

        Parameters
        ----------
        filepath : str
            Path to the JSON file (default: 'config.json').

        Returns
        -------
        LSTMConfig
            A new instance populated from the JSON file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter out unknown fields to avoid constructor errors
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)


def build_lstm_model(config: LSTMConfig) -> Model:
    """Build and compile LSTM model according to config.  
      
    Parameters
    ----------
    config : LSTMConfig
        Model configuration.
        Includes sequence length, number of features, LSTM layer sizes, dropout rates, optimizer, and loss.
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model.
    """
    
    # Input layer
    inputs = keras.Input(
        shape=(config.seq_len - 1, config.n_features),  # -1 because last timestep is target
        name='sequence_input'
    )
    
    # LSTM layers with consistent architecture
    x = inputs
    for i, units in enumerate(config.lstm_units):
        # All but last LSTM return sequences
        return_sequences = (i < len(config.lstm_units) - 1)
        
        # Add LSTM layer
        x = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=config.dropout if i == 0 else 0,  # Dropout only on first
            recurrent_dropout=config.recurrent_dropout,
            name=f'lstm_{i+1}'
        )(x)
        
        # Add dropout after certain layers
        # dropout after 2nd and 4th LSTM
        # TODO: Dropout every other layer for deeper nets?
        if i == 1 or i == len(config.lstm_units) - 1:
            x = layers.Dropout(config.dropout, name=f'dropout_{i+1}')(x)
    
    # Output layer (single value - normalized price)
    outputs = layers.Dense(1, activation='linear', name='output')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='ImprovedLSTM_FinancialForecasting')
    
    # Select optimizer
    if config.optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif config.optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)
    elif config.optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Select loss function
    if config.loss.lower() == 'mse':
        loss_fn = 'mse'
    elif config.loss.lower() == 'mae':
        loss_fn = 'mae'
    elif config.loss.lower() == 'huber':
        loss_fn = keras.losses.Huber()
    else:
        raise ValueError(f"Unknown loss: {config.loss}")
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=['mae', 'mse']
    )
    
    return model


class LSTMFinancialModel:
    """LSTM model that predicts normalized prices (not returns).
    
    --------
    >>> config = LSTMConfig(
    ...     seq_len=50,
    ...     n_features=3,
    ...     lstm_units=[100, 100, 100, 100]
    ... )
    >>> model = LSTMFinancialModel(config)
    >>> history = model.fit(X_train, y_train, X_val, y_val, epochs=20)
    """
    
    def __init__(self, config: LSTMConfig):
        """Initialize model with configuration."""
        self.config = config
        self.model = build_lstm_model(config)
        self.history = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        verbose: int = 1,
        **kwargs
    ):
        """Train the model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training sequences, shape (n_samples, seq_len-1, n_features).
        y_train : np.ndarray
            Training targets (normalized prices), shape (n_samples,).
        X_val : np.ndarray, optional
            Validation sequences.
        y_val : np.ndarray, optional
            Validation targets.
        epochs : int
            Number of training epochs (default: 20).
        batch_size : int
            Batch size (default: 32).
        callbacks : list, optional
            Keras callbacks.
        verbose : int
            Verbosity mode.
        
        Returns
        -------
        history : keras.callbacks.History
            Training history.
        """
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Use conservative callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=2,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences.
        batch_size : int
            Batch size for prediction.
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted normalized prices.
        """
        y_pred = self.model.predict(X, batch_size=batch_size, verbose=0)
        return y_pred.flatten()
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        verbose: int = 0
    ) -> dict:
        """Evaluate model on test data."""
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: Optional[LSTMConfig] = None):
        """Load model from disk."""
        loaded_model = keras.models.load_model(filepath)
        
        if config is None:
            config = LSTMConfig()
        
        wrapper = cls(config)
        wrapper.model = loaded_model
        
        print(f"Model loaded from {filepath}")
        return wrapper
    
    def summary(self):
        """Print model architecture summary."""
        self.model.summary()


def get_lstm_config(preset: str = 'standard') -> LSTMConfig:
    """Get predefined improved model configurations.
    
    Parameters
    ----------
    preset : str
        Configuration preset:
        - 'standard': Standard model (4x100 LSTM)
        - 'small': Smaller version (3x50 LSTM)
        - 'large': Larger version (5x150 LSTM)
    
    Returns
    -------
    config : LSTMConfig
        Model configuration.
    """
    configs = {
        'standard': LSTMConfig(
            seq_len=50,
            n_features=3,
            lstm_units=[100, 100, 100, 100],
            dropout=0.2,
            learning_rate=0.001,
        ),
        'small': LSTMConfig(
            seq_len=30,
            n_features=3,
            lstm_units=[50, 50, 50],
            dropout=0.2,
            learning_rate=0.001,
        ),
        'large': LSTMConfig(
            seq_len=60,
            n_features=3,
            lstm_units=[150, 150, 150, 150, 150],
            dropout=0.2,
            learning_rate=0.0005,
        )
    }
    
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")
    
    return configs[preset]


if __name__ == "__main__":
    # Example: Create and display builded model
    print("="*60)
    print("BUILD LSTM MODEL")
    print("="*60)
    
    config = LSTMConfig(
        seq_len=50,
        n_features=3,
        lstm_units=[100, 100, 100, 100],
        dropout=0.2
    )
    
    model = build_lstm_model(config)
    model.summary()
    
    print("\n" + "="*60)
    print("Using LSTMFinancialModel wrapper:")
    print("="*60)
    
    lstm_wrapper = LSTMFinancialModel(config)
    lstm_wrapper.summary()
