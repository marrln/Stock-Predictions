"""
Model V2 - Supports Price and Direction Prediction
===================================================

Features:
- Regression mode (price prediction)
- Classification mode (direction prediction)
- Consistent architecture
- Optional attention mechanism

"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


@dataclass
class ModelConfig:
    """Configuration for LSTM model."""
    
    # Architecture
    seq_len: int = 50
    n_features: int = 4
    lstm_units: List[int] = field(default_factory=lambda: [64, 64])
    
    # Regularization
    dropout: float = 0.2
    l2_reg: float = 1e-5
    use_layer_norm: bool = True
    
    # Task
    features_list: Optional[List[int]] = None
    task: Literal['price', 'direction', 'return'] = 'price'
    
    # Training
    learning_rate: float = 0.001
    
    def save_json(self, filepath: str):
        """Save config to JSON."""
        data = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ModelConfig':
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


def build_lstm_model(config: ModelConfig) -> Model:
    """Build LSTM model for price or direction prediction.
    
    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    
    Returns
    -------
    model : keras.Model
        Compiled model.
    """
    
    # Input 
    # [batch_size, seq_len - 1, n_features]
    inputs = keras.Input(
        shape=(config.seq_len - 1, config.n_features),
        name='sequence_input'
    )
    
    x = inputs

    # L2 regularizer
    reg = keras.regularizers.l2(config.l2_reg) if config.l2_reg > 0 else None
    
    # LSTM stack
    for i, units in enumerate(config.lstm_units):
        return_sequences = (i < len(config.lstm_units) - 1)
        
        x = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            kernel_regularizer=reg,
            name=f'lstm_{i+1}'
        )(x)
        
        # LayerNorm + Dropout after each layer except last
        if return_sequences:
            if config.use_layer_norm:
                x = layers.LayerNormalization(name=f'ln_{i+1}')(x)
            if config.dropout > 0:
                x = layers.Dropout(config.dropout, name=f'dropout_{i+1}')(x)
    
    # Final dropout
    if config.dropout > 0:
        x = layers.Dropout(config.dropout / 2, name='final_dropout')(x)
    
    # Output layer based on task
    if config.task == 'direction':
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        # Regression (price or return)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f'LSTM_{config.task}')
    
    # Compile based on task
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    if config.task == 'direction':
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    return model


class FinancialModel:
    """Wrapper for financial prediction models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = build_lstm_model(config)
        self.history = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
        class_weight: Optional[dict] = None,
        patience_es: int = 15,
        patience_lr: int = 5,
        verbose: int = 1
    ):
        """Train the model."""
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=patience_es,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=patience_lr,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            shuffle=False,  # Important for time series
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, batch_size=batch_size, verbose=0).flatten()
    
    def predict_direction(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict direction (for classification task)."""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> dict:
        """Evaluate model."""
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        return dict(zip(self.model.metrics_names, results))
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def save(self, filepath: str):
        """Save model."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, config: ModelConfig):
        """Load model."""
        instance = cls(config)
        instance.model = keras.models.load_model(filepath)
        return instance


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_config_preset(
    preset: str,
    task: str = 'price',
    seq_len: int = 50,
    n_features: int = 4
) -> ModelConfig:
    """Get preset model configuration.
    
    Presets:
    - 'small': 2x32 LSTM, fast training
    - 'medium': 2x64 LSTM, balanced
    - 'large': 3x128 LSTM, more capacity
    """
    
    presets = {
        'small': {
            'lstm_units': [32, 32],
            'dropout': 0.2,
            'learning_rate': 0.001
        },
        'medium': {
            'lstm_units': [64, 64],
            'dropout': 0.2,
            'learning_rate': 0.001
        },
        'large': {
            'lstm_units': [128, 128, 64],
            'dropout': 0.3,
            'learning_rate': 0.0005
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    params = presets[preset]
    
    return ModelConfig(
        seq_len=seq_len,
        n_features=n_features,
        task=task,
        **params
    )


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Model V2 - Price Prediction")
    print("="*60)
    
    config_price = ModelConfig(
        seq_len=50,
        n_features=4,
        lstm_units=[64, 64],
        task='price'
    )
    model_price = build_lstm_model(config_price)
    model_price.summary()
    
    print("\n" + "="*60)
    print("Model V2 - Direction Prediction")
    print("="*60)
    
    config_dir = ModelConfig(
        seq_len=50,
        n_features=4,
        lstm_units=[64, 64],
        task='direction'
    )
    model_dir = build_lstm_model(config_dir)
    model_dir.summary()
