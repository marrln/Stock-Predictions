"""Training example for LSTM model.

This model predicts NORMALIZED PRICES (not returns).

Key features:
1. Fewer features (4: close, volume, daily_sentiment, n_articles) or chosen by user
2. Relative normalization to first value
3. Predicts normalized prices


Usage:
Use after preparing data with `tf_1_combine_data.py`.

Usage example FULL:
    python tf_2_train.py --data processed_data/csv --tickers AAPL,MSFT,NVDA \
        --seq-len 8 --train-split-end 0.70 --val-split-end 0.85 \
        --features close,volume,daily_sentiment,n_articles --norm-cols 0,1 \
        --config custom --lstm-units 100,100,100,100 --dropout 0.2 --learning-rate 0.001 \
        --epochs 50 --batch-size 128 --verbose 1 \
        --es-monitor val_loss --es-patience 10 --es-min-delta 0.0 \
        --rlrop-factor 0.5 --rlrop-patience 2 --rlrop-min-lr 1e-7 \
        --visualize --save-model --model-dir core_tf/saved_models --model-name improved_lstm_model.keras

Simple example:
    python3 tf_2_train.py --tickers AAPL,MSFT,NVDA --seq-len 8 --visualize --es-patience 20 --config custom --lstm-units 100 --save-model
    
"""

# Reduce TensorFlow INFO logs and disable oneDNN notices unless overridden
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import argparse

from core_tf.models import LSTMFinancialModel, LSTMConfig, get_lstm_config
from core_tf.data import prepare_data, denormalize_predictions

def _parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(',') if x.strip()]

def _parse_str_list(value: str):
    if ',' in value:
        return [x.strip() for x in value.split(',') if x.strip()]
    else:
        return [x.strip() for x in value.split() if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Train Improved LSTM financial model (normalized prices)")

    # Data & preprocessing
    parser.add_argument("--data", default="processed_data/csv", help="Path to the folder that contains stock CSV files")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA", help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,GOOGL)")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length (timesteps)")
    parser.add_argument("--train-split-end", type=float, default=0.70, help="Train split fraction end")
    parser.add_argument("--val-split-end", type=float, default=0.85, help="Validation split fraction end")
    parser.add_argument("--features", default="close,volume,daily_sentiment,n_articles", help="Comma-separated feature names used by the model")
    parser.add_argument("--norm-cols", default="0,1", help="Comma-separated indices of feature columns to normalize (relative to first value)")

    # Model config
    parser.add_argument("--config", choices=["standard", "custom"], default="standard", help="Use a built-in config or define custom params")
    parser.add_argument("--lstm-units", default="100,100,100,100", help="Comma-separated LSTM units per layer (used when --config custom)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (used when --config custom)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (used when --config custom)")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--verbose", type=int, default=1, help="Training verbosity level")

    # EarlyStopping / LR scheduling
    parser.add_argument("--es-monitor", default="val_loss", help="Metric to monitor for EarlyStopping/ReduceLROnPlateau")
    parser.add_argument("--es-patience", type=int, default=10, help="Epochs with no improvement to wait before stopping early")
    parser.add_argument("--es-min-delta", type=float, default=0.0, help="Minimum change in monitored metric to qualify as improvement")
    parser.add_argument("--rlrop-factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced")
    parser.add_argument("--rlrop-patience", type=int, default=2, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument("--rlrop-min-lr", type=float, default=1e-7, help="Lower bound on the learning rate")

    # Output
    parser.add_argument("--visualize", action="store_true", help="Create and save training/forecast plots")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model to disk")
    parser.add_argument("--model-dir", default="core_tf/saved_models", help="Directory to save the model")
    return parser


def main(args):
    """Main training pipeline using LSTM model."""
    
    print("=" * 80)
    print("LSTM MODEL - Training Pipeline")
    print("=" * 80)

    
    # =====================================================================
    # 1. PREPARE DATA
    # =====================================================================
    print(f"1. Preparing data...")
    data_dir = Path(args.data)
    tickers = args.tickers
    
    print("\nUsing settings:")
    print(f"   Data dir: {data_dir}")
    print(f"   Tickers: {tickers}")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Train, val, test splits: {int(100 * args.train_split_end)}%, {int(100 * (args.val_split_end - args.train_split_end))}%, {int(100 * (1 - args.val_split_end))}%")
    
    X_train_list, y_train_list, y_train_base_list = [], [], []
    X_val_list, y_val_list, y_val_base_list = [], [], []
    X_test_list, y_test_list, y_test_base_list = [], [], []

    print("\nLoading and preparing data for each ticker...")
    for t in tickers:
        csv_path = data_dir / f"{t}.csv"
        if not csv_path.exists():
            print(f"(!) Warning: {csv_path} not found. Skipping {t}.")
            continue
        print(f"\n> Loading {csv_path}...")
        (X_train, y_train, y_train_base, X_val, y_val, y_val_base, X_test, y_test, y_test_base) = prepare_data(
            csv_file=str(csv_path),
            seq_len=args.seq_len,
            train_split=args.train_split_end,
            val_split=args.val_split_end,
            feature_cols=args.features,
            cols_to_norm=args.norm_cols,
            verbose=args.verbose
        )
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        y_train_base_list.append(y_train_base)
        
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        y_val_base_list.append(y_val_base)
        
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        y_test_base_list.append(y_test_base)
        
    if not X_train_list:
        raise FileNotFoundError("(!) No valid ticker CSVs found to train on.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_train_base = np.concatenate(y_train_base_list, axis=0)
    
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    y_val_base = np.concatenate(y_val_base_list, axis=0)
    
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    y_test_base = np.concatenate(y_test_base_list, axis=0)
    
    print(f"\nFinal data shapes:")
    print(f"  X_train: {X_train.shape}")  # (n_seq, seq_len-1, n_features)
    print(f"  y_train: {y_train.shape}")  # (n_seq,) -- The last value in each sequence (normalized price)
    
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # =====================================================================
    # 2. CREATE MODEL
    # =====================================================================
    print("\n2. Creating LSTM model...")
    
    # Option A: Use standard config or custom
    if args.config == 'standard':
        config = get_lstm_config('standard')
        config.seq_len = args.seq_len
        config.n_features = len(args.features)
    # Option B: Custom config (example shown above)
    else:
        config = LSTMConfig(
            seq_len=args.seq_len,
            n_features=len(args.features),
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )
    
    
    
    print(f"\nModel configuration:")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Number of features: {config.n_features}")    
    print(f"  Features: {args.features}")
    print(f"  LSTM units: {config.lstm_units}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Learning rate: {config.learning_rate}")
    print("\n")
    
    model = LSTMFinancialModel(config)
    model.summary()
    
    # =====================================================================
    # 3. TRAIN MODEL
    # =====================================================================
    print("\n3. Training model...")
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    es = EarlyStopping(
        monitor=args.es_monitor,
        patience=args.es_patience,
        min_delta=args.es_min_delta,
        restore_best_weights=True,
        verbose=1
    )
    rlrop = ReduceLROnPlateau(
        monitor=args.es_monitor,
        factor=args.rlrop_factor,
        patience=args.rlrop_patience,
        min_lr=args.rlrop_min_lr,
        verbose=1
    )
    # chkpt = ModelCheckpoint(
    #     filepath='best_model.keras',
    #     monitor='val_loss',
    #     save_best_only=True,
    #     verbose=1
    # )

    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        callbacks=[es, rlrop]
    )

    if getattr(es, 'stopped_epoch', 0):
        print(f"(!) EarlyStopping triggered at epoch {es.stopped_epoch} (monitor={args.es_monitor}).")
    else:
        print("EarlyStopping did not trigger.")
    
    # =====================================================================
    # 4. EVALUATE MODEL
    # =====================================================================
    print("\n4. Evaluating model on Test set...")
    
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Metrics (on normalized values):")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.6f}")
    
    # =====================================================================
    # 5. MAKE PREDICTIONS
    # =====================================================================
    print("\n5. Making predictions...")
    
    # Predict normalized values
    y_pred_norm = model.predict(X_test)
    
    # Denormalize to actual prices
    y_pred_actual = denormalize_predictions(y_pred_norm, y_test_base)
    y_test_actual = denormalize_predictions(y_test, y_test_base)
    
    print(f"\nPredictions (first 10):")
    print(f"{'Index':<8} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Error %':<12}")
    print("-" * 60)
    
    for i in range(min(10, len(y_pred_actual))):
        error = y_pred_actual[i] - y_test_actual[i]
        error_pct = (error / y_test_actual[i]) * 100
        print(f"{i:<8} ${y_pred_actual[i]:<11.2f} ${y_test_actual[i]:<11.2f} "
              f"${error:<11.2f} {error_pct:<11.2f}%")
    
    # Calculate actual price metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae_price = mean_absolute_error(y_test_actual, y_pred_actual)
    mse_price = mean_squared_error(y_test_actual, y_pred_actual)
    rmse_price = np.sqrt(mse_price)
    r2_price = r2_score(y_test_actual, y_pred_actual)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    print(f"\nTest Metrics (on actual prices):")
    print(f"  MAE: ${mae_price:.2f}")
    print(f"  RMSE: ${rmse_price:.2f}")
    print(f"  R²: {r2_price:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # =====================================================================
    # 6. VISUALIZE (Optional)
    # =====================================================================
    if args.visualize:
        print("\n6. Creating visualizations...")
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            # Plot 1: Actual vs Predicted Prices
            ax1.plot(y_test_actual, label='Actual Price', alpha=0.7, linewidth=1.5)
            ax1.plot(y_pred_actual, label='Predicted Price', alpha=0.7, linewidth=1.5)
            ax1.set_xlabel('Sample Index', fontsize=12)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title('Stock Price Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            # Plot 2: Training History
            ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
            ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss (MSE)', fontsize=12)
            ax2.set_title('Training History', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            # Save plot
            plot_dir = Path("figures")
            plot_dir.mkdir(exist_ok=True)
            plt.savefig(plot_dir / "model_results.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_dir}/model_results.png")
        except ImportError:
            print("(!) Matplotlib not available, skipping visualization")
    else:
        print("\n6. Skipping visualizations (use --visualize to enable)")
    
    # =====================================================================
    # 7. SAVE MODEL
    # =====================================================================
    if args.save_model:
        print("\n7. Saving model...")
        model_dir = Path(args.model_dir)
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"lstm_{len(tickers)}tick_{config.n_features}feat_{args.seq_len}seqlen.keras"
        model.save(str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print("\n7. Skipping model save (use --save-model to enable)")
    
    # =====================================================================
    # 8. SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nModel Type: LSTM (predicts normalized prices)")
    print(f"Architecture: {config.lstm_units} LSTM units")
    print(f"Sequence Length: {config.seq_len} timesteps")
    print(f"Features: {config.n_features} ({', '.join(args.features)})")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"\nPerformance:")
    print(f"  Test MAE: ${mae_price:.2f}")
    print(f"  Test RMSE: ${rmse_price:.2f}")
    print(f"  Test R²: {r2_price:.4f}")
    print(f"  Test MAPE: {mape:.2f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Parse lists
    args.features = _parse_str_list(args.features)
    args.norm_cols = _parse_int_list(args.norm_cols)
    args.lstm_units = _parse_int_list(args.lstm_units)
    args.tickers = _parse_str_list(args.tickers)

    main(args)
