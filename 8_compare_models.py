"""8_compare_models.py

Compare best LSTM model with baseline (persistence model).
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PriceNewsDataset import load_dataloaders
from Model import PriceNewsLSTMReg
from baselines import evaluate_persistence_on_loader
from checkpoint import load_config_from_dir
from plotter import plot_ticker_performance, save_model_comparison, plot_model_comparison


def evaluate_model(model, loader, device):
    """Evaluate model and return metrics + predictions."""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    
    mse = float(((preds - trues) ** 2).mean())
    mae = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    dir_acc = float((np.sign(preds) == np.sign(trues)).mean())
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": dir_acc,
    }, preds, trues


def main():
    parser = argparse.ArgumentParser(description="Compare best model with baseline")
    parser.add_argument("--best_model", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker for visualization")
    parser.add_argument("--save_dir", type=str, default="experiments/final_comparison", help="Output directory")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load datasets
    print(f"Loading datasets from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader = load_dataloaders(args.data_dir, args.batch_size, num_workers=0)
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {args.data_dir}")
        return
    
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Load best model
    ckpt_path = Path(args.best_model)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Loading best model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Get config from experiment directory
    exp_dir = ckpt_path.parent
    config = load_config_from_dir(exp_dir)
    
    if not config:
        print("[WARN] Config not found, using checkpoint metadata")
        # Try to infer from checkpoint
        config = ckpt.get("extra", {}).get("model_config", {})
    
    # Build model
    input_size = test_loader.dataset.X.shape[-1]
    model = PriceNewsLSTMReg(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling=config.get("pooling", "last"),
        bidirectional=config.get("bidirectional", False),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    print(f"Model loaded: {exp_dir.name}\n")
    
    print(f"{'='*80}")
    print("Evaluating on test set")
    print(f"{'='*80}\n")
    
    print("1. LSTM model...")
    lstm_metrics, lstm_preds, trues = evaluate_model(model, test_loader, device)
    print(f"   MSE: {lstm_metrics['mse']:.6f}")
    print(f"   MAE: {lstm_metrics['mae']:.6f}")
    print(f"   RMSE: {lstm_metrics['rmse']:.6f}")
    print(f"   Directional Accuracy: {lstm_metrics['directional_accuracy']:.2%}\n")
    
    print("2. Persistence baseline...")
    persistence_metrics, persistence_preds, _ = evaluate_persistence_on_loader(test_loader)
    print(f"   MSE: {persistence_metrics['mse']:.6f}")
    print(f"   MAE: {persistence_metrics['mae']:.6f}")
    print(f"   RMSE: {persistence_metrics['rmse']:.6f}")
    print(f"   Directional Accuracy: {persistence_metrics.get('dir_acc', persistence_metrics.get('directional_accuracy', 0)):.2%}\n")
    
    print(f"{'='*80}")
    print("Comparison")
    print(f"{'='*80}\n")
    
    improvement_mse = (persistence_metrics['mse'] - lstm_metrics['mse']) / persistence_metrics['mse'] * 100
    improvement_mae = (persistence_metrics['mae'] - lstm_metrics['mae']) / persistence_metrics['mae'] * 100
    
    print(f"LSTM vs Persistence:")
    print(f"  MSE improvement: {improvement_mse:+.1f}%")
    print(f"  MAE improvement: {improvement_mae:+.1f}%")
    
    if improvement_mse > 0:
        print(f"  LSTM outperforms baseline by {improvement_mse:.1f}% (MSE)")
    else:
        print(f"  [WARN] Baseline outperforms LSTM by {-improvement_mse:.1f}% (MSE)")
    print()
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        "lstm_best": {
            **lstm_metrics,
            "config": config,
            "model_path": str(ckpt_path),
        },
        "persistence": persistence_metrics,
    }
    
    save_model_comparison(comparison_data, save_dir / "model_comparison.json")
    plot_model_comparison(comparison_data, save_dir / "model_comparison.png")
    
    print(f"Results saved to {save_dir}")
    print(f"   - model_comparison.json")
    print(f"   - model_comparison.png")
    
    # Plot predictions for specific ticker
    print(f"\nGenerating plots for {args.ticker}...")
    
    # Get ticker-specific data from meta
    ticker_indices = []
    if hasattr(test_loader.dataset, 'meta') and test_loader.dataset.meta is not None:
        import pandas as pd
        meta_df = test_loader.dataset.meta
        if isinstance(meta_df, pd.DataFrame) and 'ticker' in meta_df.columns:
            ticker_mask = meta_df['ticker'] == args.ticker
            ticker_indices = meta_df.index[ticker_mask].tolist()
    
    if ticker_indices:
        ticker_preds_lstm = lstm_preds[ticker_indices]
        ticker_preds_persistence = persistence_preds[ticker_indices]
        ticker_trues = trues[ticker_indices]
        
        predicted_series = {
            "lstm_best": ticker_preds_lstm,
            "persistence": ticker_preds_persistence,
        }
        
        plot_ticker_performance(
            ticker=args.ticker,
            true_returns=ticker_trues,
            predicted_series=predicted_series,
            save_path=save_dir / f"{args.ticker}_returns.png"
        )
        
        print(f"   - {args.ticker}_returns.png")
    else:
        print(f"   [WARN] No data found for ticker {args.ticker}")
    
    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
