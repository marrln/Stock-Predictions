"""7_evaluate_models.py

Evaluate all trained models and find the best one.
"""

import torch
import numpy as np
from pathlib import Path
from PriceNewsDataset import load_dataloaders
from Model import PriceNewsLSTMReg
from checkpoint import find_all_experiments
from plotter import save_model_comparison, plot_model_comparison


def evaluate_model(model, loader, device):
    """Evaluate model and return metrics."""
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
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory used for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test data
    print(f"Loading datasets from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader = load_dataloaders(args.data_dir, args.batch_size, num_workers=0)
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {args.data_dir}")
        print("Make sure you've run 6_train_models.py first")
        return
    
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Find all experiments
    experiments = find_all_experiments()
    
    if not experiments:
        print("[ERROR] No experiments found in experiments/ directory")
        print("Run 6_train_models.py first")
        return
    
    print(f"Found {len(experiments)} trained models\n")
    print(f"{'='*80}")
    print("Evaluating models on test set")
    print(f"{'='*80}\n")
    
    # Evaluate each model
    results = {}
    input_size = test_loader.dataset.X.shape[-1]
    
    for exp_dir, config, ckpt_path in experiments:
        model_name = exp_dir.name
        
        print(f"Evaluating {model_name}...")
        
        # Load model
        model = PriceNewsLSTMReg(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pooling=config.get("pooling", "last"),
            bidirectional=config.get("bidirectional", False),
        ).to(device)
        
        # Load checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            print(f"  ❌ Failed to load checkpoint: {e}")
            continue
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        # Get training info
        best_val = ckpt.get("best_val", float('inf'))
        epoch = ckpt.get("epoch", "N/A")
        
        results[model_name] = {
            **metrics,
            "config": config,
            "best_val_loss": best_val,
            "best_epoch": epoch,
        }
        
        print(f"  Test MSE: {metrics['mse']:.6f}")
        print(f"  Test MAE: {metrics['mae']:.6f}")
        print(f"  Test RMSE: {metrics['rmse']:.6f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        print()
    
    # Find best model
    print(f"{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")
    
    best_model = min(results.items(), key=lambda x: x[1]["mse"])
    best_name, best_metrics = best_model
    
    print(f"Best model: {best_name}")
    print(f"  Test MSE: {best_metrics['mse']:.6f}")
    print(f"  Test MAE: {best_metrics['mae']:.6f}")
    print(f"  Test RMSE: {best_metrics['rmse']:.6f}")
    print(f"  Directional Accuracy: {best_metrics['directional_accuracy']:.2%}")
    print()
    
    # Save results
    comparison_dir = Path("experiments/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    save_model_comparison(results, comparison_dir / "model_comparison.json")
    plot_model_comparison(results, comparison_dir / "model_comparison.png")
    
    print(f"Results saved to {comparison_dir}")
    print(f"   - model_comparison.json")
    print(f"   - model_comparison.png")
    print("\nNext step:")
    print(f"  python 8_compare_models.py --best_model experiments/{best_name}/best.pt --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
