"""Model evaluation and comparison utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import pandas as pd

from .models.lstm import PriceNewsLSTMReg
from .checkpoint import find_all_experiments
from .training.trainer import get_predictions, compute_regression_metrics
from .utils.plotting import save_model_comparison, plot_model_comparison
from .baselines import evaluate_persistence_on_loader


class ModelEvaluator:
    """Evaluate trained models and compare performance."""
    
    def __init__(
        self, 
        data_dir: str | Path,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        # Will be set during setup
        self.test_loader = None
        self.input_size = None
        
    def setup(self):
        """Setup data loader."""
        from .data.loaders import load_dataloaders
        
        try:
            _, _, test_loader = load_dataloaders(
                str(self.data_dir), 
                self.batch_size, 
                num_workers=0
            )
            self.test_loader = test_loader
            self.input_size = test_loader.dataset.X.shape[-1]
            
            print(f"Loaded test data: {len(test_loader.dataset)} samples")
            print(f"Input size: {self.input_size}")
            
        except FileNotFoundError:
            raise RuntimeError(f"Data not found at {self.data_dir}")
    
    def load_model(self, checkpoint_path: Path, config: Dict[str, Any]) -> PriceNewsLSTMReg:
        """Load model from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model configuration (prefer explicit model_config, then run_metadata config, fallback to provided config)
            extra = checkpoint.get("extra", {})
            model_config = (
                extra.get("model_config")
                or extra.get("run_metadata", {}).get("model_config")
                or extra.get("run_metadata", {}).get("config")
                or config
            )

            # Infer num_tickers from dataset if missing but embedding was used in run config
            num_tickers = model_config.get("num_tickers")
            if num_tickers is None and model_config.get("use_ticker_embedding") and self.test_loader is not None:
                ds = getattr(self.test_loader, "dataset", None)
                if ds is not None and hasattr(ds, "ticker_to_idx"):
                    num_tickers = len(ds.ticker_to_idx)

            # Reconstruct model
            model = PriceNewsLSTMReg(
                input_size=self.input_size,
                hidden_size=model_config.get("hidden_size", 128),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.2),
                pooling=model_config.get("pooling", "last"),
                bidirectional=model_config.get("bidirectional", False),
                num_tickers=num_tickers,
                ticker_emb_dim=model_config.get("ticker_emb_dim", 16),
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint["model_state"])
            
            return model
            
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
            raise
    
    def evaluate_model(self, model: PriceNewsLSTMReg) -> Dict[str, float]:
        """Evaluate a single model on test set."""
        if self.test_loader is None:
            raise RuntimeError("Call setup() first")
        
        # Get predictions
        predictions, targets = get_predictions(model, self.test_loader, self.device)
        
        # Compute metrics
        metrics = compute_regression_metrics(predictions, targets, include_directional=True)
        
        return metrics
    
    def evaluate_all_experiments(
        self, 
        experiments_dir: str = "experiments",
        include_baselines: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained experiments."""
        if self.test_loader is None:
            self.setup()
        
        all_results = {}
        
        # Evaluate trained models
        experiments = find_all_experiments(experiments_dir)
        
        if not experiments:
            print(f"No experiments found in {experiments_dir}")
            return {}
        
        print(f"Found {len(experiments)} experiments")
        print("=" * 80)
        
        for exp_dir, config, checkpoint_path in experiments:
            print(f"Evaluating: {exp_dir.name}")
            
            try:
                # Load and evaluate model
                model = self.load_model(checkpoint_path, config)
                metrics = self.evaluate_model(model)
                
                # Get training info from checkpoint
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                all_results[exp_dir.name] = {
                    "metrics": metrics,
                    "config": config,
                    "checkpoint_info": {
                        "path": str(checkpoint_path),
                        "best_val_loss": checkpoint.get("best_val"),
                        "best_epoch": checkpoint.get("epoch"),
                    }
                }
                
                print(f"  Test MSE: {metrics['mse']:.6f}")
                print(f"  Test MAE: {metrics['mae']:.6f}")
                print(f"  Test RMSE: {metrics['rmse']:.6f}")
                print(f"  Directional Accuracy: {metrics.get('dir_acc', 0.0):.2%}")
                print()
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        # Evaluate baselines
        if include_baselines and self.test_loader is not None:
            print("Evaluating baselines...")
            
            # Persistence baseline
            persistence_metrics, _, _ = evaluate_persistence_on_loader(self.test_loader)
            all_results["persistence"] = {
                "metrics": persistence_metrics,
                "config": {"model": "persistence"},
                "checkpoint_info": None
            }
            
            print(f"  Persistence MSE: {persistence_metrics['mse']:.6f}")
            print(f"  Persistence MAE: {persistence_metrics['mae']:.6f}")
            print(f"  Persistence RMSE: {persistence_metrics['rmse']:.6f}")
            print(f"  Directional Accuracy: {persistence_metrics['dir_acc']:.2%}")
            print()
        
        return all_results
    
    def find_best_model(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Find the best model based on MSE."""
        if not results:
            raise ValueError("No results to compare")
        
        # Filter out baselines if present
        model_results = {
            name: data for name, data in results.items() 
            if name != "persistence" and "metrics" in data
        }
        
        if not model_results:
            return ("persistence", results["persistence"])
        
        best_name = min(
            model_results.items(),
            key=lambda x: x[1]["metrics"]["mse"]
        )[0]
        
        return best_name, results[best_name]
    
    def create_summary_table(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a summary table of all results."""
        rows = []
        
        for name, data in results.items():
            metrics = data.get("metrics", {})
            config = data.get("config", {})
            checkpoint_info = data.get("checkpoint_info") or {}
            
            row = {
                "Model": name,
                "MSE": metrics.get("mse", float("nan")),
                "MAE": metrics.get("mae", float("nan")),
                "RMSE": metrics.get("rmse", float("nan")),
                "DirAcc": metrics.get("dir_acc", float("nan")),
                "Hidden Size": config.get("hidden_size", "N/A"),
                "Layers": config.get("num_layers", "N/A"),
                "Dropout": config.get("dropout", "N/A"),
                "Pooling": config.get("pooling", "N/A"),
                "Best Val Loss": checkpoint_info.get("best_val_loss", "N/A"),
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(
        self, 
        results: Dict[str, Dict[str, Any]], 
        output_dir: str | Path
    ) -> None:
        """Save evaluation results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_results = {}
        for name, data in results.items():
            checkpoint_info = data.get("checkpoint_info") or {}
            detailed_results[name] = {
                "metrics": data.get("metrics", {}),
                "config": data.get("config", {}),
                "checkpoint_path": checkpoint_info.get("path"),
            }
        
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save summary table
        summary_df = self.create_summary_table(results)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        
        # Plot comparison
        metrics_for_plot = {
            name: data["metrics"] 
            for name, data in results.items() 
            if "metrics" in data
        }
        
        plot_model_comparison(
            metrics_for_plot,
            save_path=output_dir / "model_comparison.png",
            show=False
        )
        
        print(f"\nResults saved to {output_dir}")
        print(f"  - detailed_results.json")
        print(f"  - summary.csv")
        print(f"  - model_comparison.png")


def evaluate_single_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Evaluate a single model checkpoint."""
    evaluator = ModelEvaluator(data_dir, batch_size)
    evaluator.setup()
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load config (try adjacent config.json, then checkpoint)
    config = {}
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # Load and evaluate model
    model = evaluator.load_model(checkpoint_path, config)
    metrics = evaluator.evaluate_model(model)
    
    return {
        "metrics": metrics,
        "config": config,
        "checkpoint_path": str(checkpoint_path),
    }


def compare_experiments(
    experiments_dir: str = "experiments",
    data_dir: str = "processed_data",
    output_dir: str = "experiments/comparison",
    batch_size: int = 64,
    include_baselines: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Main function to compare all experiments and find the best model."""
    print("=" * 80)
    print("Model Evaluation and Comparison")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(data_dir, batch_size)
    
    # Evaluate all experiments
    results = evaluator.evaluate_all_experiments(
        experiments_dir, 
        include_baselines=include_baselines
    )
    
    if not results:
        print("No models evaluated successfully")
        return "", {}
    
    # Find best model
    best_name, best_result = evaluator.find_best_model(results)
    
    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Name: {best_name}")
    
    metrics = best_result.get("metrics", {})
    print(f"Test MSE: {metrics.get('mse', 'N/A'):.6f}")
    print(f"Test MAE: {metrics.get('mae', 'N/A'):.6f}")
    print(f"Test RMSE: {metrics.get('rmse', 'N/A'):.6f}")
    print(f"Directional Accuracy: {metrics.get('dir_acc', 0.0):.2%}")
    
    # Save results
    evaluator.save_results(results, output_dir)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_df = evaluator.create_summary_table(results)
    print(summary_df.to_string(index=False))
    
    return best_name, best_result