#!/usr/bin/env python3
"""
Compare model predictions with baselines and analyze performance.
Enhanced version with unscaling and detailed ticker analysis.

Usage:
    python3 8_compare_models.py --model experiments/h64_l5_d0.0_lr0.0001_b64_optadam_losshuber_hd1.0_ptscale/best.pt --data-dir processed_data/5tickers_seq8 --ticker AAPL
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import torch

from core.data.loaders import load_dataloaders
from core.models.lstm import PriceNewsLSTMReg
from core.baselines import evaluate_persistence_on_loader
from core.training.trainer import get_predictions
from core.training.metrics import compute_regression_metrics
from core.utils.plotting import plot_ticker_performance


class ModelComparator:
    """Compare model performance with baselines."""
    
    def __init__(
        self,
        model_path: str | Path,
        data_dir: str | Path,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        self.test_loader = None
        self.model = None
        self.config = {}
        self.target_type = "return"
        
    def setup(self):
        """Setup data and model."""
        print(f"Loading data from {self.data_dir}...")
        try:
            _, _, self.test_loader = load_dataloaders(
                str(self.data_dir), 
                self.batch_size, 
                num_workers=0
            )
            print(f"Test samples: {len(self.test_loader.dataset)}")
            
            dataset = self.test_loader.dataset
            self.target_type = getattr(dataset, 'target_type', 'return')
            print(f"Target type: {self.target_type}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        
        print(f"Loading model from {self.model_path}...")
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Try to load config from adjacent config.json
            config_path = self.model_path.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Get model config from checkpoint (prefer checkpoint over file)
            extra = checkpoint.get("extra", {})
            model_config = (
                extra.get("model_config")
                or extra.get("run_metadata", {}).get("model_config")
                or extra.get("run_metadata", {}).get("config")
                or self.config
            )

            # Get input size from dataset
            input_size = self.test_loader.dataset.X.shape[-1]

            # Infer num_tickers from dataset if missing but embedding was used in run config
            num_tickers = model_config.get("num_tickers")
            if num_tickers is None and model_config.get("use_ticker_embedding") and self.test_loader is not None:
                ds = getattr(self.test_loader, "dataset", None)
                if ds is not None and hasattr(ds, "ticker_to_idx"):
                    num_tickers = len(ds.ticker_to_idx)

            # Reconstruct model
            self.model = PriceNewsLSTMReg(
                input_size=input_size,
                hidden_size=model_config.get("hidden_size", 128),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.2),
                pooling=model_config.get("pooling", "last"),
                bidirectional=model_config.get("bidirectional", False),
                num_tickers=num_tickers,
                ticker_emb_dim=model_config.get("ticker_emb_dim", 16),
                expansion_factor=model_config.get("expansion_factor", 4),
            ).to(self.device)

            # Load weights (support different keys and fall back to non-strict load)
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # If checkpoint appears to be a state dict (tensor values), use it directly
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    # Look for nested state dict
                    state_dict = None
                    for k, v in checkpoint.items():
                        if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                            state_dict = v
                            break
                    if state_dict is None:
                        raise KeyError("Checkpoint doesn't contain model weights")

            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: strict load failed: {e}. Trying non-strict load.")
                load_result = self.model.load_state_dict(state_dict, strict=False)
                # load_state_dict may return a NamedTuple or dict depending on torch version
                missing = getattr(load_result, "missing_keys", None) or load_result[0] if isinstance(load_result, tuple) else None
                unexpected = getattr(load_result, "unexpected_keys", None) or load_result[1] if isinstance(load_result, tuple) else None
                print(f"  Missing keys: {missing}")
                print(f"  Unexpected keys: {unexpected}")

            print(f"Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _get_target_scaler(self):
        """Get target scaler from dataset if available."""
        dataset = self.test_loader.dataset
        if hasattr(dataset, 'target_scaler') and dataset.target_scaler is not None:
            return dataset.target_scaler
        return None
    
    def evaluate_model(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model and return metrics, predictions, and true values."""
        # Get predictions
        predictions, targets = get_predictions(self.model, self.test_loader, self.device)
        
        # Compute metrics
        metrics = compute_regression_metrics(predictions, targets, include_directional=True, include_r2=True, include_sharpe=True)
        
        return metrics, predictions, targets
    
    def evaluate_baseline(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate persistence baseline."""
        return evaluate_persistence_on_loader(self.test_loader)
    
    def unscale_predictions(
        self, 
        predictions: np.ndarray, 
        ticker: Optional[str] = None
    ) -> np.ndarray:
        """Unscale predictions using dataset's target scaler."""
        scaler = self._get_target_scaler()
        if scaler is None:
            return predictions
        
        # Get metadata for ticker filtering
        dataset = self.test_loader.dataset
        if hasattr(dataset, 'meta') and dataset.meta is not None:
            meta = dataset.meta
        else:
            return predictions
        
        # If specific ticker requested, filter by ticker
        if ticker and 'Ticker' in meta.columns:
            ticker_mask = meta['Ticker'].str.upper() == ticker.upper()
            indices = np.where(ticker_mask)[0]
        else:
            indices = np.arange(len(predictions))
        
        if len(indices) == 0:
            return predictions
        
        # Apply inverse transform
        try:
            if isinstance(scaler, dict):
                # Per-ticker scalers
                if ticker:
                    ticker_scaler = scaler.get(ticker.upper())
                    if ticker_scaler:
                        unscaled = ticker_scaler.inverse_transform(
                            predictions[indices].reshape(-1, 1)
                        ).ravel()
                        result = predictions.copy()
                        result[indices] = unscaled
                        return result
            else:
                # Single scaler for all
                unscaled = scaler.inverse_transform(
                    predictions[indices].reshape(-1, 1)
                ).ravel()
                result = predictions.copy()
                result[indices] = unscaled
                return result
        except Exception as e:
            print(f"Warning: Failed to unscale predictions: {e}")
        
        return predictions
    
    def analyze_ticker(
        self, 
        ticker: str,
        model_predictions: np.ndarray,
        baseline_predictions: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance for a specific ticker."""
        dataset = self.test_loader.dataset
        
        if not hasattr(dataset, 'meta') or dataset.meta is None:
            return {}
        
        meta = dataset.meta
        if 'Ticker' not in meta.columns:
            return {}
        
        # Filter by ticker
        ticker_mask = meta['Ticker'].str.upper() == ticker.upper()
        indices = np.where(ticker_mask)[0]
        
        if len(indices) == 0:
            print(f"Warning: No data found for ticker {ticker}")
            return {}
        
        # Get predictions for this ticker
        model_preds_ticker = model_predictions[indices]
        baseline_preds_ticker = baseline_predictions[indices]
        true_values_ticker = true_values[indices]
        
        # Get dates for plotting
        dates = []
        if 'Date' in meta.columns:
            dates = meta.loc[ticker_mask, 'Date'].tolist()
        
        # Unscale if possible
        scaler = self._get_target_scaler()
        if scaler is not None:
            try:
                if isinstance(scaler, dict):
                    # Per-ticker scaler
                    ticker_scaler = scaler.get(ticker.upper())
                    if ticker_scaler:
                        model_preds_ticker = ticker_scaler.inverse_transform(
                            model_preds_ticker.reshape(-1, 1)
                        ).ravel()
                        baseline_preds_ticker = ticker_scaler.inverse_transform(
                            baseline_preds_ticker.reshape(-1, 1)
                        ).ravel()
                        true_values_ticker = ticker_scaler.inverse_transform(
                            true_values_ticker.reshape(-1, 1)
                        ).ravel()
                else:
                    # Single scaler
                    model_preds_ticker = scaler.inverse_transform(
                        model_preds_ticker.reshape(-1, 1)
                    ).ravel()
                    baseline_preds_ticker = scaler.inverse_transform(
                        baseline_preds_ticker.reshape(-1, 1)
                    ).ravel()
                    true_values_ticker = scaler.inverse_transform(
                        true_values_ticker.reshape(-1, 1)
                    ).ravel()
            except Exception as e:
                print(f"Warning: Failed to unscale ticker data: {e}")
        
        # Compute metrics
        model_metrics = compute_regression_metrics(
            model_preds_ticker, true_values_ticker, include_directional=True
        )
        baseline_metrics = compute_regression_metrics(
            baseline_preds_ticker, true_values_ticker, include_directional=True
        )
        
        return {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "data": {
                "dates": dates,
                "model_predictions": model_preds_ticker,
                "baseline_predictions": baseline_preds_ticker,
                "true_values": true_values_ticker,
                "sample_count": len(indices),
            }
        }
    
    def create_comparison_table(
        self, 
        model_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Create a comparison table between model and baseline."""
        data = {
            "Metric": ["MSE", "MAE", "RMSE", "Directional Accuracy"],
            "Model": [
                f"{model_metrics.get('mse', 0):.6f}",
                f"{model_metrics.get('mae', 0):.6f}",
                f"{model_metrics.get('rmse', 0):.6f}",
                f"{model_metrics.get('dir_acc', 0):.2%}",
            ],
            "Baseline": [
                f"{baseline_metrics.get('mse', 0):.6f}",
                f"{baseline_metrics.get('mae', 0):.6f}",
                f"{baseline_metrics.get('rmse', 0):.6f}",
                f"{baseline_metrics.get('dir_acc', 0):.2%}",
            ],
        }
        
        # Calculate improvements
        improvements = []
        for metric in ["mse", "mae", "rmse"]:
            model_val = model_metrics.get(metric, 0)
            baseline_val = baseline_metrics.get(metric, 0)
            if baseline_val > 0:
                improvement = (1 - model_val / baseline_val) * 100
                improvements.append(f"{improvement:+.1f}%")
            else:
                improvements.append("N/A")
        
        # Directional accuracy improvement
        dir_improvement = (model_metrics.get('dir_acc', 0) - baseline_metrics.get('dir_acc', 0)) * 100
        improvements.append(f"{dir_improvement:+.1f}%")
        
        data["Improvement"] = improvements
        
        return pd.DataFrame(data)
    
    def save_results(
        self,
        model_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        ticker_analysis: Dict[str, Dict],
        output_dir: Path
    ):
        """Save comparison results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall metrics
        overall_results = {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "config": self.config,
            "model_path": str(self.model_path),
            "data_dir": str(self.data_dir),
        }
        
        with open(output_dir / "overall_comparison.json", "w") as f:
            json.dump(overall_results, f, indent=2)
        
        # Save ticker analysis
        if ticker_analysis:
            with open(output_dir / "ticker_analysis.json", "w") as f:
                json.dump(ticker_analysis, f, indent=2, default=str)
        
        # Save comparison table as CSV
        comparison_df = self.create_comparison_table(model_metrics, baseline_metrics)
        comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
        
        print(f"\nResults saved to {output_dir}")
        print(f"  - overall_comparison.json")
        print(f"  - comparison_table.csv")
        if ticker_analysis:
            print(f"  - ticker_analysis.json")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions with baselines"
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--ticker", type=str, default="AAPL",
                       help="Specific ticker to analyze")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL COMPARISON WITH BASELINES")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ModelComparator(
        model_path=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    try:
        # Setup data and model
        comparator.setup()
        
        # Evaluate model
        print("\n1. Evaluating LSTM model...")
        model_metrics, model_preds, true_values = comparator.evaluate_model()
        
        print(f"   MSE: {model_metrics.get('mse', 0):.6f}")
        print(f"   MAE: {model_metrics.get('mae', 0):.6f}")
        print(f"   RMSE: {model_metrics.get('rmse', 0):.6f}")
        print(f"   Directional Accuracy: {model_metrics.get('dir_acc', 0):.2%}")
        
        # Evaluate baseline
        print("\n2. Evaluating persistence baseline...")
        baseline_metrics, baseline_preds, _ = comparator.evaluate_baseline()
        
        print(f"   MSE: {baseline_metrics.get('mse', 0):.6f}")
        print(f"   MAE: {baseline_metrics.get('mae', 0):.6f}")
        print(f"   RMSE: {baseline_metrics.get('rmse', 0):.6f}")
        print(f"   Directional Accuracy: {baseline_metrics.get('dir_acc', 0):.2%}")
        
        # Analyze specific ticker
        print(f"\n3. Analyzing ticker: {args.ticker}")
        ticker_analysis = comparator.analyze_ticker(
            args.ticker, model_preds, baseline_preds, true_values
        )
        
        if ticker_analysis:
            model_ticker_metrics = ticker_analysis["model"]
            baseline_ticker_metrics = ticker_analysis["baseline"]
            data = ticker_analysis["data"]
            
            print(f"   Found {data['sample_count']} samples")
            print(f"   Model (unscaled) - MSE: {model_ticker_metrics.get('mse', 0):.6f}, "
                  f"MAE: {model_ticker_metrics.get('mae', 0):.6f}")
            print(f"   Baseline (unscaled) - MSE: {baseline_ticker_metrics.get('mse', 0):.6f}, "
                  f"MAE: {baseline_ticker_metrics.get('mae', 0):.6f}")
            
            # Plot ticker performance
            if data['dates'] and len(data['dates']) == len(data['true_values']):
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plot_ticker_performance(
                    ticker=args.ticker,
                    dates=data['dates'],
                    true_values=data['true_values'],
                    predicted_series={
                        "LSTM Model": data['model_predictions'],
                        "Persistence Baseline": data['baseline_predictions']
                    },
                    save_path=output_dir / f"{args.ticker}_comparison.png",
                    show=False,
                    target_type=comparator.target_type
                )
                print(f"   Plot saved to {output_dir / f'{args.ticker}_comparison.png'}")
        
        # Create and display comparison table
        print("\n" + "=" * 80)
        print("OVERALL COMPARISON")
        print("=" * 80)
        
        comparison_df = comparator.create_comparison_table(model_metrics, baseline_metrics)
        print("\n" + comparison_df.to_string(index=False))
        
        # Calculate overall improvement
        improvement_mse = 0
        if baseline_metrics.get('mse', 0) > 0:
            improvement_mse = (1 - model_metrics.get('mse', 0) / baseline_metrics.get('mse', 0)) * 100
        
        print(f"\nOverall MSE Improvement: {improvement_mse:+.1f}%")
        if improvement_mse > 0:
            print(f"✓ LSTM outperforms baseline by {improvement_mse:.1f}%")
        else:
            print(f"⚠ Baseline outperforms LSTM by {-improvement_mse:.1f}%")
        
        # Save results
        output_dir = Path(args.output_dir)
        comparator.save_results(
            model_metrics, baseline_metrics, ticker_analysis, output_dir
        )
        
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()