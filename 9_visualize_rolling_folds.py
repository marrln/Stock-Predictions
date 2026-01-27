#!/usr/bin/env python3
"""
Compare and visualize model predictions across rolling folds.

Usage:
    python3 9_visualize_rolling_folds.py --experiment experiments/rolling_example --data-dir processed_data/3tickers_seq30_rolling --ticker AAPL
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
import torch

from core.data.loaders import load_rolling_folds
from core.models.lstm import PriceNewsLSTMReg
from core.training.trainer import get_predictions, compute_unscaled_metrics
from core.training.metrics import compute_regression_metrics
from core.utils.plotting import plot_rolling_predictions, plot_fold_comparison


class RollingFoldVisualizer:
    """Visualize predictions across rolling folds."""
    
    def __init__(
        self,
        experiment_dir: str | Path,
        data_dir: str | Path,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        self.folds = None
        self.fold_configs = []
        self.target_type = "return"
        
    def setup(self):
        """Load rolling folds data."""
        print(f"Loading rolling folds from {self.data_dir}...")
        self.folds = load_rolling_folds(
            str(self.data_dir), 
            self.batch_size, 
            num_workers=0
        )
        print(f"Loaded {len(self.folds)} folds")
        
        if len(self.folds) > 0:
            train_loader, _, _ = self.folds[0]
            dataset = train_loader.dataset
            self.target_type = getattr(dataset, 'target_type', 'return')
            print(f"Target type: {self.target_type}")
    
    def load_fold_model(self, fold_idx: int) -> Optional[PriceNewsLSTMReg]:
        """Load model checkpoint for a specific fold.

        The function is robust to different experiment layouts. It will look for:
          - <experiment_dir>/fold_{i}/best.pt
          - any subtree under <experiment_dir> containing a parent named 'fold_{i}' (useful when experiments are saved under top-level fold dirs)
          - fall back to the first best.pt found under <experiment_dir> if no fold-specific checkpoint exists
        """
        # Preferred location: <experiment_dir>/fold_{i}/best.pt
        fold_dir = self.experiment_dir / f"fold_{fold_idx}"
        checkpoint_path = fold_dir / "best.pt"

        if not checkpoint_path.exists():
            # Search recursively under experiment_dir first
            candidates = list(self.experiment_dir.rglob("best.pt"))

            # Also search one level up (covers layout: experiments/fold_{i}/<exp>/best.pt)
            if self.experiment_dir.parent != self.experiment_dir:
                candidates += list(self.experiment_dir.parent.rglob("best.pt"))

            # Prefer a candidate that has a parent named fold_{fold_idx}
            chosen = None
            for c in candidates:
                if any(p.name == f"fold_{fold_idx}" for p in c.parents):
                    chosen = c
                    break

            if chosen is None and candidates:
                # Fallback to the first candidate
                chosen = candidates[0]

            if chosen is None:
                print(f"Warning: No checkpoint found for fold {fold_idx} under {self.experiment_dir}")
                return None

            checkpoint_path = chosen
            fold_dir = checkpoint_path.parent

        config_path = fold_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            extra = checkpoint.get("extra", {})
            model_config = (
                extra.get("model_config")
                or extra.get("run_metadata", {}).get("model_config")
                or extra.get("run_metadata", {}).get("config")
                or config
            )
            
            train_loader, _, _ = self.folds[fold_idx]
            input_size = train_loader.dataset.X.shape[-1]
            
            num_tickers = model_config.get("num_tickers")
            if num_tickers is None and model_config.get("use_ticker_embedding"):
                ds = train_loader.dataset
                if hasattr(ds, "ticker_to_idx"):
                    num_tickers = len(ds.ticker_to_idx)
            
            model = PriceNewsLSTMReg(
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
            
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading model for fold {fold_idx}: {e}")
            return None
    
    def collect_predictions_for_ticker(
        self, 
        ticker: str,
        split: str = "test"
    ) -> List[Dict]:
        """Collect predictions across all folds for a specific ticker."""
        fold_data = []
        
        for fold_idx in range(len(self.folds)):
            print(f"Processing fold {fold_idx}...")
            
            train_loader, val_loader, test_loader = self.folds[fold_idx]
            
            if split == "train":
                loader = train_loader
            elif split == "val":
                loader = val_loader
            elif split == "test":
                if test_loader is None:
                    continue
                loader = test_loader
            else:
                continue
            
            model = self.load_fold_model(fold_idx)
            if model is None:
                continue
            
            dataset = loader.dataset
            if not hasattr(dataset, 'meta') or dataset.meta is None:
                continue
            
            meta = dataset.meta
            if 'Ticker' not in meta.columns:
                continue
            
            ticker_mask = meta['Ticker'].str.upper() == ticker.upper()
            indices = np.where(ticker_mask)[0]
            
            if len(indices) == 0:
                continue
            
            predictions, targets = get_predictions(model, loader, self.device)
            
            # Extract series for this ticker (predictions/targets are in dataset order)
            ticker_preds_scaled = np.asarray(predictions)[indices]
            ticker_targets_scaled = np.asarray(targets)[indices]
            ticker_dates = meta.iloc[indices]['Date'].values

            # Attempt to inverse-transform using dataset's target_scaler (may be per-ticker dict)
            ds = loader.dataset
            ticker_preds = ticker_preds_scaled
            ticker_targets = ticker_targets_scaled

            scaler = getattr(ds, 'target_scaler', None)
            if scaler is not None:
                try:
                    # Per-ticker scalers stored as dict
                    if isinstance(scaler, dict):
                        s = scaler.get(ticker)
                        if s is not None:
                            ticker_preds = s.inverse_transform(ticker_preds_scaled.reshape(-1, 1)).ravel()
                            ticker_targets = s.inverse_transform(ticker_targets_scaled.reshape(-1, 1)).ravel()
                    else:
                        # Single scaler for all samples
                        ticker_preds = scaler.inverse_transform(ticker_preds_scaled.reshape(-1, 1)).ravel()
                        ticker_targets = scaler.inverse_transform(ticker_targets_scaled.reshape(-1, 1)).ravel()
                except Exception as e:
                    print(f"Warning: Failed to inverse transform predictions for {ticker} on fold {fold_idx}: {e}")
                    ticker_preds = ticker_preds_scaled
                    ticker_targets = ticker_targets_scaled

            # Diagnostics: print scaled vs unscaled means/stds so we can tell if predictions were near-zero due to scaling
            try:
                print(f"  Fold {fold_idx} {ticker} scaled_pred mean={np.mean(ticker_preds_scaled):.6f}, std={np.std(ticker_preds_scaled):.6f}")
                print(f"  Fold {fold_idx} {ticker} unscaled_pred mean={np.mean(ticker_preds):.6f}, std={np.std(ticker_preds):.6f}")
                print(f"  Fold {fold_idx} {ticker} scaled_true mean={np.mean(ticker_targets_scaled):.6f}, std={np.std(ticker_targets_scaled):.6f}")
                print(f"  Fold {fold_idx} {ticker} unscaled_true mean={np.mean(ticker_targets):.6f}, std={np.std(ticker_targets):.6f}")
            except Exception:
                pass

            fold_data.append({
                'fold_idx': fold_idx,
                'dates': ticker_dates,
                'true_values': ticker_targets,
                'predictions': ticker_preds,
                'split': split
            })
        
        return fold_data
    
    def evaluate_all_folds(self, split: str = "test") -> List[Dict[str, float]]:
        """Evaluate metrics for all folds."""
        fold_metrics = []
        
        for fold_idx in range(len(self.folds)):
            train_loader, val_loader, test_loader = self.folds[fold_idx]
            
            if split == "train":
                loader = train_loader
            elif split == "val":
                loader = val_loader
            elif split == "test":
                if test_loader is None:
                    continue
                loader = test_loader
            else:
                continue
            
            model = self.load_fold_model(fold_idx)
            if model is None:
                continue
            
            # Compute unscaled metrics if possible (preferred) so MAE/RMSE are in original units
            y_scaler = getattr(loader.dataset, 'target_scaler', None)
            if y_scaler is not None:
                try:
                    metrics = compute_unscaled_metrics(model, loader, self.device, y_scaler)
                except Exception as e:
                    print(f"Warning: failed to compute unscaled metrics for fold {fold_idx}: {e}")
                    predictions, targets = get_predictions(model, loader, self.device)
                    metrics = compute_regression_metrics(predictions, targets, include_directional=True, include_r2=True, include_sharpe=True)
            else:
                predictions, targets = get_predictions(model, loader, self.device)
                metrics = compute_regression_metrics(predictions, targets, include_directional=True, include_r2=True, include_sharpe=True)

            fold_metrics.append(metrics)
            print(f"Fold {fold_idx}: MAE={metrics.get('mae', float('nan')):.4f}, R2={metrics.get('r2', float('nan')):.4f}")
        
        return fold_metrics
    
    def visualize_ticker(
        self, 
        ticker: str, 
        output_dir: Path,
        split: str = "test",
        show_sign_markers: bool = True,
        show_error_hist: bool = True,
    ):
        """Create visualizations for a specific ticker across all folds."""
        print(f"\nVisualizing {ticker} predictions across folds...")
        
        fold_data = self.collect_predictions_for_ticker(ticker, split)
        
        if not fold_data:
            print(f"No data found for ticker {ticker}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Full timeline across folds
        plot_rolling_predictions(
            ticker=ticker,
            fold_data=fold_data,
            save_path=output_dir / f"{ticker}_rolling_{split}.png",
            target_type=self.target_type
        )
        print(f"Saved plot to {output_dir / f'{ticker}_rolling_{split}.png'}")

        # Detailed time-series plot with sign markers and error histogram
        # Concatenate data across folds by date
        all_dates = []
        all_true = []
        all_pred = []
        for fd in fold_data:
            all_dates.extend(fd['dates'])
            all_true.extend(fd['true_values'])
            all_pred.extend(fd['predictions'])

        # Sort by date
        df = pd.DataFrame({'Date': pd.to_datetime(all_dates), 'True': all_true, 'Pred': all_pred}).sort_values('Date')

        from core.utils.plotting import plot_ticker_performance
        plot_ticker_performance(
            ticker=ticker,
            dates=df['Date'].tolist(),
            true_values=df['True'].tolist(),
            predicted_series=df['Pred'].tolist(),
            save_path=output_dir / f"{ticker}_detailed_{split}.png",
            target_type=self.target_type,
            show_sign_markers=show_sign_markers,
            show_error_hist=show_error_hist
        )
        print(f"Saved detailed plot to {output_dir / f'{ticker}_detailed_{split}.png'}")
    
    def visualize_fold_metrics(self, output_dir: Path, split: str = "test"):
        """Create visualization of metrics across folds."""
        print(f"\nEvaluating all folds on {split} set...")
        
        fold_metrics = self.evaluate_all_folds(split)
        
        if not fold_metrics:
            print("No fold metrics to visualize")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_fold_comparison(
            fold_metrics=fold_metrics,
            save_path=output_dir / f"fold_metrics_{split}.png"
        )
        
        print(f"Saved fold metrics plot to {output_dir / f'fold_metrics_{split}.png'}")
        
        from numbers import Number

        avg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics if key in m]
            # Keep only numeric (scalar) entries for aggregation
            numeric_values = [float(v) for v in values if isinstance(v, Number)]
            if not numeric_values:
                continue

            # Convert aggregated numpy types to Python native floats for JSON
            avg_metrics[key] = {
                'mean': float(np.mean(numeric_values)),
                'std': float(np.std(numeric_values)),
                'min': float(np.min(numeric_values)),
                'max': float(np.max(numeric_values)),
            }

        print("\nAverage metrics across folds:")
        for metric, stats in avg_metrics.items():
            print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")

        summary_path = output_dir / f"fold_summary_{split}.json"
        with open(summary_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"\nSaved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rolling fold predictions"
    )
    
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment directory containing fold checkpoints")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Dataset directory with rolling folds")
    parser.add_argument("--ticker", type=str, default="AAPL",
                       help="Ticker symbol to visualize")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Data split to visualize")
    parser.add_argument("--output-dir", type=str, default="figures/rolling_folds",
                       help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--all-metrics", action="store_true",
                       help="Also generate fold metrics comparison plot")
    parser.add_argument("--sign-markers", action="store_true",
                       help="Enable sign markers on detailed ticker plots")
    parser.add_argument("--error-hist", action="store_true",
                       help="Include error histogram on detailed ticker plots")

    args = parser.parse_args()
    
    print("=" * 80)
    print("ROLLING FOLD VISUALIZATION")
    print("=" * 80)
    
    visualizer = RollingFoldVisualizer(
        experiment_dir=args.experiment,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    try:
        visualizer.setup()
        
        output_dir = Path(args.output_dir)
        
        visualizer.visualize_ticker(
            ticker=args.ticker,
            output_dir=output_dir,
            split=args.split,
            show_sign_markers=args.sign_markers,
            show_error_hist=args.error_hist,
        )
        
        if args.all_metrics:
            visualizer.visualize_fold_metrics(
                output_dir=output_dir,
                split=args.split
            )
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
