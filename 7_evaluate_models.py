#!/usr/bin/env python3
"""
Evaluate trained models and find the best one.

Usage:
    python3 7_evaluate_models.py --data-dir processed_data/5tickers_seq8
    python3 7_evaluate_models.py --single-model experiments/h128_l2_d0.2_lr0.001_b64/best.pt
"""

import argparse
from pathlib import Path

from core.evaluation import (
    compare_experiments,
    evaluate_single_checkpoint,
    ModelEvaluator
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and find the best one"
    )
    
    # Evaluation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--compare", action="store_true",
                          help="Compare all experiments (default)")
    mode_group.add_argument("--single-model", type=str,
                          help="Evaluate a single model checkpoint")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Dataset directory used for training")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--fold-idx", type=int, default=0,
                       help="Fold index to evaluate (for rolling CV)")
    
    # Experiment arguments
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Directory containing experiments (for compare mode)")
    parser.add_argument("--output-dir", type=str, default="experiments/comparison",
                       help="Output directory for results")
    parser.add_argument("--no-baselines", action="store_true",
                       help="Skip baseline evaluation")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu",
                       help="Device to use for evaluation (default: cpu)")
    
    args = parser.parse_args()
    
    if args.single_model:
        # Single model evaluation
        print(f"Evaluating single model: {args.single_model}")
        
        result = evaluate_single_checkpoint(
            checkpoint_path=args.single_model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=args.device
        )
        
        metrics = result.get("metrics", {})
        print(f"\nResults:")
        print(f"  MSE: {metrics.get('mse', 'N/A'):.6f}")
        print(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
        print(f"  Directional Accuracy: {metrics.get('dir_acc', 0.0):.2%}")
        
    else:
        # Compare all experiments
        best_name, best_result = compare_experiments(
            experiments_dir=args.experiments_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            include_baselines=not args.no_baselines,
            device=args.device
        )
        
        if best_name:
            print(f"\nNext step:")
            print(f"  python 8_compare_models.py --model experiments/{best_name}/best.pt --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()