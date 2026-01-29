"""
Training Script 
================================================

This script facilitates training financial prediction models using TensorFlow.
It supports various experiment setups including single experiments, ablation studies, multi-horizon studies, and full experiment suites. 
Results can be saved automatically.

Usage Examples:
--------------

# Single experiment with results saving
python3 tf_2_train.py --task price --horizon 1 --save-results

# Ablation study
python3 tf_2_train.py --ablation --save-results --output-dir results/ablation

# Multi-horizon study
python3 tf_2_train.py --multi-horizon --save-results --output-dir results/horizons

# Full experiment suite
python3 tf_2_train.py --full-experiment --save-results --output-dir results/full_suite

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse

# Import from core_tf modules
from core_tf.data import FEATURE_SETS
from core_tf.training import (
    ResultsSaver,
    run_single_experiment,
    run_ablation_study,
    run_multi_horizon_study,
    run_full_experiment_suite
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Financial Prediction Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data
    parser.add_argument("--data-dir", default="processed_data/csv_v3", help="Data directory")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA", help="Comma-separated tickers")
    # TODO: Add option to top N tickers based on data availability
    
    # Task configuration
    parser.add_argument("--task", choices=['price', 'direction', 'return'], default='price', help="Prediction task/type")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days ahead)")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length")
    
    # Features
    parser.add_argument("--features", default="close,volume,daily_sentiment,n_articles", help="Feature columns")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS.keys()), default=None, help="Use predefined feature set (overrides --features)")
    
    # Model
    parser.add_argument("--model-preset", choices=['small', 'medium', 'large'], default='medium', help="Model size preset")
    parser.add_argument("--lstm-units", default=None, help="Custom LSTM units (e.g., 64,64)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    # Experiments setups
    parser.add_argument("--ablation", action="store_true", help="Run ablation study on feature sets")
    parser.add_argument("--multi-horizon", action="store_true", help="Test multiple prediction horizons")
    parser.add_argument("--full-experiment", action="store_true", help="Run complete experiment suite")
    
    # Results saving
    parser.add_argument("--save-results", action="store_true", help="Save results to files")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom experiment name")
    
    # Output
    parser.add_argument("--verbose", type=int, default=1)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("FINANCIAL PREDICTION MODEL TRAINING")
    print("="*80)
    
    print("\nSettings:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Initialize saver (if results saving is enabled)
    saver = None
    if args.save_results:
        saver = ResultsSaver(args.output_dir, args.experiment_name)
    
    # Run experiments
    try:
        if args.full_experiment:
            results = run_full_experiment_suite(args, saver)
        elif args.ablation:
            results = run_ablation_study(args, saver)
        elif args.multi_horizon:
            results = run_multi_horizon_study(args, saver)
        else:
            results = run_single_experiment(args, saver)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    if args.save_results and saver:
        print(f"\n✓ Results saved to: {saver.experiment_dir}")


if __name__ == "__main__":
    main()