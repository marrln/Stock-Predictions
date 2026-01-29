"""
Experiment Setups with Results Saving
=====================================

Complete experiment suite with integrated results saving:
- Single experiments
- Ablation studies
- Multi-horizon studies
- Full experiment suite

All functions support ResultsSaver for automatic results tracking.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core_tf.data import PreprocessingConfig, PredictionTask
from core_tf.models import ModelConfig, get_config_preset
from .train import train_and_evaluate, load_multi_ticker_data
from .metrics import print_evaluation_results


# =============================================================================
# FEATURE SETS FOR ABLATION STUDY
# =============================================================================

FEATURE_SETS = {
    'price_only': {
        'cols': ['close'],
        'norm_cols': [0],
        'description': 'Only closing price'
    },
    'price_volume': {
        'cols': ['close', 'volume'],
        'norm_cols': [0, 1],
        'description': 'Price and volume'
    },
    'price_volume_sentiment': {
        'cols': ['close', 'volume', 'daily_sentiment'],
        'norm_cols': [0, 1],
        'description': 'Price, volume, and sentiment'
    },
    'all_features': {
        'cols': ['close', 'volume', 'daily_sentiment', 'n_articles'],
        'norm_cols': [0, 1],
        'description': 'All features including article count'
    },
    'sentiment_only': {
        'cols': ['close', 'daily_sentiment'],
        'norm_cols': [0],
        'description': 'Price and sentiment only (no volume)'
    }
}


def get_feature_set(name: str) -> dict:
    """Get predefined feature set for ablation study."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {name}. Choose from {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[name]


# =============================================================================
# RESULTS SAVER
# =============================================================================

class ResultsSaver:
    """Save training results in organized structure."""
    
    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"\n[INFO] Results will be saved to: {self.experiment_dir}")
    
    def save_config(self, prep_config: PreprocessingConfig, model_config: ModelConfig, 
                   feature_set: dict, args: dict):
        """Save experiment configuration."""
        config_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'preprocessing_config': {
                'seq_len': prep_config.seq_len,
                'horizon': prep_config.horizon,
                'task': prep_config.task.value if hasattr(prep_config.task, 'value') else str(prep_config.task),
                'feature_cols': prep_config.feature_cols,
                'cols_to_norm': prep_config.cols_to_norm,
                'n_articles_scale': prep_config.n_articles_scale
            },
            'model_config': {
                'seq_len': model_config.seq_len,
                'n_features': model_config.n_features,
                'lstm_units': model_config.lstm_units,
                'dropout': model_config.dropout,
                'l2_reg': model_config.l2_reg,
                'use_layer_norm': model_config.use_layer_norm,
                'task': model_config.task,
                'learning_rate': model_config.learning_rate
            },
            'feature_set': feature_set,
            'training_args': args
        }
        
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"[SAVED] Config → {config_path}")
    
    def save_results(self, results: Dict, split: str = 'test'):
        """Save evaluation results."""
        # Convert numpy types to Python types for JSON serialization
        results_json = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            else:
                results_json[k] = v
        
        # Save as JSON
        metrics_path = self.experiment_dir / f'metrics_{split}.json'
        with open(metrics_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"[SAVED] Metrics → {metrics_path}")
        
        # Save as CSV
        metrics_df = pd.DataFrame([{k: v for k, v in results_json.items() if not isinstance(v, (dict, list))}])
        csv_path = self.experiment_dir / f'metrics_{split}.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"[SAVED] Metrics CSV → {csv_path}")
    
    def save_baseline_comparison(self, results: Dict, task: str = 'price'):
        """Save baseline comparison table."""
        if task == 'direction':
            comparison_data = {
                'Model': ['LSTM', 'Majority Baseline'],
                'Accuracy': [
                    results.get('Accuracy', 0),
                    results.get('Baseline_Accuracy', 0)
                ],
                'vs_Baseline': [
                    results.get('vs_Baseline', 0),
                    0
                ]
            }
        else:  # price
            comparison_data = {
                'Model': ['LSTM', 'Naive'],
                'MAPE': [
                    results.get('MAPE', 0),
                    results.get('Naive_MAPE', 0)
                ],
                'R²': [
                    results.get('R2', 0),
                    0
                ]
            }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.experiment_dir / 'baseline_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"[SAVED] Baseline Comparison → {comparison_path}")
        
        # Print comparison
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print("="*60)
    
    def save_model(self, model):
        """Save trained model."""
        model_path = self.experiment_dir / 'model.keras'
        model.save(str(model_path))
        print(f"[SAVED] Model → {model_path}")
    
    def create_summary(self, results: Dict, task: str = 'price'):
        """Create summary report."""
        if task == 'direction':
            summary = {
                'Experiment': self.experiment_name,
                'Timestamp': datetime.now().isoformat(),
                'Task': task,
                'LSTM Performance': {
                    'Accuracy': f"{results.get('Accuracy', 0):.2f}%",
                    'AUC': f"{results.get('AUC', 0):.2f}%",
                    'F1': f"{results.get('F1', 0):.2f}%"
                },
                'vs Baseline': {
                    'Improvement': f"{results.get('vs_Baseline', 0):+.2f}%"
                }
            }
        else:  # price
            summary = {
                'Experiment': self.experiment_name,
                'Timestamp': datetime.now().isoformat(),
                'Task': task,
                'LSTM Performance': {
                    'MAPE': f"{results.get('MAPE', 0):.2f}%",
                    'R²': f"{results.get('R2', 0):.4f}",
                    'Direction Accuracy': f"{results.get('Direction_Accuracy', 0):.2f}%"
                },
                'vs Baseline': {
                    'MAPE_vs_Naive': f"{results.get('MAPE_vs_Naive', 0):+.2f}%"
                }
            }
        
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] Summary → {summary_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
        print("="*60)


# =============================================================================
# EXPERIMENT FUNCTIONS WITH SAVER
# =============================================================================

def run_single_experiment(args, saver: Optional[ResultsSaver] = None) -> Dict:
    """Run a single training experiment with optional results saving."""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: task={args.task}, horizon={args.horizon}")
    print("="*80)
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Get feature configuration
    if args.feature_set:
        fs = get_feature_set(args.feature_set)
        feature_cols = fs['cols']
        cols_to_norm = fs['norm_cols']
    else:
        feature_cols = [f.strip() for f in args.features.split(',')]
        cols_to_norm = [0, 1]  # Default: normalize close and volume
    
    # Preprocessing config
    task_enum = PredictionTask(args.task)
    prep_config = PreprocessingConfig(
        seq_len=args.seq_len,
        horizon=args.horizon,
        task=task_enum,
        feature_cols=feature_cols,
        cols_to_norm=cols_to_norm
    )
    
    # Load data
    print("\n1. Loading data...")
    data = load_multi_ticker_data(
        Path(args.data_dir),
        tickers,
        prep_config,
        verbose=args.verbose > 0
    )
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {data['train'][0].shape}")
    print(f"  X_val:   {data['val'][0].shape}")
    print(f"  X_test:  {data['test'][0].shape}")
    
    # Model config
    if args.lstm_units:
        lstm_units = [int(u) for u in args.lstm_units.split(',')]
    else:
        preset = get_config_preset(args.model_preset, args.task, args.seq_len, len(feature_cols))
        lstm_units = preset.lstm_units
    
    model_config = ModelConfig(
        seq_len=args.seq_len,
        n_features=len(feature_cols),
        lstm_units=lstm_units,
        task=args.task
    )
    
    print(f"\n2. Model configuration:")
    print(f"   Task: {args.task}")
    print(f"   Horizon: {args.horizon} day(s)")
    print(f"   Features: {feature_cols}")
    print(f"   LSTM units: {lstm_units}")
    
    # Save config if saver provided
    if saver:
        saver.save_config(prep_config, model_config, fs if args.feature_set else {}, vars(args))
    
    # Train and evaluate
    print("\n3. Training...")
    results, model, y_pred = train_and_evaluate(
        data, model_config, args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=args.verbose
    )
    
    # Save results if saver provided
    if saver:
        saver.save_results(results)
        saver.save_baseline_comparison(results, args.task)
        saver.save_model(model)
        saver.create_summary(results, args.task)
    
    # Print results
    print_evaluation_results(results, args.task)
    
    return {
        'task': args.task,
        'horizon': args.horizon,
        'features': feature_cols,
        'results': results,
        'model': model
    }


def run_ablation_study(args, saver: Optional[ResultsSaver] = None) -> List[Dict]:
    """Run ablation study on different feature sets with results saving."""
    
    print("\n" + "="*80)
    print("ABLATION STUDY - Feature Importance")
    print("="*80)
    
    all_results = []
    
    # Create parent saver if needed
    if saver is None and hasattr(args, 'save_results') and args.save_results:
        ablation_dir = getattr(args, 'output_dir', 'experiments') or 'experiments'
        saver = ResultsSaver(ablation_dir, 'ablation_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    for feature_set_name, fs in FEATURE_SETS.items():
        print(f"\n{'='*60}")
        print(f"Feature Set: {feature_set_name}")
        print(f"Description: {fs['description']}")
        print(f"Columns: {fs['cols']}")
        print(f"{'='*60}")
        
        # Override args
        original_feature_set = args.feature_set
        original_features = args.features
        
        args.feature_set = feature_set_name
        args.features = ','.join(fs['cols'])
        
        try:
            # Create sub-saver for this feature set
            if saver:
                feature_saver = ResultsSaver(saver.experiment_dir, f'feature_{feature_set_name}')
            else:
                feature_saver = None
            
            result = run_single_experiment(args, feature_saver)
            result['feature_set'] = feature_set_name
            all_results.append(result)
            
        except Exception as e:
            print(f"[ERROR] {feature_set_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Restore original args
            args.feature_set = original_feature_set
            args.features = original_features
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    summary_data = []
    for r in all_results:
        res = r['results']
        if args.task == 'direction':
            summary_data.append({
                'Feature Set': r['feature_set'],
                'Accuracy (%)': res.get('Accuracy', 0),
                'AUC (%)': res.get('AUC', 0),
                'vs Baseline': res.get('vs_Baseline', 0)
            })
        else:
            summary_data.append({
                'Feature Set': r['feature_set'],
                'MAPE (%)': res.get('MAPE', 0),
                'vs Naive': res.get('MAPE_vs_Naive', 0),
                'Dir Acc (%)': res.get('Direction_Accuracy', 0)
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save summary if saver provided
    if saver:
        summary_path = saver.experiment_dir / 'ablation_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[SAVED] Ablation Summary → {summary_path}")
    
    # Best feature set
    if args.task == 'direction':
        best_idx = df_summary['Accuracy (%)'].idxmax()
    else:
        best_idx = df_summary['MAPE (%)'].idxmin()
    
    print(f"\n  Best feature set: {df_summary.iloc[best_idx]['Feature Set']}")
    
    return all_results


def run_multi_horizon_study(args, saver: Optional[ResultsSaver] = None) -> List[Dict]:
    """Test multiple prediction horizons with results saving."""
    
    print("\n" + "="*80)
    print("MULTI-HORIZON STUDY")
    print("="*80)
    
    horizons = [1, 3, 5, 10]
    all_results = []
    
    # Create parent saver if needed
    if saver is None and hasattr(args, 'save_results') and args.save_results:
        horizon_dir = getattr(args, 'output_dir', 'experiments') or 'experiments'
        saver = ResultsSaver(horizon_dir, 'horizons_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Save original horizon
    original_horizon = args.horizon
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon} day(s)")
        print(f"{'='*60}")
        
        args.horizon = horizon
        
        try:
            # Create sub-saver for this horizon
            if saver:
                horizon_saver = ResultsSaver(saver.experiment_dir, f'horizon_{horizon}d')
            else:
                horizon_saver = None
            
            result = run_single_experiment(args, horizon_saver)
            all_results.append(result)
            
        except Exception as e:
            print(f"[ERROR] Horizon {horizon}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Restore original horizon
    args.horizon = original_horizon
    
    # Summary
    print("\n" + "="*80)
    print("MULTI-HORIZON SUMMARY")
    print("="*80)
    
    summary_data = []
    for r in all_results:
        res = r['results']
        if args.task == 'direction':
            summary_data.append({
                'Horizon': r['horizon'],
                'Accuracy (%)': res.get('Accuracy', 0),
                'vs Baseline': res.get('vs_Baseline', 0)
            })
        else:
            summary_data.append({
                'Horizon': r['horizon'],
                'MAPE (%)': res.get('MAPE', 0),
                'Naive MAPE (%)': res.get('Naive_MAPE', 0),
                'vs Naive': res.get('MAPE_vs_Naive', 0),
                'Dir Acc (%)': res.get('Direction_Accuracy', 0)
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save summary if saver provided
    if saver:
        summary_path = saver.experiment_dir / 'horizon_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[SAVED] Horizon Summary → {summary_path}")
    
    return all_results


def run_full_experiment_suite(args, saver: Optional[ResultsSaver] = None) -> Dict:
    """Run complete experiment suite with comprehensive results saving."""
    
    print("\n" + "#"*80)
    print("#" + " "*30 + "FULL EXPERIMENT SUITE" + " "*27 + "#")
    print("#"*80)
    
    # Create parent saver if needed
    if saver is None and hasattr(args, 'save_results') and args.save_results:
        suite_dir = getattr(args, 'output_dir', 'experiments') or 'experiments'
        saver = ResultsSaver(suite_dir, 'suite_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    all_results = {
        'price_experiments': [],
        'direction_experiments': [],
        'ablation': [],
        'multi_horizon': []
    }
    
    # Save original task
    original_task = args.task
    
    # 1. Direction prediction with different horizons
    print("\n" + "="*80)
    print("PART 1: Direction Prediction")
    print("="*80)
    
    args.task = 'direction'
    for horizon in [1, 5]:
        args.horizon = horizon
        args.feature_set = 'all_features'
        
        if saver:
            exp_saver = ResultsSaver(saver.experiment_dir, f'direction_h{horizon}')
        else:
            exp_saver = None
        
        result = run_single_experiment(args, exp_saver)
        all_results['direction_experiments'].append(result)
    
    # 2. Price prediction ablation
    print("\n" + "="*80)
    print("PART 2: Price Prediction - Ablation Study")
    print("="*80)
    
    args.task = 'price'
    args.horizon = 1
    
    if saver:
        ablation_saver = ResultsSaver(saver.experiment_dir, 'ablation')
    else:
        ablation_saver = None
    
    ablation_results = run_ablation_study(args, ablation_saver)
    all_results['ablation'] = ablation_results
    
    # 3. Multi-horizon (price)
    print("\n" + "="*80)
    print("PART 3: Multi-Horizon Price Prediction")
    print("="*80)
    
    args.feature_set = 'all_features'
    
    if saver:
        horizon_saver = ResultsSaver(saver.experiment_dir, 'multi_horizon')
    else:
        horizon_saver = None
    
    horizon_results = run_multi_horizon_study(args, horizon_saver)
    all_results['multi_horizon'] = horizon_results
    
    # Restore original task
    args.task = original_task
    
    # Final summary
    print("\n" + "#"*80)
    print("#" + " "*30 + "FINAL SUMMARY" + " "*35 + "#")
    print("#"*80)
    
    print("\n Direction Prediction Results:")
    for r in all_results['direction_experiments']:
        print(f"   Horizon {r['horizon']}: Accuracy={r['results']['Accuracy']:.1f}%, "
              f"vs Baseline={r['results']['vs_Baseline']:+.1f}%")
    
    print("\n Best Feature Set (Ablation):")
    if all_results['ablation']:
        best_ablation = min(all_results['ablation'], 
                          key=lambda x: x['results'].get('MAPE', float('inf')))
        print(f"   {best_ablation['feature_set']}: MAPE={best_ablation['results']['MAPE']:.2f}%")
    
    print("\n Key Findings:")
    
    # Check if direction beats baseline
    dir_results = all_results['direction_experiments']
    if dir_results:
        best_dir = max(dir_results, key=lambda x: x['results'].get('vs_Baseline', 0))
        if best_dir['results']['vs_Baseline'] > 2:
            print(f"     SUCCESS: Direction prediction beats baseline by {best_dir['results']['vs_Baseline']:.1f}%")
        else:
            print(f"   (!) Direction prediction does NOT significantly beat baseline")
    
    # Check if sentiment helps
    if all_results['ablation']:
        price_only = next((r for r in all_results['ablation'] 
                          if r['feature_set'] == 'price_only'), None)
        with_sent = next((r for r in all_results['ablation'] 
                         if r['feature_set'] == 'price_volume_sentiment'), None)
        
        if price_only and with_sent:
            mape_diff = price_only['results']['MAPE'] - with_sent['results']['MAPE']
            if mape_diff > 0.1:
                print(f"     SUCCESS: Sentiment improves MAPE by {mape_diff:.2f}%")
            else:
                print(f"   (!) Sentiment does NOT significantly improve predictions")
    
    # Save overall summary if saver provided
    if saver:
        overall_summary = {
            'experiment_suite': 'full',
            'timestamp': datetime.now().isoformat(),
            'direction_best': max(dir_results, key=lambda x: x['results']['vs_Baseline'])['results'] if dir_results else None,
            'ablation_best': min(all_results['ablation'], key=lambda x: x['results']['MAPE'])['results'] if all_results['ablation'] else None,
            'horizon_results': [{'horizon': r['horizon'], 'MAPE': r['results']['MAPE']} for r in all_results['multi_horizon']]
        }
        
        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        summary_path = saver.experiment_dir / 'suite_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(convert_numpy(overall_summary), f, indent=2)
        print(f"\n[SAVED] Suite Summary → {summary_path}")
    
    return all_results