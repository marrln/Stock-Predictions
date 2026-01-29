"""
Results Visualization & Analysis
=================================

Analyze and visualize results from training experiments.

Usage:
    # Visualize single experiment
    python3 visualize_results.py --experiment-dir results/my_experiment/experiment_20260129_143022
    
    # Compare multiple experiments
    python3 visualize_results.py --experiments-dir results/ablation_study/ablation_20260129_150000
    
    # Analyze ablation study
    python3 visualize_results.py --ablation-summary results/ablation_study/.../ablation_summary.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_experiment_results(experiment_dir: Path) -> Dict:
    """Load all results from an experiment directory."""
    
    results = {}
    
    # Load config
    config_path = experiment_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load metrics
    metrics_path = experiment_dir / 'metrics_test.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load history
    history_path = experiment_dir / 'history_test.csv'
    if history_path.exists():
        results['history'] = pd.read_csv(history_path)
    
    # Load baseline comparison
    baseline_path = experiment_dir / 'baseline_comparison.csv'
    if baseline_path.exists():
        results['baseline'] = pd.read_csv(baseline_path)
    
    # Load summary
    summary_path = experiment_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    return results


def plot_training_history(history: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot training and validation loss/metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # MAE
    axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training MAE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Training history plot → {save_path}")
    
    plt.show()


def plot_baseline_comparison(baseline_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot comparison with baseline models."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # MAPE
    axes[0].bar(baseline_df['Model'], baseline_df['MAPE'], 
                color=['#2ecc71' if m == 'LSTM' else '#95a5a6' for m in baseline_df['Model']])
    axes[0].set_ylabel('MAPE (%)')
    axes[0].set_title('Mean Absolute Percentage Error\n(Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(baseline_df['MAPE']):
        axes[0].text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # MAE
    axes[1].bar(baseline_df['Model'], baseline_df['MAE'], color=['#2ecc71' if m == 'LSTM' else '#95a5a6' for m in baseline_df['Model']])
    axes[1].set_ylabel('MAE ($)')
    axes[1].set_title('Mean Absolute Error\n(Lower is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(baseline_df['MAE']):
        axes[1].text(i, v + 0.1, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    axes[2].bar(baseline_df['Model'], baseline_df['R²'], color=['#2ecc71' if m == 'LSTM' else '#95a5a6' for m in baseline_df['Model']])
    axes[2].set_ylabel('R^2 Score')
    axes[2].set_title('R^2 Score\n(Higher is Better)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    for i, v in enumerate(baseline_df['R²']):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Baseline comparison plot → {save_path}")
    
    plt.show()


def plot_metrics_summary(metrics: Dict, save_path: Optional[Path] = None):
    """Plot summary of key metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. LSTM vs Baselines (MAPE)
    models = ['LSTM', 'Naive', 'MA', 'Momentum']
    mapes = [
        metrics.get('LSTM_MAPE', 0),
        metrics.get('Naive_MAPE', 0),
        metrics.get('MA_MAPE', 0),
        metrics.get('Momentum_MAPE', 0)
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    axes[0, 0].barh(models, mapes, color=colors)
    axes[0, 0].set_xlabel('MAPE (%)')
    axes[0, 0].set_title('MAPE Comparison (Lower is Better)')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(mapes):
        axes[0, 0].text(v + 0.1, i, f'{v:.2f}%', va='center', fontweight='bold')
    
    # 2. Improvement vs Baselines
    improvements = {
        'vs Naive': metrics.get('LSTM_vs_Naive_MAPE', 0),
        'vs MA': metrics.get('LSTM_vs_MA_MAPE', 0),
        'vs Momentum': metrics.get('LSTM_vs_Momentum_MAPE', 0)
    }
    
    colors_imp = ['#2ecc71' if v < 0 else '#e74c3c' for v in improvements.values()]
    axes[0, 1].barh(list(improvements.keys()), list(improvements.values()), color=colors_imp)
    axes[0, 1].set_xlabel('MAPE Difference (%)')
    axes[0, 1].set_title('LSTM Improvement over Baselines\n(Negative = Better)')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    for i, (k, v) in enumerate(improvements.items()):
        axes[0, 1].text(v - 0.2 if v < 0 else v + 0.2, i, f'{v:+.2f}%', va='center', ha='right' if v < 0 else 'left', fontweight='bold')
    
    # 3. Performance Metrics
    perf_metrics = {
        'MAE': metrics.get('LSTM_MAE', 0),
        'RMSE': metrics.get('LSTM_RMSE', 0),
        'MAPE (%)': metrics.get('LSTM_MAPE', 0)
    }
    
    axes[1, 0].bar(perf_metrics.keys(), perf_metrics.values(), color='#3498db')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('LSTM Performance Metrics')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(perf_metrics.items()):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. R² and Direction Accuracy
    quality_metrics = {
        'R² Score': metrics.get('LSTM_R2', 0),
        'Direction Acc (%)': metrics.get('Direction_Accuracy', 0) / 100
    }
    
    axes[1, 1].bar(quality_metrics.keys(), quality_metrics.values(), color=['#9b59b6', '#1abc9c'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Quality Metrics')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(quality_metrics.items()):
        display_val = f'{v:.4f}' if 'R^2' in k else f'{v*100:.1f}%'
        axes[1, 1].text(i, v + 0.02, display_val, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Metrics summary plot → {save_path}")
    
    plt.show()


def plot_ablation_results(ablation_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot ablation study results."""
    
    # Sort by LSTM_MAPE
    ablation_df = ablation_df.sort_values('LSTM_MAPE')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. MAPE comparison
    axes[0, 0].barh(ablation_df['Feature Set'], ablation_df['LSTM_MAPE'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(ablation_df))))
    axes[0, 0].set_xlabel('MAPE (%)')
    axes[0, 0].set_title('Feature Set Comparison: LSTM MAPE\n(Lower is Better)')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(ablation_df['LSTM_MAPE']):
        axes[0, 0].text(v + 0.05, i, f'{v:.2f}%', va='center', fontweight='bold')
    
    # 2. Improvement vs Naive
    axes[0, 1].barh(ablation_df['Feature Set'], ablation_df['vs_Naive'],
                     color=['#2ecc71' if v < 0 else '#e74c3c' for v in ablation_df['vs_Naive']])
    axes[0, 1].set_xlabel('MAPE Difference vs Naive (%)')
    axes[0, 1].set_title('Improvement over Naive Baseline\n(Negative = Better)')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(ablation_df['vs_Naive']):
        axes[0, 1].text(v - 0.1 if v < 0 else v + 0.1, i, f'{v:+.2f}%', 
                       va='center', ha='right' if v < 0 else 'left', fontweight='bold')
    
    # 3. Direction Accuracy
    axes[1, 0].barh(ablation_df['Feature Set'], ablation_df['Direction_Acc'],
                     color=plt.cm.plasma(np.linspace(0, 1, len(ablation_df))))
    axes[1, 0].set_xlabel('Direction Accuracy (%)')
    axes[1, 0].set_title('Direction Prediction Accuracy\n(Higher is Better)')
    axes[1, 0].axvline(x=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].legend()
    
    for i, v in enumerate(ablation_df['Direction_Acc']):
        axes[1, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 4. R² Score
    axes[1, 1].barh(ablation_df['Feature Set'], ablation_df['R2'],
                     color=plt.cm.cool(np.linspace(0, 1, len(ablation_df))))
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_title('R² Score (Higher is Better)')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(ablation_df['R2']):
        axes[1, 1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Ablation results plot → {save_path}")
    
    plt.show()


def create_summary_report(results: Dict, output_path: Optional[Path] = None):
    """Create text summary report."""
    
    metrics = results.get('metrics', {})
    config = results.get('config', {})
    
    report = []
    report.append("="*70)
    report.append("EXPERIMENT SUMMARY REPORT")
    report.append("="*70)
    
    # Experiment Info
    if config:
        report.append("\n[CONFIGURATION]")
        report.append(f"  Experiment: {config.get('experiment_name', 'N/A')}")
        report.append(f"  Timestamp: {config.get('timestamp', 'N/A')}")
        
        prep_config = config.get('preprocessing_config', {})
        report.append(f"  Sequence Length: {prep_config.get('seq_len', 'N/A')}")
        report.append(f"  Horizon: {prep_config.get('horizon', 'N/A')} days")
        report.append(f"  Task: {prep_config.get('task', 'N/A')}")
        report.append(f"  Feature Set: {prep_config.get('feature_set', 'N/A')}")
    
    # LSTM Performance
    report.append("\n[LSTM PERFORMANCE]")
    report.append(f"  MAE: ${metrics.get('LSTM_MAE', 0):.2f}")
    report.append(f"  RMSE: ${metrics.get('LSTM_RMSE', 0):.2f}")
    report.append(f"  MAPE: {metrics.get('LSTM_MAPE', 0):.2f}%")
    report.append(f"  R² Score: {metrics.get('LSTM_R2', 0):.4f}")
    report.append(f"  Direction Accuracy: {metrics.get('Direction_Accuracy', 0):.2f}%")
    
    # Baseline Comparisons
    report.append("\n[BASELINE COMPARISONS]")
    
    report.append("\n  Naive Baseline:")
    report.append(f"    MAPE: {metrics.get('Naive_MAPE', 0):.2f}%")
    report.append(f"    LSTM vs Naive: {metrics.get('LSTM_vs_Naive_MAPE', 0):+.2f}% " + 
                 ("✓ BETTER" if metrics.get('LSTM_vs_Naive_MAPE', 0) < 0 else "✗ WORSE"))
    
    report.append("\n  Moving Average Baseline:")
    report.append(f"    MAPE: {metrics.get('MA_MAPE', 0):.2f}%")
    report.append(f"    LSTM vs MA: {metrics.get('LSTM_vs_MA_MAPE', 0):+.2f}% " +
                 ("✓ BETTER" if metrics.get('LSTM_vs_MA_MAPE', 0) < 0 else "✗ WORSE"))
    
    report.append("\n  Momentum Baseline:")
    report.append(f"    MAPE: {metrics.get('Momentum_MAPE', 0):.2f}%")
    report.append(f"    LSTM vs Momentum: {metrics.get('LSTM_vs_Momentum_MAPE', 0):+.2f}% " +
                 ("✓ BETTER" if metrics.get('LSTM_vs_Momentum_MAPE', 0) < 0 else "✗ WORSE"))
    
    # Overall Assessment
    report.append("\n[ASSESSMENT]")
    
    r2 = metrics.get('LSTM_R2', 0)
    if r2 > 0.95:
        report.append("  R2 Score:     EXCELLENT (>0.95)")
    elif r2 > 0.80:
        report.append("  R2 Score:     GOOD (0.80-0.95)")
    elif r2 > 0.50:
        report.append("  R2 Score:     OK (0.50-0.80)")
    else:
        report.append("  R2 Score: (!) POOR (<0.50)")
    
    dir_acc = metrics.get('Direction_Accuracy', 0)
    if dir_acc > 60:
        report.append("  Direction Accuracy:     GOOD (>60%)")
    elif dir_acc > 55:
        report.append("  Direction Accuracy:     OK (55-60%)")
    else:
        report.append("  Direction Accuracy: (!) POOR (<55%)")
    
    vs_naive = metrics.get('LSTM_vs_Naive_MAPE', 0)
    if vs_naive < -2:
        report.append("  vs Naive: SIGNIFICANT IMPROVEMENT (>2% better)")
    elif vs_naive < 0:
        report.append("  vs Naive: IMPROVEMENT")
    else:
        report.append("  vs Naive: (!) NO IMPROVEMENT (worse than naive)")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\n[SAVED] Summary report → {output_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Visualize Training Results")
    
    parser.add_argument('--experiment-dir', type=str, help='Path to experiment directory')
    parser.add_argument('--ablation-summary', type=str, help='Path to ablation_summary.csv')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    if args.ablation_summary:
        # Ablation study visualization
        print(f"\n[INFO] Loading ablation results from {args.ablation_summary}")
        ablation_df = pd.read_csv(args.ablation_summary)
        
        save_path = None
        if args.save_plots:
            save_path = Path(args.ablation_summary).parent / 'ablation_plot.png'
        
        plot_ablation_results(ablation_df, save_path)
        
    elif args.experiment_dir:
        # Single experiment visualization
        experiment_dir = Path(args.experiment_dir)
        print(f"\n[INFO] Loading results from {experiment_dir}")
        
        results = load_experiment_results(experiment_dir)
        
        # Create summary report
        report_path = experiment_dir / 'summary_report.txt' if args.save_plots else None
        create_summary_report(results, report_path)
        
        # Plot training history
        if 'history' in results:
            save_path = experiment_dir / 'training_history.png' if args.save_plots else None
            plot_training_history(results['history'], save_path)
        
        # Plot baseline comparison
        if 'baseline' in results:
            save_path = experiment_dir / 'baseline_comparison.png' if args.save_plots else None
            plot_baseline_comparison(results['baseline'], save_path)
        
        # Plot metrics summary
        if 'metrics' in results:
            save_path = experiment_dir / 'metrics_summary.png' if args.save_plots else None
            plot_metrics_summary(results['metrics'], save_path)
    
    else:
        print("[ERROR] Specify --experiment-dir or --ablation-summary")


if __name__ == "__main__":
    main()
