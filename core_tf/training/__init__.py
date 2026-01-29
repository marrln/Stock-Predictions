"""
Training and experiment functions with integrated results saving.
"""

from .train import (
    load_multi_ticker_data,
    train_and_evaluate
)

from .experiment_setups import (
    ResultsSaver,
    run_single_experiment,
    run_ablation_study,
    run_multi_horizon_study,
    run_full_experiment_suite,
    FEATURE_SETS,
    get_feature_set
)

from .metrics import (
    evaluate_price_predictions,
    evaluate_direction_predictions,
    direction_baseline_majority,
    direction_baseline_random,
    naive_baseline,
    moving_average_baseline,
    momentum_baseline
)

__all__ = [
    # Training
    'load_multi_ticker_data',
    'train_and_evaluate',
    # Results Saving
    'ResultsSaver',    
    # Experiments
    'run_single_experiment',
    'run_ablation_study',
    'run_multi_horizon_study',
    'run_full_experiment_suite'
    # Metrics
    'evaluate_price_predictions',
    'evaluate_direction_predictions',
    'direction_baseline_majority',
    'direction_baseline_random',
    'naive_baseline',
    'moving_average_baseline',
    'momentum_baseline'
]