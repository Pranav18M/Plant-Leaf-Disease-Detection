"""
Batch Training Script
Allows training with different parameter configurations
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.train import main as train_main


def save_experiment_config(exp_name, params, results):
    """Save experiment configuration and results"""
    exp_dir = os.path.join(config.RESULTS_DIR, 'experiments')
    os.makedirs(exp_dir, exist_ok=True)
    
    exp_file = os.path.join(exp_dir, f"{exp_name}.json")
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'results': results
    }
    
    with open(exp_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Experiment saved to {exp_file}")


def run_experiment(exp_name, params):
    """Run a single experiment with given parameters"""
    print(f"\n{'#'*60}")
    print(f"# Running Experiment: {exp_name}")
    print(f"{'#'*60}\n")
    
    # Update config with experiment parameters
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Set {key} = {value}")
    
    print()
    
    # Run training
    start_time = time.time()
    
    try:
        # This would call the training function
        # For now, just print the configuration
        print("Training with configuration:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # In actual use, call: train_main()
        # accuracy = train_main()
        
        accuracy = 0.85  # Placeholder
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        accuracy = 0.0
    
    elapsed_time = time.time() - start_time
    
    results = {
        'accuracy': accuracy,
        'training_time': elapsed_time
    }
    
    save_experiment_config(exp_name, params, results)
    
    return results


def main():
    """Run multiple experiments with different configurations"""
    
    experiments = [
        {
            'name': 'baseline',
            'params': {
                'SVM_KERNEL': 'rbf',
                'SVM_C': 1.0,
                'COLOR_SPACE': 'LAB',
                'KMEANS_CLUSTERS': 3
            }
        },
        {
            'name': 'linear_kernel',
            'params': {
                'SVM_KERNEL': 'linear',
                'SVM_C': 1.0,
                'COLOR_SPACE': 'LAB',
                'KMEANS_CLUSTERS': 3
            }
        },
        {
            'name': 'ycrcb_colorspace',
            'params': {
                'SVM_KERNEL': 'rbf',
                'SVM_C': 1.0,
                'COLOR_SPACE': 'YCrCb',
                'KMEANS_CLUSTERS': 3
            }
        },
        {
            'name': 'high_c_parameter',
            'params': {
                'SVM_KERNEL': 'rbf',
                'SVM_C': 10.0,
                'COLOR_SPACE': 'LAB',
                'KMEANS_CLUSTERS': 3
            }
        }
    ]
    
    print("\n" + "="*60)
    print("Batch Training - Multiple Experiments")
    print("="*60)
    print(f"\nTotal experiments: {len(experiments)}")
    print("="*60 + "\n")
    
    all_results = {}
    
    for exp in experiments:
        results = run_experiment(exp['name'], exp['params'])
        all_results[exp['name']] = results
        print(f"\nExperiment '{exp['name']}' completed!")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Time: {results['training_time']:.2f}s")
        print("-" * 60)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of All Experiments")
    print("="*60)
    
    for name, results in all_results.items():
        print(f"{name:20s}: {results['accuracy']*100:6.2f}% ({results['training_time']:6.2f}s)")
    
    print("="*60 + "\n")
    
    # Find best experiment
    best_exp = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best experiment: {best_exp[0]}")
    print(f"Best accuracy: {best_exp[1]['accuracy']*100:.2f}%\n")


if __name__ == "__main__":
    main()