"""
Meta-Learning NSGA-II Improvement Demo
=======================================

This script demonstrates how meta-learning improves performance over multiple
sequential runs of NSGA-II.

The approach shows:
1. Warm-starting: Initialize population with previously found good solutions
2. Adaptive operators: Adjust mutation rate based on population diversity
3. Meta-knowledge accumulation: Learning improves over successive runs

Run this script to see:
- Baseline NSGA-II performance (single run)
- Meta-Learning NSGA-II improvement over multiple sequential runs
- Performance trends and speedup factors (printed to console)
"""

import os
import sys
import time
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import helpers: prefer relative imports when run as module, but fall back
# to absolute imports when the script is executed directly.
try:
    # Running as package: python -m nsga2.meta_learning_demo
    from .nsga2 import nsga2, nondominated_sort
    from .meta_learner import MetaLearner
except Exception:
    try:
        # Running from project root: python nsga2/meta_learning_demo.py
        from nsga2.nsga2 import nsga2, nondominated_sort
        from nsga2.meta_learner import MetaLearner
    except Exception:
        # Last resort: try importing local modules (if cwd is nsga2/)
        from nsga2 import nsga2 as _nsga2_mod
        nsga2 = getattr(_nsga2_mod, 'nsga2', None)
        nondominated_sort = getattr(_nsga2_mod, 'nondominated_sort', None)
        from meta_learner import MetaLearner


def hypervolume_indicator(front, ref_point=None):
    """
    Compute hypervolume indicator (approximate).
    Higher is better.
    
    Args:
        front: Pareto front (list of dicts with 'accuracy', 'size')
        ref_point: Reference point (default: worst point)
    
    Returns:
        Hypervolume value
    """
    if not front:
        return 0.0
    
    accs = [p['accuracy'] for p in front]
    sizes = [p['size'] for p in front]
    
    if ref_point is None:
        # Use a point worse than all solutions
        ref_point = (min(accs) - 0.1, max(sizes) + 1000)
    
    # Simple hypervolume approximation: area under the curve
    sorted_front = sorted(front, key=lambda x: x['accuracy'], reverse=True)
    hv = 0
    prev_size = ref_point[1]
    for p in sorted_front:
        width = prev_size - p['size']
        height = p['accuracy'] - ref_point[0]
        hv += width * height
        prev_size = p['size']
    
    return hv


def run_meta_learning_improvement(data_path, pop_size=15, generations=8, num_runs=5):
    """
    Demonstrate meta-learning improvement over multiple sequential runs.
    
    Args:
        data_path: Path to dataset CSV
        pop_size: Population size
        generations: Number of generations
        num_runs: Number of sequential meta-learning runs to perform
    """
    
    print("\n" + "="*70)
    print("META-LEARNING NSGA-II IMPROVEMENT OVER MULTIPLE RUNS")
    print("="*70 + "\n")
    
    # Test data path
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Please provide a valid CSV file path")
        return
    
    print(f"Dataset: {os.path.basename(data_path)}")
    print(f"Population Size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Number of Sequential Meta-Learning Runs: {num_runs}\n")
    
    results = {
        'baseline': None,
        'meta_runs': [],
        'improvement_trends': []
    }
    
    # ============ BASELINE RUN (No meta-learning) ============
    print("\n" + "-"*70)
    print("BASELINE: Standard NSGA-II (No Meta-Learning)")
    print("-"*70 + "\n")
    
    # Remove any existing meta-knowledge
    if os.path.exists('meta_knowledge.pkl'):
        os.remove('meta_knowledge.pkl')
    
    start_time = time.time()
    pop_baseline = nsga2(
        pop_size=pop_size, 
        generations=generations, 
        data_path=data_path,
        plot_path=None,  # No plotting
        use_warm_start=False,
        adaptive_operators=False
    )
    baseline_time = time.time() - start_time
    
    # Extract Pareto front
    fronts_baseline = nondominated_sort(pop_baseline)
    pareto_baseline = [{'accuracy': ind['accuracy'], 'size': ind['size']} 
                       for ind in fronts_baseline[0]] if fronts_baseline else []
    hv_baseline = hypervolume_indicator(pareto_baseline)
    
    results['baseline'] = {
        'time': baseline_time,
        'pareto_size': len(pareto_baseline),
        'hypervolume': hv_baseline,
        'avg_accuracy': np.mean([p['accuracy'] for p in pareto_baseline]) if pareto_baseline else 0
    }
    
    print(f"\n✓ Baseline completed in {baseline_time:.2f}s")
    print(f"  - Pareto front size: {len(pareto_baseline)}")
    print(f"  - Hypervolume: {hv_baseline:.2f}")
    print(f"  - Avg accuracy: {results['baseline']['avg_accuracy']:.4f}")
    
    # ============ MULTIPLE META-LEARNING RUNS ============
    print("\n" + "="*70)
    print("META-LEARNING RUNS: Demonstrating Knowledge Accumulation")
    print("="*70 + "\n")
    
    for run_num in range(1, num_runs + 1):
        print(f"\nRun {run_num}: Meta-Learning NSGA-II")
        print("-"*50)
        
        start_time = time.time()
        pop_meta = nsga2(
            pop_size=pop_size, 
            generations=generations, 
            data_path=data_path,
            plot_path=None,  # No plotting
            use_warm_start=True,
            adaptive_operators=True
        )
        meta_time = time.time() - start_time
        
        # Extract Pareto front
        fronts_meta = nondominated_sort(pop_meta)
        pareto_meta = [{'accuracy': ind['accuracy'], 'size': ind['size']} 
                       for ind in fronts_meta[0]] if fronts_meta else []
        hv_meta = hypervolume_indicator(pareto_meta)
        avg_acc_meta = np.mean([p['accuracy'] for p in pareto_meta]) if pareto_meta else 0
        
        run_result = {
            'run': run_num,
            'time': meta_time,
            'pareto_size': len(pareto_meta),
            'hypervolume': hv_meta,
            'avg_accuracy': avg_acc_meta
        }
        
        results['meta_runs'].append(run_result)
        
        # Calculate improvement over baseline
        time_improvement = results['baseline']['time'] / meta_time if meta_time > 0 else 1
        hv_improvement = hv_meta - hv_baseline
        acc_improvement = avg_acc_meta - results['baseline']['avg_accuracy']
        
        improvement = {
            'run': run_num,
            'time_speedup': time_improvement,
            'hv_improvement': hv_improvement,
            'acc_improvement': acc_improvement
        }
        results['improvement_trends'].append(improvement)
        
        print(f"\n✓ Run {run_num} completed in {meta_time:.2f}s")
        print(f"  - Pareto front size: {len(pareto_meta)}")
        print(f"  - Hypervolume: {hv_meta:.2f} ({hv_improvement:+.2f} vs baseline)")
        print(f"  - Avg accuracy: {avg_acc_meta:.4f} ({acc_improvement:+.4f} vs baseline)")
        print(f"  - Time: {time_improvement:.2f}x {'faster' if time_improvement > 1 else 'slower'} than baseline")
        
        # Show trend from previous run
        if run_num > 1:
            prev_hv = results['meta_runs'][-2]['hypervolume']
            trend = "↑ Improving" if hv_meta > prev_hv else "↓ Declining" if hv_meta < prev_hv else "→ Stable"
            print(f"  - Trend vs Run {run_num-1}: {trend}")
    
    # ============ PERFORMANCE ANALYSIS ============
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Extract trends
    runs = [r['run'] for r in results['meta_runs']]
    times = [r['time'] for r in results['meta_runs']]
    hypervolumes = [r['hypervolume'] for r in results['meta_runs']]
    accuracies = [r['avg_accuracy'] for r in results['meta_runs']]
    
    print(f"\nHypervolume Trend:")
    for i, hv in enumerate(hypervolumes):
        marker = "↑" if i > 0 and hv > hypervolumes[i-1] else "↓" if i > 0 and hv < hypervolumes[i-1] else "→"
        print(f"  Run {runs[i]}: {hv:.2f} {marker}")
    
    print(f"\nExecution Time Trend:")
    for i, t in enumerate(times):
        marker = "↓" if i > 0 and t < times[i-1] else "↑" if i > 0 and t > times[i-1] else "→"
        print(f"  Run {runs[i]}: {t:.2f}s {marker}")
    
    # Calculate overall improvement
    if num_runs > 1:
        first_run = results['meta_runs'][0]
        last_run = results['meta_runs'][-1]
        
        overall_hv_improvement = last_run['hypervolume'] - first_run['hypervolume']
        overall_time_improvement = first_run['time'] / last_run['time'] if last_run['time'] > 0 else 1
        
        print(f"\nOverall Meta-Learning Improvement (Run 1 → Run {num_runs}):")
        print(f"  - Hypervolume: {overall_hv_improvement:+.2f}")
        print(f"  - Execution Time: {overall_time_improvement:.2f}x {'faster' if overall_time_improvement > 1 else 'slower'}")
    
    # ============ VISUALIZATION ============
    print("\n" + "="*70)
    print("Performance Results Summary")
    print("="*70 + "\n")
    
    # Print detailed results table
    print(f"{'Run':<5} {'Time (s)':<10} {'Pareto Size':<12} {'Hypervolume':<12} {'Avg Accuracy':<14} {'Speedup':<8}")
    print("-" * 75)
    
    # Baseline
    print(f"{'Base':<5} {results['baseline']['time']:<10.2f} {results['baseline']['pareto_size']:<12} {results['baseline']['hypervolume']:<12.2f} {results['baseline']['avg_accuracy']:<14.4f} {'1.00x':<8}")
    
    # Meta runs
    for run in results['meta_runs']:
        speedup = results['baseline']['time'] / run['time'] if run['time'] > 0 else 1
        print(f"{run['run']:<5} {run['time']:<10.2f} {run['pareto_size']:<12} {run['hypervolume']:<12.2f} {run['avg_accuracy']:<14.4f} {speedup:<8.2f}x")
    
    # Skip plotting - just print results
    print("\n✓ Performance results displayed above")
    print("  (Plotting disabled for faster execution)")
    
    # Meta-knowledge summary
    meta_stats_path = 'meta_summary.txt'
    if os.path.exists(meta_stats_path):
        print(f"\n✓ Meta-knowledge summary saved to: {meta_stats_path}")
        with open(meta_stats_path, 'r') as f:
            summary = f.read()
            print("\nMeta-Knowledge Summary:")
            print("-" * 40)
            print(summary)
    
    return results


if __name__ == '__main__':
    # Configuration
    DATA_PATH = 'liver.csv'  # Change to your dataset
    POP_SIZE = 15
    GENERATIONS = 8
    NUM_RUNS = 5  # Number of sequential meta-learning runs
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset '{DATA_PATH}' not found!")
        print("\nAvailable CSV files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        sys.exit(1)
    
    # Run meta-learning improvement demonstration
    results = run_meta_learning_improvement(
        data_path=DATA_PATH,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_runs=NUM_RUNS
    )
    
    print("\n" + "="*70)
    print("Meta-Learning Improvement Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Meta-learning shows improvement over multiple sequential runs")
    print("2. Warm-starting with accumulated knowledge accelerates convergence")
    print("3. Adaptive operators help maintain population diversity")
    print("4. Performance typically improves as meta-knowledge accumulates")
    print("5. Best suited for optimization problems run repeatedly on similar datasets")
    print("\nFor more information, see: meta_learner.py and nsga2.py")
