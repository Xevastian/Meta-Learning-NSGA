"""
Meta-Learning NSGA-II Demo
===========================

This script demonstrates how meta-learning accelerates NSGA-II convergence.

The approach uses:
1. Warm-starting: Initialize population with previously found good solutions
2. Adaptive operators: Adjust mutation rate based on population diversity
3. Meta-knowledge base: Learn from previous optimization runs

Run this script to compare:
- Baseline NSGA-II (without meta-learning)
- Meta-Learning NSGA-II (with warm-start & adaptive operators)
"""

import os
import sys
import time
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import helpers: prefer relative imports when run as module, but fall back
# to absolute imports when the script is executed directly.
try:
    # Running as package: python -m nsga2.meta_learning_demo
    from .nsga2 import nsga2, nondominated_sort
    from .meta_learner import MetaLearner
except Exception:
    # Running from nsga2 directory: python meta_learning_demo.py
    from nsga2 import nsga2, nondominated_sort
    from meta_learner import MetaLearner


def hypervolume_indicator(front, ref_point=None):
    """Compute the 2D hypervolume of a Pareto front.

    This implementation assumes objectives:
      - accuracy (maximize)
      - size (minimize)

    The hypervolume is computed exactly in 2D by sweeping along the size axis.
    """
    if not front:
        return 0.0

    # Filter out points with infinite or invalid values
    valid_front = []
    for p in front:
        acc = p['accuracy']
        size = p['size']
        if isinstance(acc, (int, float)) and isinstance(size, (int, float)) and not (np.isinf(acc) or np.isnan(acc) or np.isinf(size) or np.isnan(size)):
            valid_front.append(p)
    
    if not valid_front:
        return 0.0

    accs = [p['accuracy'] for p in valid_front]
    sizes = [p['size'] for p in valid_front]

    if ref_point is None:
        # Determinism/comparability across runs: use a fixed reference point.
        # Objectives: accuracy in [0,1] (maximize), size >= 0 (minimize).
        ref_point = (0.0, 1e12)

    ref_acc, ref_size = ref_point

    # Sort by size (minimize), ascending.
    sorted_front = sorted(valid_front, key=lambda x: x['size'])

    hv = 0.0
    best_acc = ref_acc

    for p in sorted_front:
        acc = p['accuracy']
        size = p['size']

        # Only count if this point improves accuracy over previous points
        if acc <= best_acc:
            continue

        width = max(0.0, ref_size - size)
        height = acc - best_acc
        hv += width * height
        best_acc = acc

    return hv


def run_baseline_vs_meta_learning(data_path, pop_size=15, generations=8, num_runs=2):
    """
    Compare baseline NSGA-II vs Meta-Learning NSGA-II.
    
    Args:
        data_path: Path to dataset CSV
        pop_size: Population size
        generations: Number of generations
        num_runs: Number of sequential runs to perform
    """
    
    print("\n" + "="*70)
    print("META-LEARNING NSGA-II COMPARISON")
    print("="*70 + "\n")
    
    # Test data path
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Please provide a valid CSV file path")
        return
    
    print(f"Dataset: {os.path.basename(data_path)}")
    print(f"Population Size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Number of Sequential Runs: {num_runs}\n")
    
    results = {
        'baseline': [],
        'meta_learning': [],
        'speedup': []
    }
    
    # ============ RUN 1: Baseline (no meta-learning) ============
    print("\n" + "-"*70)
    print("RUN 1: BASELINE NSGA-II (No Meta-Learning)")
    print("-"*70 + "\n")
    
    # Remove meta-knowledge to start fresh
    if os.path.exists('meta_knowledge.pkl'):
        os.remove('meta_knowledge.pkl')
    
    start_time = time.time()
    pop_baseline = nsga2(
        pop_size=pop_size, 
        generations=generations, 
        data_path=data_path,
        plot_path='baseline_progression.png',
        use_warm_start=False,
        adaptive_operators=False,  # Don't use adaptive operators
        seed=67,
        save_plot=False,
        show_plot=False
    )
    baseline_time = time.time() - start_time
    
    # Extract Pareto front
    fronts_baseline = nondominated_sort(pop_baseline)
    pareto_baseline = [{'accuracy': ind['accuracy'], 'size': ind['size']} 
                       for ind in fronts_baseline[0]] if fronts_baseline else []
    hv_baseline = hypervolume_indicator(pareto_baseline)
    
    results['baseline'].append({
        'time': baseline_time,
        'pareto_size': len(pareto_baseline),
        'hypervolume': hv_baseline
    })
    
    print(f"\n✓ Baseline NSGA-II completed in {baseline_time:.2f}s")
    print(f"  - Pareto front size: {len(pareto_baseline)}")
    print(f"  - Hypervolume: {hv_baseline:.2f}")
    if pareto_baseline:
        print("  - Final Pareto Front:")
        for i, p in enumerate(sorted(pareto_baseline, key=lambda x: x['accuracy'], reverse=True)):
            print(f"    {i+1}. Accuracy: {p['accuracy']:.4f}, Size: {p['size']:.0f}")
    else:
        print("  - No valid Pareto front found.")
    
    # ============ RUN 2: Meta-Learning (warm-start + adaptive) ============
    print("\n" + "-"*70)
    print("RUN 2: META-LEARNING NSGA-II (With Warm-Start & Adaptive Operators)")
    print("-"*70 + "\n")
    
    start_time = time.time()
    pop_meta = nsga2(
        pop_size=pop_size, 
        generations=generations, 
        data_path=data_path,
        plot_path='meta_learning_progression.png',
        use_warm_start=True,  # Enable warm-start
        adaptive_operators=True,  # Enable adaptive operators
        seed=67,
        save_plot=False,
        show_plot=False
    )
    meta_time = time.time() - start_time
    
    # Extract Pareto front
    fronts_meta = nondominated_sort(pop_meta)
    pareto_meta = [{'accuracy': ind['accuracy'], 'size': ind['size']} 
                   for ind in fronts_meta[0]] if fronts_meta else []
    hv_meta = hypervolume_indicator(pareto_meta)
    
    results['meta_learning'].append({
        'time': meta_time,
        'pareto_size': len(pareto_meta),
        'hypervolume': hv_meta
    })
    
    speedup = baseline_time / meta_time
    results['speedup'].append(speedup)
    
    print(f"\n✓ Meta-Learning NSGA-II completed in {meta_time:.2f}s")
    print(f"  - Pareto front size: {len(pareto_meta)}")
    print(f"  - Hypervolume: {hv_meta:.2f}")
    if pareto_meta:
        print("  - Final Pareto Front:")
        for i, p in enumerate(sorted(pareto_meta, key=lambda x: x['accuracy'], reverse=True)):
            print(f"    {i+1}. Accuracy: {p['accuracy']:.4f}, Size: {p['size']:.0f}")
    else:
        print("  - No valid Pareto front found.")
    
    # ============ RESULTS SUMMARY ============
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Meta-Learning':<20} {'Improvement':<15}")
    print("-"*85)
    print(f"{'Execution Time (s)':<30} {baseline_time:<20.2f} {meta_time:<20.2f} {speedup:.2f}x faster" if speedup > 1 
          else f"{'Execution Time (s)':<30} {baseline_time:<20.2f} {meta_time:<20.2f} {1/speedup:.2f}x slower")
    print(f"{'Pareto Front Size':<30} {len(pareto_baseline):<20} {len(pareto_meta):<20} {len(pareto_meta) - len(pareto_baseline):+.0f}")
    print(f"{'Hypervolume':<30} {hv_baseline:<20.2f} {hv_meta:<20.2f} {hv_meta - hv_baseline:+.2f}")
    
    # Quality metric: average accuracy in Pareto front
    if pareto_baseline:
        avg_acc_baseline = np.mean([p['accuracy'] for p in pareto_baseline])
        avg_size_baseline = np.mean([p['size'] for p in pareto_baseline])
    else:
        avg_acc_baseline = 0
        avg_size_baseline = 0
    
    if pareto_meta:
        avg_acc_meta = np.mean([p['accuracy'] for p in pareto_meta])
        avg_size_meta = np.mean([p['size'] for p in pareto_meta])
    else:
        avg_acc_meta = 0
        avg_size_meta = 0
    
    print(f"{'Avg Accuracy in Pareto':<30} {avg_acc_baseline:<20.4f} {avg_acc_meta:<20.4f} {avg_acc_meta - avg_acc_baseline:+.4f}")
    print(f"{'Avg Size in Pareto':<30} {avg_size_baseline:<20.2f} {avg_size_meta:<20.2f} {avg_size_meta - avg_size_baseline:+.2f}")
    
    # ============ SEQUENTIAL RUNS (Demonstrating meta-knowledge accumulation) ============
    if num_runs > 2:
        print("\n" + "="*70)
        print(f"ADDITIONAL RUNS: Demonstrating Meta-Knowledge Accumulation")
        print("="*70 + "\n")
        
        for run_num in range(1, num_runs + 1):
            print(f"\nRun {run_num}: Meta-Learning NSGA-II (with accumulated meta-knowledge)")
            print("-"*70)
            
            start_time = time.time()
            pop_meta = nsga2(
                pop_size=pop_size, 
                generations=generations, 
                data_path=data_path,
                plot_path=f'meta_learning_run{run_num}.png',
                use_warm_start=True,
                adaptive_operators=True,
                seed=67,
                save_plot=False,
                show_plot=False
            )
            meta_time = time.time() - start_time
            
            fronts_meta = nondominated_sort(pop_meta)
            pareto_meta = [{'accuracy': ind['accuracy'], 'size': ind['size']} 
                           for ind in fronts_meta[0]] if fronts_meta else []
            hv_meta = hypervolume_indicator(pareto_meta)
            
            results['meta_learning'].append({
                'time': meta_time,
                'pareto_size': len(pareto_meta),
                'hypervolume': hv_meta,
                'Avg Accuracy': None
            })
            
            print(f"\n✓ Run {run_num} completed in {meta_time:.2f}s")
            print(f"  - Pareto front size: {len(pareto_meta)}")
            print(f"  - Hypervolume: {hv_meta:.2f}")
            if pareto_meta:
                print("  - Final Pareto Front:")
                for i, p in enumerate(sorted(pareto_meta, key=lambda x: x['accuracy'], reverse=True)):
                    print(f"    {i+1}. Accuracy: {p['accuracy']:.4f}, Size: {p['size']:.0f}")
            else:
                print("  - No valid Pareto front found.")
            print(f"  - Trend: {'↑' if hv_meta > results['meta_learning'][-2]['hypervolume'] else '↓'} Improving" 
                  if len(results['meta_learning']) > 1 else "  - First run (baseline)")
    
    # ============ VISUALIZATION (Disabled) ============
    print("\n" + "="*70)
    print("Visualization output is disabled (save_plot=False). No plot files were written.")
    print("="*70 + "\n")

    # Meta-knowledge summary (if present)
    meta_stats_path = 'meta_summary.txt'
    if os.path.exists(meta_stats_path):
        print(f"✓ Meta-knowledge summary saved to: {meta_stats_path}")
        with open(meta_stats_path, 'r') as f:
            summary = f.read()
            print("\n" + summary)

    return results


if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = 'Spam.csv'  # Change to your dataset
    POP_SIZE = 8
    GENERATIONS = 4
    NUM_RUNS = 2
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset '{DATA_PATH}' not found!")
        print("\nAvailable CSV files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        sys.exit(1)
    
    # Run comparison
    results = run_baseline_vs_meta_learning(
        data_path=DATA_PATH,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_runs=NUM_RUNS
    )
    
    print("\n" + "="*70)
    print("Meta-Learning NSGA-II Demo Complete!")
    print("="*70)