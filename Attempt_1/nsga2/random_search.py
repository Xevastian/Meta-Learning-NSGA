"""
Random Search: Baseline Algorithm for Multi-Objective Optimization

Simple baseline that randomly samples models from the solution space
without any evolutionary mechanisms or intelligent search strategies.

Key Features:
- Uniform random sampling of models
- No selection or mating restrictions
- Purely exploratory (no exploitation)
- Good baseline for comparison with evolutionary algorithms
- Model/Trainer integration for multi-objective model optimization
"""

import os
import random
import copy
import warnings
import numpy as np
import time
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")


def format_size(size_bytes):
    """Convert bytes to human-readable format (KB or MB)"""
    if size_bytes == float('inf') or size_bytes < 0:
        return "N/A"
    if size_bytes >= 1024**2:  # MB
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:  # KB
        return f"{size_bytes / 1024:.2f} KB"
    else:  # Bytes
        return f"{size_bytes:.0f} B"

try:
    from .models import Model
    from .trainer import Trainer
except ImportError:
    from models import Model
    from trainer import Trainer


# ============================================================================
# Hypervolume Computation (for metrics)
# ============================================================================

def compute_2d_hypervolume(front: List[Dict], ref_point: Tuple[float, float]) -> float:
    """
    Compute exact 2D hypervolume for a list of individuals.
    
    Objectives:
    - accuracy (maximize)
    - size (minimize)
    
    Args:
        front: List of dicts with 'accuracy' and 'size' keys
        ref_point: Reference point (typically worst objective values)
    
    Returns:
        Hypervolume value
    """
    if not front:
        return 0.0
    
    # Filter valid points
    valid_front = []
    for p in front:
        acc = p['accuracy']
        size = p['size']
        if isinstance(acc, (int, float)) and isinstance(size, (int, float)):
            if not (np.isinf(acc) or np.isnan(acc) or np.isinf(size) or np.isnan(size)):
                valid_front.append(p)
    
    if not valid_front:
        return 0.0
    
    accs = [p['accuracy'] for p in valid_front]
    sizes = [p['size'] for p in valid_front]
    ref_acc, ref_size = ref_point
    
    # Sort by size (minimization objective)
    sorted_front = sorted(valid_front, key=lambda x: x['size'])
    
    hv = 0.0
    best_acc = ref_acc
    
    for p in sorted_front:
        acc = p['accuracy']
        size = p['size']
        
        if acc <= best_acc:
            continue
        
        width = max(0.0, ref_size - size)
        height = acc - best_acc
        hv += width * height
        best_acc = acc
    
    return hv


def dominates(a: Dict, b: Dict) -> bool:
    """Check if individual a dominates b."""
    return (a['accuracy'] >= b['accuracy'] and a['size'] <= b['size']) and (
        a['accuracy'] > b['accuracy'] or a['size'] < b['size']
    )


def nondominated_sort(pop: List[Dict]) -> List[List[Dict]]:
    """Perform non-dominated sorting to extract Pareto front."""
    S = {i: [] for i in range(len(pop))}
    n = {i: 0 for i in range(len(pop))}
    fronts = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)

    result = []
    for front in fronts[:-1]:
        result.append([copy.deepcopy(pop[idx]) for idx in front])
    return result


def evaluate_model(model: Model, data_path: str, verbose: bool = False, 
                   random_state: int = 42) -> Tuple[float, float]:
    """
    Train model via Trainer and return (accuracy, size).
    
    Args:
        model: Model to evaluate
        data_path: Path to dataset
        verbose: Print detailed info
        random_state: Random state for reproducibility
    
    Returns:
        (accuracy, size) tuple
    """
    try:
        trainer = Trainer(model, data_path, random_state=random_state)
        acc = float(trainer.getAccuracy())
        size = float(trainer.getSize())

        if verbose:
            try:
                params = model.getModelParams()
                name = model.getModelName()
                print(f"\nModel: {name}")
                print(f"Parameters: {params}")
                print(f"Accuracy: {acc:.4f}, Size: {size:.0f}\n")
            except Exception:
                pass

        if acc is None or not isinstance(acc, (int, float)):
            acc = 0.0
        if size is None or not isinstance(size, (int, float)):
            size = float('inf')
        return acc, size
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0, 1e9


# ============================================================================
# Random Search Algorithm
# ============================================================================

def random_search(
    n_samples: int = 200,
    data_path: Optional[str] = None,
    seed: Optional[int] = None,
    hv_ref_point: Tuple[float, float] = (0.0, 1e12),
    verbose: bool = True
) -> List[Dict]:
    """
    Random Search: Sample models uniformly at random from the solution space.
    
    Algorithm Overview:
    1. For n_samples iterations:
       a. Randomly generate a model
       b. Evaluate the model
       c. Add to population
    2. Extract and return non-dominated (Pareto) solutions
    
    This serves as a baseline for comparison with evolutionary algorithms.
    It provides an estimate of what uniform random sampling achieves.
    
    Args:
        n_samples: Number of random samples to generate
        data_path: Path to dataset (required)
        seed: Random seed for reproducibility
        hv_ref_point: Reference point for hypervolume computation
        verbose: Print progress information
    
    Returns:
        Final population (list of dicts with 'model', 'accuracy', 'size')
    """
    start_time = time.time()
    
    if data_path is None:
        raise ValueError("data_path must be provided")
    
    # Seed initialization
    if seed is None:
        seed = 67
        if verbose:
            print("No seed provided, using default seed 67")
    
    base_seed = int(seed) % (2**31 - 1)
    
    # Environment setup for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    random.seed(base_seed)
    np.random.seed(base_seed)
    
    if verbose:
        print(f"Random seed set to: {seed} (base_seed={base_seed})")
        print(f"First random verification: {random.random():.6f}, {np.random.random():.6f}")
    
    # Print header
    if verbose:
        print("\n" + "="*60)
        print("Random Search Baseline")
        print("Uniform Random Sampling (No Evolutionary Mechanisms)")
        print("="*60)
        print(f"Number of samples: {n_samples}")
        print("="*60 + "\n")
    
    def draw_model_seed():
        """Draw a new seed for model creation."""
        return random.randint(0, 2**31 - 2)
    
    # Random sampling loop
    population = []
    pareto_history = []
    hv_history = []
    
    for sample_idx in range(n_samples):
        # Generate random model
        model = Model(seed=draw_model_seed())
        
        # Evaluate model
        acc, size = evaluate_model(model, data_path, verbose=False, random_state=base_seed)
        population.append({'model': model, 'accuracy': acc, 'size': size})
        
        # Compute Pareto front every 10 samples
        if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
            fronts = nondominated_sort(population)
            pareto_front = fronts[0] if fronts else []
            hv = compute_2d_hypervolume(pareto_front, hv_ref_point)
            hv_history.append(hv)
            pareto_history.append(
                [{'accuracy': ind['accuracy'], 'size': ind['size']} for ind in pareto_front]
            )
            
            if verbose:
                print(f"Sample {sample_idx + 1}/{n_samples}: "
                      f"Population size={len(population)}, "
                      f"Pareto size={len(pareto_front)}, "
                      f"HV={hv:.2f}")
    
    # Final Pareto front
    if verbose:
        print("\n" + "="*60)
        print("FINAL PARETO FRONT SOLUTIONS")
        print("="*60)
    
    fronts = nondominated_sort(population)
    final_pareto = fronts[0] if fronts else []
    
    # Deduplicate
    _seen_final = set()
    _deduped_final = []
    for ind in final_pareto:
        try:
            _n = ind['model'].getModelName()
        except Exception:
            _n = 'Unknown'
        _k = (_n, round(ind['accuracy'], 8), round(ind['size'], 2))
        if _k not in _seen_final:
            _seen_final.add(_k)
            _deduped_final.append(ind)
    
    final_pareto = _deduped_final
    
    if verbose and final_pareto:
        for i, ind in enumerate(final_pareto, 1):
            try:
                name = ind['model'].getModelName()
            except Exception:
                name = "Model"
            print(f"\nSolution {i}: {name}")
            print(f"  Accuracy: {ind['accuracy']:.4f}")
            print(f"  Size: {format_size(ind['size'])}")
        
        print("\n" + "="*60)
        print("FINAL METRICS")
        print("="*60)
        avg_accuracy = np.mean([ind['accuracy'] for ind in final_pareto])
        avg_size = np.mean([ind['size'] for ind in final_pareto])
        final_hv = compute_2d_hypervolume(final_pareto, hv_ref_point)
        
        print(f"Total Samples: {n_samples}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Pareto Front Count: {len(final_pareto)}")
        print(f"Average Size: {format_size(avg_size)}")
        print(f"Hypervolume: {final_hv:.2f}")
        print(f"Max Hypervolume (history): {max(hv_history):.2f}" if hv_history else "N/A")
        print("="*60)
    
    # Runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    if verbose:
        print(f"\nTotal Runtime: {total_runtime:.2f} seconds\n")
    
    return population


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python random_search.py <data_path> [n_samples] [--seed <int>]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = None
    
    if '--seed' in sys.argv:
        seed_idx = sys.argv.index('--seed')
        seed = int(sys.argv[seed_idx + 1])
    
    result = random_search(
        n_samples=n_samples,
        data_path=data_path,
        seed=seed,
        verbose=True
    )
    
    print("\nRandom Search completed successfully!")
