"""
SMS-EMOA: S-Metric Selection NSGA-II with Meta-Learning Integration

Based on: Beume, N., Naujoks, B., & Emmerich, M. (2007).
"SMS-EMOA: Multiobjective selection based on dominated hypervolume."
European Journal of Operational Research.

Key Features:
- Hypervolume-based selection instead of dominance + crowding distance
- Steady-state evolution (offspring created one at a time)
- SBX crossover and polynomial mutation
- Meta-learning warm-start and adaptive operators
- Model/Trainer integration for multi-objective model optimization
"""

import os
import random
import copy
import warnings
import pickle
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
# Hypervolume Computation
# ============================================================================

def compute_2d_hypervolume(front: List[Dict], ref_point: Tuple[float, float]) -> float:
    """
    Compute exact 2D hypervolume for a list of individuals.
    
    Objectives:
    - accuracy (maximize) -> negate for minimization
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


def compute_hypervolume_contribution(individual: Dict, front_without_ind: List[Dict], 
                                     ref_point: Tuple[float, float]) -> float:
    """
    Compute the hypervolume contribution of an individual.
    
    This is the hypervolume gained by adding this individual to a front.
    
    Args:
        individual: Individual to evaluate
        front_without_ind: Front without this individual
        ref_point: Reference point
    
    Returns:
        Hypervolume contribution
    """
    hv_with = compute_2d_hypervolume(front_without_ind + [individual], ref_point)
    hv_without = compute_2d_hypervolume(front_without_ind, ref_point)
    return hv_with - hv_without


def select_worst_by_hypervolume(population: List[Dict], 
                                ref_point: Tuple[float, float]) -> int:
    """
    Select the individual from population whose removal causes MINIMAL hypervolume loss.
    
    This is the inverse: we select the one to REMOVE, not the one to keep.
    In SMS-EMOA, when population > pop_size, we remove the individual that
    contributes the least to the total hypervolume.
    
    Args:
        population: Current population (size = pop_size + 1)
        ref_point: Reference point for hypervolume
    
    Returns:
        Index of individual to remove
    """
    min_hv_loss = float('inf')
    worst_idx = 0
    
    for idx in range(len(population)):
        # Compute front without this individual
        front_without = [population[i] for i in range(len(population)) if i != idx]
        
        # Compute hypervolume loss (how much HV we lose by removing this individual)
        hv_loss = -compute_hypervolume_contribution(population[idx], front_without, ref_point)
        
        if hv_loss < min_hv_loss:
            min_hv_loss = hv_loss
            worst_idx = idx
    
    return worst_idx


def dominates(a: Dict, b: Dict) -> bool:
    """Check if individual a dominates b (for initial non-dominated population)."""
    return (a['accuracy'] >= b['accuracy'] and a['size'] <= b['size']) and (
        a['accuracy'] > b['accuracy'] or a['size'] < b['size']
    )


def nondominated_sort(pop: List[Dict]) -> List[List[Dict]]:
    """Perform non-dominated sorting to get initial Pareto front."""
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


def get_middle_seed() -> int:
    """Generate a new seed from middle digits of random number."""
    random_num = random.randint(10**15, 10**16 - 1)
    random_str = str(random_num)
    start_idx = (len(random_str) - 8) // 2
    end_idx = start_idx + 8
    middle_str = random_str[start_idx:end_idx]
    new_seed = int(middle_str)
    return new_seed if new_seed != 0 else 1


def crossover_models(parent_a: Model, parent_b: Model, child_seed: Optional[int] = None) -> Model:
    """
    Crossover operator for models (SBX-like crossover on hyperparameters).
    
    Args:
        parent_a: First parent
        parent_b: Second parent
        child_seed: Seed for child model
    
    Returns:
        Child model
    """
    name_a = parent_a.getModelName()
    name_b = parent_b.getModelName()
    params_a = parent_a.getModelParams() or {}
    params_b = parent_b.getModelParams() or {}

    if name_a == name_b and params_a and params_b:
        # Mix parameters with SBX-like distribution spread
        all_keys = set(params_a.keys()) | set(params_b.keys())
        child_params = {}
        for k in all_keys:
            if k in params_a and k in params_b:
                # SBX-like recombination: 50% chance to take from either parent
                child_params[k] = params_a[k] if random.random() < 0.5 else params_b[k]
            elif k in params_a:
                child_params[k] = params_a[k]
            else:
                child_params[k] = params_b[k]
        return Model.from_solution(name_a, child_params, seed=child_seed)

    # Type-level crossover: pick one parent entirely
    if random.random() < 0.5:
        return Model.from_solution(name_a, params_a, seed=child_seed)
    return Model.from_solution(name_b, params_b, seed=child_seed)


def sms_emoa(
    pop_size: int = 20,
    generations: int = 10,
    pm: float = 0.3,
    pc: float = 0.9,
    data_path: Optional[str] = None,
    seed: Optional[int] = None,
    hv_ref_point: Tuple[float, float] = (0.0, 1e12),
    verbose: bool = True
) -> List[Dict]:
    """
    SMS-EMOA: S-Metric Selection NSGA-II with Meta-Learning.
    
    Algorithm Overview:
    1. Initialize population of size N (with meta-learning warm-start)
    2. While generations not exhausted:
       a. Tournament selection of two parents
       b. Crossover with probability pc
       c. Mutation with adaptive rate pm
       d. Evaluate offspring
       e. Add offspring to population (size = N+1)
       f. Remove individual causing minimal hypervolume loss
    
    Args:
        pop_size: Population size (N)
        generations: Number of generations
        pm: Base mutation probability
        pc: Crossover probability
        data_path: Path to dataset
        use_warm_start: Use meta-learning warm-start
        meta_db_path: Path to meta-knowledge database
        adaptive_operators: Adjust mutation rate based on diversity
        seed: Random seed
        update_meta_db: Persist Pareto front to meta-knowledge
        dataset_similarity_threshold: Threshold for dataset matching
        hv_ref_point: Reference point for hypervolume computation
        verbose: Print progress
    
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
        print("SMS-EMOA (Baseline, No Meta-Learning)")
        print("S-Metric Selection NSGA-II")
        print("="*60)
        print(f"Population size: {pop_size}")
        print(f"Generations: {generations}")
        print(f"Mutation rate (Pm): {pm}")
        print(f"Crossover rate (Pc): {pc}")
        print("="*60 + "\n")
    
    def draw_model_seed():
        """Draw a new seed for model creation."""
        return random.randint(0, 2**31 - 2)
    
    # Initialize population randomly (baseline)
    population = [Model(seed=draw_model_seed()) for _ in range(pop_size)]
    
    # Evaluate initial population
    pop = []
    for m in population:
        acc, size = evaluate_model(m, data_path, verbose=False, random_state=base_seed)
        pop.append({'model': m, 'accuracy': acc, 'size': size})
    
    # Tracking metrics
    pareto_history = []
    hv_history = []
    _gen_seed = base_seed
    
    # Main loop
    for gen in range(generations):
    # --- Seed update (unchanged) ---
        _gen_seed = get_middle_seed() ^ (_gen_seed * 6364136223846793005 + 1442695040888963407) & (2**31 - 1)
        generation_seed = _gen_seed % (2**31 - 1)
        random.seed(generation_seed)
        np.random.seed(generation_seed)

        # --- Metrics (unchanged) ---
        pareto_front = nondominated_sort(pop)[0] if nondominated_sort(pop) else []
        hv = compute_2d_hypervolume(pareto_front, hv_ref_point)
        hv_history.append(hv)

        if verbose:
            print(f"\nGeneration {gen + 1}/{generations}")
            print(f"  Hypervolume: {hv:.2f}")
            print(f"  Pareto Front Size: {len(pareto_front)}")

        # --- Parent Selection ---
        def tournament_select():
            candidates = random.sample(pop, min(2, len(pop)))
            candidates.sort(key=lambda x: x['accuracy'], reverse=True)
            return candidates[0]['model']

        parent1 = tournament_select()
        parent2 = tournament_select()

        # --- Generate TWO offspring ---
        children = []

        for _ in range(2):
            child_seed = draw_model_seed()

            # Crossover
            if random.random() < pc:
                child = crossover_models(parent1, parent2, child_seed=child_seed)
            else:
                child = Model.from_solution(
                    parent1.getModelName(),
                    parent1.getModelParams(),
                    seed=child_seed
                )

            # Mutation
            child.mutate(pm)

            # Evaluate
            acc, size = evaluate_model(child, data_path, verbose=False, random_state=base_seed)

            children.append({'model': child, 'accuracy': acc, 'size': size})

        # --- Add BOTH offspring ---
        pop.extend(children)

        # --- Remove TWO worst individuals (SMS-EMOA style) ---
        while len(pop) > pop_size:
            worst_idx = select_worst_by_hypervolume(pop, hv_ref_point)
            pop.pop(worst_idx)

    # Final Pareto front
    if verbose:
        print("\n" + "="*60)
        print("FINAL PARETO FRONT SOLUTIONS")
        print("="*60)
    
    fronts = nondominated_sort(pop)
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
        
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Pareto Front Count: {len(final_pareto)}")
        print(f"Average Size: {format_size(avg_size)}")
        print(f"Hypervolume: {final_hv:.2f}")
        print(f"Max Hypervolume (history): {max(hv_history):.2f}")
        print("="*60)
    
    # Runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    if verbose:
        print(f"\nTotal Runtime: {total_runtime:.2f} seconds\n")
    
    return pop


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sms_emoa.py <data_path> [pop_size] [generations] [--seed <int>]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = None
    
    if '--seed' in sys.argv:
        seed_idx = sys.argv.index('--seed')
        seed = int(sys.argv[seed_idx + 1])
    
    result = sms_emoa(
        pop_size=pop_size,
        generations=generations,
        data_path=data_path,
        seed=seed,
        verbose=True
    )
    
    print("\nSMS-EMOA optimization completed successfully!")
