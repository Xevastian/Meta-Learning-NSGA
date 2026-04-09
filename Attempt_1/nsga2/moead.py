import os
import random
import warnings
import argparse
import numpy as np
import time
import concurrent.futures

# Import modules with fallbacks for relative imports
try:
    from .models import Model
    from .trainer import Trainer
except ImportError:
    from models import Model
    from trainer import Trainer

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


def _set_worker_thread_limits():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _normalize_n_jobs(n_jobs):
    if n_jobs is None:
        return max(1, min(4, os.cpu_count() or 1))
    return max(1, int(n_jobs))


def _evaluate_model_worker(args):
    model_name, model_params, data_path, random_state = args
    _set_worker_thread_limits()
    model = Model.from_solution(model_name, model_params)
    return evaluate_model(model, data_path, verbose=False, random_state=random_state)


def tchebycheff(individual, weights, ref_point):
    """
    Tchebycheff function for aggregation: max_i {w_i * |f_i - z_i|}
    where:
    - individual: dict with 'accuracy' (maximize) and 'size' (minimize)
    - weights: [w_acc, w_size] normalized so sum = 1
    - ref_point: [z_acc, z_size] ideal point for normalization
    
    Returns:
        Scalar fitness value (lower is better for minimization)
    """
    w_acc, w_size = weights
    z_acc, z_size = ref_point
    
    # For accuracy (maximize): normalize as -(accuracy - z_acc) so higher accuracy = lower fitness
    term_acc = w_acc * abs(-(individual['accuracy'] - z_acc))
    
    # For size (minimize): keep as is
    term_size = w_size * abs(individual['size'] - z_size)
    
    return max(term_acc, term_size)


def weighted_sum(individual, weights, ref_point=None):
    """
    Simple weighted sum aggregation: w_acc * accuracy - w_size * size
    where:
    - individual: dict with 'accuracy' (maximize) and 'size' (minimize)
    - weights: [w_acc, w_size] normalized so sum = 1
    - ref_point: (unused, kept for compatibility)
    
    Returns:
        Scalar fitness value (higher is better for maximization)
    """
    w_acc, w_size = weights
    return w_acc * individual['accuracy'] - w_size * individual['size']


def generate_weight_vectors(num_vectors, num_objectives=2, method='uniform'):
    """
    Generate weight vectors for decomposition.
    
    Args:
        num_vectors: Number of weight vectors to generate
        num_objectives: Number of objectives (default 2: accuracy, size)
        method: 'uniform' (uniform distribution), 'random' (random sampling)
    
    Returns:
        List of normalized weight vectors
    """
    weights = []
    
    if method == 'uniform':
        # Generate uniformly spaced weights for 2D
        if num_objectives == 2:
            for i in range(num_vectors):
                w_acc = i / max(1, num_vectors - 1) if num_vectors > 1 else 0.5
                w_size = 1.0 - w_acc
                weights.append(np.array([w_acc, w_size]))
    else:  # random
        for _ in range(num_vectors):
            w = np.random.dirichlet(np.ones(num_objectives))
            weights.append(w)
    
    return weights


def compute_neighborhoods(weight_vectors, neighbor_size):
    """
    Compute neighborhoods for all subproblems based on Euclidean distance
    between weight vectors, as specified in Algorithm 2.

    For each weight vector λⁱ, finds the T closest weight vectors
    (including itself) by Euclidean distance and stores their indices in B(i).

    Args:
        weight_vectors: List of weight vectors (numpy arrays)
        neighbor_size: Number of closest neighbors T (including self)

    Returns:
        List of lists: neighborhoods[i] = list of T neighbor indices for subproblem i
    """
    num_subproblems = len(weight_vectors)
    neighborhoods = []

    for i in range(num_subproblems):
        # Compute Euclidean distance from weight_vectors[i] to all others
        distances = [
            (np.linalg.norm(weight_vectors[i] - weight_vectors[j]), j)
            for j in range(num_subproblems)
        ]
        # Sort by distance, take the T closest (includes self at distance 0)
        distances.sort(key=lambda x: x[0])
        neighbors = [idx for _, idx in distances[:neighbor_size]]
        neighborhoods.append(neighbors)

    return neighborhoods


def evaluate_model(model, data_path, verbose=False, random_state=42):
    """Train model via Trainer and return (accuracy, size)."""
    try:
        trainer = Trainer(model, data_path, random_state=random_state)
        acc = float(trainer.getAccuracy())
        size = float(trainer.getSize())
        
        if acc is None or not isinstance(acc, (int, float)):
            acc = 0.0
        if size is None or not isinstance(size, (int, float)):
            size = float('inf')
        
        if verbose:
            try:
                name = model.getModelName()
                params = model.getModelParams()
                print(f"\nModel: {name}")
                print(f"Parameters: {params}")
                print(f"Accuracy: {acc}, Size: {size}\n")
            except Exception:
                pass
        
        return acc, size
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0, 1e9


def get_middle_seed():
    """Generate a new seed using middle digits extraction."""
    random_num = random.randint(10**15, 10**16 - 1)
    random_str = str(random_num)
    start_idx = (len(random_str) - 8) // 2
    end_idx = start_idx + 8
    middle_str = random_str[start_idx:end_idx]
    new_seed = int(middle_str)
    return new_seed if new_seed != 0 else 1


def crossover_models(parent_a, parent_b, child_seed=None):
    """Simple crossover operator over hyperparameter dictionaries."""
    name_a = parent_a.getModelName()
    name_b = parent_b.getModelName()
    params_a = parent_a.getModelParams() or {}
    params_b = parent_b.getModelParams() or {}

    if name_a == name_b and params_a and params_b:
        all_keys = set(params_a.keys()) | set(params_b.keys())
        child_params = {}
        for k in all_keys:
            if k in params_a and k in params_b:
                child_params[k] = params_a[k] if random.random() < 0.5 else params_b[k]
            elif k in params_a:
                child_params[k] = params_a[k]
            else:
                child_params[k] = params_b[k]
        return Model.from_solution(name_a, child_params, seed=child_seed)

    if random.random() < 0.5:
        return Model.from_solution(name_a, params_a, seed=child_seed)
    return Model.from_solution(name_b, params_b, seed=child_seed)


def _update_ep(EP, new_solutions):
    """
    Update the External Population (EP) with new solutions, as per Algorithm 2:
      - Remove from EP all vectors dominated by F(y).
      - Add F(y) to EP if no vector in EP dominates F(y).

    Args:
        EP: Current list of non-dominated archive solutions
        new_solutions: List of candidate solutions to consider adding

    Returns:
        Updated EP list
    """
    def dominates(a, b):
        """Return True if solution a dominates solution b."""
        return (a['accuracy'] >= b['accuracy'] and a['size'] <= b['size']) and \
               (a['accuracy'] > b['accuracy'] or a['size'] < b['size'])

    for candidate in new_solutions:
        # Remove all EP members dominated by the candidate
        EP = [ep_sol for ep_sol in EP if not dominates(candidate, ep_sol)]

        # Add candidate only if it is not dominated by any remaining EP member
        if not any(dominates(ep_sol, candidate) for ep_sol in EP):
            EP.append(candidate)

    # Deduplicate by (model name, accuracy, size)
    seen = set()
    deduped = []
    for ind in EP:
        try:
            name = ind['model'].getModelName()
        except Exception:
            name = 'Unknown'
        key = (name, round(ind['accuracy'], 8), round(ind['size'], 2))
        if key not in seen:
            seen.add(key)
            deduped.append(ind)

    return deduped


def moead(pop_size=20, generations=10, pm=0.3, pc=0.9, data_path=None,
          seed=None, aggregation='tchebycheff', neighbor_size=None,
          replacement_rate=0.9, save_plot=False, show_plot=False, verbose=True,
          n_jobs=None):
    """
    MOEA/D: Multiobjective Evolutionary Algorithm Based on Decomposition.
    
    The algorithm decomposes the MOP into a set of weighted scalar optimization subproblems,
    each solved independently with influence from neighboring subproblems.
    
    Args:
        pop_size: Number of subproblems (population size)
        generations: Number of generations
        pm: Base mutation probability
        pc: Crossover probability
        data_path: Path to dataset
        seed: Random seed for reproducibility
        aggregation: 'tchebycheff' or 'weighted_sum'
        neighbor_size: Size of neighborhood (default: pop_size // 5)
        replacement_rate: Proportion of neighbors to update (0-1)
        save_plot: Save Pareto progression plot
        show_plot: Display plot
    
    Returns:
        Final population (list of dicts with 'model', 'accuracy', 'size', 'subproblem_idx', 'weight')
    """
    start_time = time.time()
    
    if data_path is None:
        raise ValueError("data_path must be provided")
    
    # Initialize seeding
    if seed is None:
        seed = 67
        print("No seed provided, using default seed 67")
    
    base_seed = int(seed) % (2**31 - 1)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    random.seed(base_seed)
    np.random.seed(base_seed)
    
    print(f"Random seed set to: {seed} (base_seed={base_seed})")
    print(f"First random numbers for verification: {random.random():.6f}, {np.random.random():.6f}")
    
    n_jobs = _normalize_n_jobs(n_jobs)
    if verbose:
        print(f"Parallel workers: {n_jobs}")

    # Setup decomposition
    num_subproblems = pop_size
    weight_vectors = generate_weight_vectors(num_subproblems, num_objectives=2, method='uniform')

    if neighbor_size is None:
        neighbor_size = max(5, num_subproblems // 5)

    # --- Fix 1: Pre-compute all neighborhoods using Euclidean distance ---
    # B(i) = the T closest weight vectors to λⁱ, measured by Euclidean distance
    neighborhoods = compute_neighborhoods(weight_vectors, neighbor_size)

    print("\n" + "="*60)
    print("MOEA/D Baseline")
    print("="*60)
    print(f"Number of Subproblems: {num_subproblems}")
    print(f"Aggregation Function: {aggregation}")
    print(f"Neighborhood Size: {neighbor_size}")
    print(f"Replacement Rate: {replacement_rate:.2f}")
    print("="*60 + "\n")

    def draw_model_seed():
        """Draw a model-level seed."""
        return random.randint(0, 2**31 - 2)

    # Initialize population randomly
    population = [Model(seed=draw_model_seed()) for _ in range(num_subproblems)]

    # Evaluate initial population
    pop = []
    if n_jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            payloads = [
                (m.getModelName(), m.getModelParams(), data_path, base_seed)
                for m in population
            ]
            results = list(executor.map(_evaluate_model_worker, payloads))
        for i, (m, (acc, size)) in enumerate(zip(population, results)):
            pop.append({
                'model': m,
                'accuracy': acc,
                'size': size,
                'subproblem_idx': i,
                'weight': weight_vectors[i]
            })
    else:
        for i, m in enumerate(population):
            acc, size = evaluate_model(m, data_path, verbose=False, random_state=base_seed)
            pop.append({
                'model': m,
                'accuracy': acc,
                'size': size,
                'subproblem_idx': i,
                'weight': weight_vectors[i]
            })

    # Compute ideal reference point z from the initial population
    ideal_point = [max(ind['accuracy'] for ind in pop), min(ind['size'] for ind in pop)]

    # Select aggregation function
    if aggregation == 'tchebycheff':
        agg_func = tchebycheff
    else:
        agg_func = weighted_sum

    # --- Fix 2: Initialize EP = ∅ (External Population / Pareto archive) ---
    # EP stores all non-dominated solutions found so far across all generations.
    EP = []

    # Seed EP with the initial population's non-dominated solutions
    EP = _update_ep(EP, pop)

    current_pm = pm
    _gen_seed = base_seed

    # Main evolution loop
    for gen in range(generations):
        # Generate generation seed
        _gen_seed = get_middle_seed() ^ (_gen_seed * 6364136223846793005 + 1442695040888963407) & (2**31 - 1)
        generation_seed = _gen_seed % (2**31 - 1)
        random.seed(generation_seed)
        np.random.seed(generation_seed)

        print(f"Generation {gen + 1}/{generations} (seed: {generation_seed})")

        # Update ideal reference point z from current population
        current_accs = [ind['accuracy'] for ind in pop]
        current_sizes = [ind['size'] for ind in pop]
        ideal_point = [max(current_accs), min(current_sizes)]

        print(f"  Mutation Rate: {current_pm:.3f}")
        print(f"  Ideal Point: Acc={ideal_point[0]:.4f}, Size={format_size(ideal_point[1])}")

        children = []
        for i in range(num_subproblems):
            # B(i): neighborhood pre-computed via Euclidean distance (Fix 1)
            neighbors = neighborhoods[i]

            # Randomly select two indexes k, l from B(i)
            parent_indices = random.sample(neighbors, min(2, len(neighbors)))
            parent1_idx = parent_indices[0]
            parent2_idx = parent_indices[1] if len(parent_indices) > 1 else parent_indices[0]

            parent1 = pop[parent1_idx]['model']
            parent2 = pop[parent2_idx]['model']

            # Generate new solution y from xᵏ and xˡ via genetic operators
            child_seed_1 = draw_model_seed()
            if random.random() < pc:
                child = crossover_models(parent1, parent2, child_seed=child_seed_1)
            else:
                child = Model.from_solution(
                    parent1.getModelName(),
                    parent1.getModelParams(),
                    seed=child_seed_1
                )

            # Mutation
            child.mutate(current_pm)

            children.append({
                'model': child,
                'subproblem_idx': i,
                'weight': weight_vectors[i]
            })

        if n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                payloads = [
                    (child['model'].getModelName(), child['model'].getModelParams(), data_path, base_seed)
                    for child in children
                ]
                results = list(executor.map(_evaluate_model_worker, payloads))
            for child, (acc, size) in zip(children, results):
                child['accuracy'] = acc
                child['size'] = size
        else:
            for child in children:
                acc, size = evaluate_model(child['model'], data_path, verbose=False, random_state=base_seed)
                child['accuracy'] = acc
                child['size'] = size

        for child_solution in children:
            # Update z: ∀j=1…n, if z_j < f_j(y) then z_j = f_j(y)
            ideal_point[0] = max(ideal_point[0], child_solution['accuracy'])   # maximize accuracy
            ideal_point[1] = min(ideal_point[1], child_solution['size'])  # minimize size

            # Update neighboring solutions: for each j ∈ B(i),
            # if g^te(y | λʲ, z) ≤ g^te(xʲ | λʲ, z) then set xʲ = y
            neighbors = neighborhoods[child_solution['subproblem_idx']]
            num_updates = max(1, int(len(neighbors) * replacement_rate))
            neighbors_to_update = random.sample(neighbors, min(num_updates, len(neighbors)))

            for j in neighbors_to_update:
                neighbor_fitness = agg_func(pop[j], weight_vectors[j], ideal_point)
                child_fitness_j = agg_func(child_solution, weight_vectors[j], ideal_point)

                if child_fitness_j < neighbor_fitness:
                    pop[j] = child_solution.copy()
                    pop[j]['subproblem_idx'] = j
                    pop[j]['weight'] = weight_vectors[j]

            # --- Fix 2: Update EP ---
            # Remove from EP all vectors dominated by F(y).
            # Add F(y) to EP if no vector in EP dominates F(y).
            EP = _update_ep(EP, [child_solution])

        print(f"  EP Size: {len(EP)}")
        if EP:
            for ind in EP[:3]:  # Print top 3
                try:
                    name = ind['model'].getModelName()
                except Exception:
                    name = "Model"
                print(f"    * {name}: Acc={ind['accuracy']:.4f}, Size={format_size(ind['size'])}")
        print()

    # --- Fix 3: Output EP (not pop) ---
    if EP:
        avg_accuracy = np.mean([ind['accuracy'] for ind in EP])
        avg_size = np.mean([ind['size'] for ind in EP])

        print("\n" + "="*60)
        print("FINAL EP (EXTERNAL POPULATION) SOLUTIONS")
        print("="*60)

        for i, ind in enumerate(EP, 1):
            try:
                name = ind['model'].getModelName()
                params = ind['model'].getModelParams()
            except Exception:
                name = "Model"
                params = {}

            print(f"\nSolution {i}: {name}")
            print(f"  Accuracy: {ind['accuracy']:.4f}")
            print(f"  Size: {format_size(ind['size'])}")
            if params:
                print(f"  Configuration: {params}")

        print("\n" + "="*60)
        print("FINAL EP METRICS")
        print("="*60)
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"EP Size: {len(EP)}")
        print(f"Average Size: {format_size(avg_size)}")
        print("="*60)
    else:
        print("\nNo EP solutions found!")

    end_time = time.time()
    print(f"\nTotal Runtime: {end_time - start_time:.2f} seconds")

    # Return EP as specified in Algorithm 2 (not the raw population)
    return EP


def extract_pareto_front(population):
    """
    Extract Pareto optimal solutions from population.
    
    Args:
        population: List of solutions
    
    Returns:
        List of Pareto optimal solutions
    """
    def dominates(a, b):
        """Return True if a dominates b."""
        return (a['accuracy'] >= b['accuracy'] and a['size'] <= b['size']) and \
               (a['accuracy'] > b['accuracy'] or a['size'] < b['size'])
    
    pareto = []
    for p in population:
        is_dominated = False
        for q in population:
            if p is not q and dominates(q, p):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(p)
    
    # Deduplicate
    seen = set()
    deduped = []
    for ind in pareto:
        try:
            name = ind['model'].getModelName()
        except:
            name = 'Unknown'
        key = (name, round(ind['accuracy'], 8), round(ind['size'], 2))
        if key not in seen:
            seen.add(key)
            deduped.append(ind)
    
    return deduped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone MOEA/D runner')
    parser.add_argument('data_path', help='Path to dataset')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('generations', type=int, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pm', type=float, default=0.3, help='Mutation probability')
    parser.add_argument('--pc', type=float, default=0.9, help='Crossover probability')
    parser.add_argument('--aggregation', choices=['tchebycheff', 'weighted_sum'], default='tchebycheff', help='Aggregation function')
    parser.add_argument('--neighbor-size', type=int, default=None, help='Neighborhood size')
    parser.add_argument('--replacement-rate', type=float, default=0.9, help='Replacement rate')
    parser.add_argument('--save-plot', action='store_true', help='Save plot output')
    parser.add_argument('--show-plot', action='store_true', help='Show plot output')
    parser.add_argument('--verbose', action='store_true', help='Print verbose progress')

    args = parser.parse_args()
    moead(
        pop_size=args.pop_size,
        generations=args.generations,
        data_path=args.data_path,
        seed=args.seed,
        pm=args.pm,
        pc=args.pc,
        aggregation=args.aggregation,
        neighbor_size=args.neighbor_size,
        replacement_rate=args.replacement_rate,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
        verbose=args.verbose
    )