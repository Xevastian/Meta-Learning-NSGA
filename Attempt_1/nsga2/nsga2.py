import os
import random
import copy
import warnings
# import matplotlib.pyplot as plt
import pickle
import numpy as np

# Import modules with fallbacks for relative imports
try:
    from .models import Model
    from .trainer import Trainer
    from .meta_learner import MetaLearner
except ImportError:
    # Fallback for when run as script
    from models import Model
    from trainer import Trainer
    from meta_learner import MetaLearner

warnings.filterwarnings("ignore")


def dominates(a, b):
    """Return True if individual a dominates b. a,b are dicts with 'accuracy' (maximize) and 'size' (minimize)."""
    return (a['accuracy'] >= b['accuracy'] and a['size'] <= b['size']) and (
        a['accuracy'] > b['accuracy'] or a['size'] < b['size']
    )


def nondominated_sort(pop):
    """Perform non-dominated sorting on population (list of dicts). Return list of fronts (each front is list of individuals)."""
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

    # convert indices to individuals and drop last empty front
    result = []
    for front in fronts[:-1]:
        result.append([copy.deepcopy(pop[idx]) for idx in front])
    return result


def crowding_distance(front):
    """Compute crowding distance for a front (list of individuals). Adds/sets 'cd' key."""
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind['cd'] = 0.0

    # accuracy (maximize) -> sort descending for distances but compute normalized differences
    accs = [ind['accuracy'] for ind in front]
    sizes = [ind['size'] for ind in front]
    acc_min, acc_max = min(accs), max(accs)
    size_min, size_max = min(sizes), max(sizes)

    # accuracy: rank extremes infinite
    front.sort(key=lambda x: x['accuracy'], reverse=True)
    front[0]['cd'] = float('inf')
    front[-1]['cd'] = float('inf')
    if acc_max != acc_min:
        for i in range(1, l - 1):
            front[i]['cd'] += (front[i - 1]['accuracy'] - front[i + 1]['accuracy']) / (acc_max - acc_min)

    # size: minimize -> sort ascending
    front.sort(key=lambda x: x['size'])
    front[0]['cd'] = float('inf')
    front[-1]['cd'] = float('inf')
    if size_max != size_min:
        for i in range(1, l - 1):
            front[i]['cd'] += (front[i + 1]['size'] - front[i - 1]['size']) / (size_max - size_min)


def compute_hypervolume(pareto_front, ref_point=None):
    """
    Compute the 2D hypervolume of a Pareto front.

    This implementation assumes objectives:
      - accuracy (maximize)
      - size (minimize)

    The hypervolume is computed exactly in 2D by sweeping along the size axis.
    """
    if not pareto_front:
        return 0.0

    # Filter out points with infinite or invalid values
    valid_front = []
    for p in pareto_front:
        acc = p['accuracy']
        size = p['size']
        if isinstance(acc, (int, float)) and isinstance(size, (int, float)) and not (np.isinf(acc) or np.isnan(acc) or np.isinf(size) or np.isnan(size)):
            valid_front.append(p)
    
    if not valid_front:
        return 0.0

    accs = [p['accuracy'] for p in valid_front]
    sizes = [p['size'] for p in valid_front]

    if ref_point is None:
        ref_point = (min(accs) - 0.1, max(sizes) + 1000)

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


def tournament_selection(pop, k=2):
    """Binary tournament using rank then crowding distance. If rank not present, perform nondominated sort first."""
    if not pop:
        return None
    # ensure rank exists
    if 'rank' not in pop[0]:
        fronts = nondominated_sort(pop)
        for r, front in enumerate(fronts):
            for ind in front:
                ind['rank'] = r
        # flatten back to pop-shaped list preserving models etc.
        # We'll just replace pop contents for selection convenience
        pop[:] = [ind for front in fronts for ind in front]

    competitors = random.sample(pop, k)
    # lower rank better
    competitors.sort(key=lambda x: (x.get('rank', 0), -x.get('cd', 0)))
    return competitors[0]


def evaluate_model(model, data_path, verbose=False):
    """Train model via Trainer and return (accuracy, size). Print model info and serialized size when verbose=True."""
    try:
        trainer = Trainer(model, data_path)
        acc = float(trainer.getAccuracy())
        size = float(trainer.getSize())

        # try to get params and name from model
        try:
            params = model.getModelParams()
        except Exception:
            try:
                params = trainer.getModel().getModelParams()
            except Exception:
                params = {}

        try:
            name = model.getModelName()
        except Exception:
            try:
                name = trainer.getModelName()
            except Exception:
                name = "Model"

        # measure serialized size (bytes) via pickle
        storage_bytes = None
        try:
            ser = pickle.dumps(trainer.getModel())
            storage_bytes = len(ser)
            storage_kb = storage_bytes / 1024
            storage_mb = storage_kb / 1024
        except Exception:
            storage_bytes = None

        if verbose:
            print(f"\nModel: {name}")
            print("Parameter: ")
            print(params)
            print(f"\nAccuracy: {acc}")
            print(f"Size (parameter count): {size}")
            if storage_bytes is not None:
                print(f"Serialized size: {storage_bytes} bytes ({storage_kb:.2f} KB, {storage_mb:.4f} MB)\n")
            else:
                print("Serialized size: unavailable (pickle failed)\n")

        # Defensive checks
        if acc is None or not isinstance(acc, (int, float)):
            acc = 0.0
        if size is None or not isinstance(size, (int, float)):
            size = float('inf')
        return acc, size
    except Exception as e:
        print(f"Evaluation failed for model {getattr(model, 'getModelName', lambda: 'Model')()}: {e}")
        return 0.0, 1e9


def nsga2(pop_size=20, generations=10, pm=0.3, data_path=None, plot_path='pareto_progression.png', 
          use_warm_start=True, meta_db_path='meta_knowledge.pkl', adaptive_operators=True, seed=67, 
          save_plot=True, show_plot=True):
    """
    NSGA-II with meta-learning enhancements.
    
    Args:
        pop_size: Population size
        generations: Number of generations
        pm: Base mutation probability
        data_path: Path to dataset
        plot_path: Path to save Pareto progression plot
        use_warm_start: If True, initialize with meta-knowledge
        meta_db_path: Path to meta-knowledge database
        adaptive_operators: If True, adjust mutation rate based on population diversity
        seed: Random seed for reproducibility (int or None)
        save_plot: If False, skip writing plot files (useful for headless runs)
        show_plot: If False, do not display the plot window (useful in non-interactive environments)
    
    Returns:
        Final population
    """
    if data_path is None:
        raise ValueError("data_path must be provided")
    
    # Initialize meta-learner
    meta_learner = MetaLearner(meta_db_path=meta_db_path, seed=seed)
    
    # Initialize population
    print("\n" + "="*60)
    print("NSGA-II with Meta-Learning")
    print("="*60)
    print(f"Use warm-start: {use_warm_start}")
    print(f"Adaptive operators: {adaptive_operators}")
    print("="*60 + "\n")
    
    if use_warm_start:
        print("Initializing population with meta-knowledge...")
        warm_pop = meta_learner.get_warm_start_population(pop_size)
        if warm_pop:
            population = warm_pop
            print(f"✓ Warm-started with {len(population)} solutions from meta-knowledge\n")
        else:
            print("✗ No meta-knowledge available, using random initialization\n")
            population = [Model() for _ in range(pop_size)]
    else:
        population = [Model() for _ in range(pop_size)]
    
    pop = []
    for m in population:
        acc, size = evaluate_model(m, data_path, verbose=False)
        pop.append({'model': m, 'accuracy': acc, 'size': size})

    pareto_history = []
    mutation_history = []
    diversity_history = []
    
    current_pm = pm  # Adaptive mutation rate

    for gen in range(generations):
        # Non-dominated sorting
        fronts = nondominated_sort(pop)
        
        # Compute population diversity
        diversity = meta_learner.compute_population_diversity(pop)
        diversity_history.append(diversity)
        
        # Adaptive mutation rate
        if adaptive_operators:
            current_pm = meta_learner.get_adaptive_mutation_rate(diversity)
        
        mutation_history.append(current_pm)

        # --- Print Pareto front for this generation ---
        print(f"\nGeneration {gen + 1}/{generations}")
        print(f"Population Diversity: {diversity:.3f}")
        print(f"Mutation Rate (Pm): {current_pm:.3f}")
        print(f"Pareto Front Size: {len(fronts[0]) if fronts else 0}")
        
        if fronts and len(fronts) > 0 and fronts[0]:
            for ind in fronts[0]:
                try:
                    name = ind['model'].getModelName()
                except Exception:
                    name = "Model"
                print(f"  → {name}: Acc={ind.get('accuracy'):.4f}, Size={ind.get('size'):.0f}")
        else:
            print("  (Empty Pareto front)")
        # ----------------------------------------------

        # record Pareto front (gen 0 included)
        pareto_history.append([{'accuracy': ind['accuracy'], 'size': ind['size']} for ind in fronts[0]] if fronts else [])
        
        # Add to meta-knowledge
        if fronts and fronts[0]:
            meta_learner.add_pareto_front(fronts[0], dataset_id=os.path.basename(data_path))

        # assign rank and crowding distance
        for r, front in enumerate(fronts):
            for ind in front:
                ind['rank'] = r
            crowding_distance(front)

        # create offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop)
            p2 = tournament_selection(pop)
            child1 = copy.deepcopy(p1['model'])
            child2 = copy.deepcopy(p2['model'])
            child1.mutate(current_pm)  # Use adaptive mutation rate
            child2.mutate(current_pm)
            for child in (child1, child2):
                acc, size = evaluate_model(child, data_path, verbose=False)
                offspring.append({'model': child, 'accuracy': acc, 'size': size})
                if len(offspring) >= pop_size:
                    break

        # combine and select next generation
        union = pop + offspring
        new_pop = []
        fronts = nondominated_sort(union)
        i = 0
        while i < len(fronts) and len(new_pop) + len(fronts[i]) <= pop_size:
            crowding_distance(fronts[i])
            new_pop.extend(fronts[i])
            i += 1

        if len(new_pop) < pop_size and i < len(fronts):
            crowding_distance(fronts[i])
            fronts[i].sort(key=lambda x: x.get('cd', 0), reverse=True)
            need = pop_size - len(new_pop)
            new_pop.extend(fronts[i][:need])

        pop = new_pop

    # final pareto record
    fronts = nondominated_sort(pop)
    final_pareto = [{'accuracy': ind['accuracy'], 'size': ind['size']} for ind in fronts[0]] if fronts else []
    pareto_history.append(final_pareto)
    
    # Compute and print final metrics
    if final_pareto:
        avg_accuracy = np.mean([ind['accuracy'] for ind in final_pareto])
        count_pareto = len(final_pareto)
        avg_size = np.mean([ind['size'] for ind in final_pareto])
        hypervolume = compute_hypervolume(final_pareto)
        
        print("\n" + "="*60)
        print("FINAL PARETO FRONT METRICS")
        print("="*60)
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Pareto Front Count: {count_pareto}")
        print(f"Hypervolume: {hypervolume:.4f}")
        print(f"Average Size: {avg_size:.0f}")
        print("="*60)
    else:
        print("\nNo Pareto front found!")
    
    # Add final front to meta-knowledge
    if fronts and fronts[0]:
        meta_learner.add_pareto_front(fronts[0], dataset_id=os.path.basename(data_path))
        meta_learner.save_meta_knowledge()
        meta_learner.export_meta_knowledge_summary()

    # Plot progression
    #fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pareto front progression
    #ax = axes[0, 0]
    #cmap = plt.cm.viridis
    gens = len(pareto_history)
    for idx, pareto in enumerate(pareto_history):
        if not pareto:
            continue
        accs = [p['accuracy'] for p in pareto]
        sizes = [p['size'] for p in pareto]
        #ax.scatter(sizes, accs, color=cmap(idx / max(1, gens - 1)), label=f'gen {idx}', alpha=0.7, s=30)
    # ax.set_xlabel('Size (lower is better)')
    # ax.set_ylabel('Accuracy (higher is better)')
    # ax.set_title('Pareto-front Progression')
    # if gens <= 12:
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # # Population diversity over generations
    # ax = axes[0, 1]
    # ax.plot(range(len(diversity_history)), diversity_history, marker='o', color='steelblue')
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Diversity')
    # ax.set_title('Population Diversity Evolution')
    # ax.grid(True, alpha=0.3)
    
    # Adaptive mutation rate over generations
    # ax = axes[1, 0]
    # ax.plot(range(len(mutation_history)), mutation_history, marker='s', color='coral')
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Mutation Rate (Pm)')
    # ax.set_title('Adaptive Mutation Rate')
    # ax.grid(True, alpha=0.3)
    # ax.axhline(y=pm, color='gray', linestyle='--', label=f'Base Pm={pm}')
    # ax.legend()
    
    # # Pareto front size over generations
    # ax = axes[1, 1]
    # pf_sizes = [len(p) for p in pareto_history]
    # ax.plot(range(len(pf_sizes)), pf_sizes, marker='^', color='darkgreen')
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Pareto Front Size')
    # ax.set_title('Pareto Front Size Evolution')
    # ax.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # if save_plot and plot_path:
    #     plt.savefig(plot_path, dpi=150)
    #     print(f"\n✓ Save visualization to {plot_path}")

    # if show_plot:
    #     try:
    #         plt.show()
    #     except Exception:
    #         pass

    # Close the figure to avoid memory buildup in long-running scripts
    # plt.close()

    return pop


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nsga2.py <data_path> [pop_size] [generations] [--no-warm-start] [--no-adaptive]")
        sys.exit(1)
    data = sys.argv[1]
    ps = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    gens = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    use_warm = '--no-warm-start' not in sys.argv
    use_adaptive = '--no-adaptive' not in sys.argv
    
    nsga2(pop_size=ps, generations=gens, data_path=data, 
          use_warm_start=use_warm, adaptive_operators=use_adaptive, seed=67)