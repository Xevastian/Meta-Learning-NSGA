import os
import random
import copy
import warnings
import matplotlib.pyplot as plt
import pickle
from models import Model
from trainer import Trainer

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


def evaluate_model(model, data_path, verbose=True):
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


def nsga2(pop_size=20, generations=10, pm=0.3, data_path=None, plot_path='pareto_progression.png'):
    if data_path is None:
        raise ValueError("data_path must be provided")

    # Initialize population
    population = [Model() for _ in range(pop_size)]
    pop = []
    for m in population:
        acc, size = evaluate_model(m, data_path)
        pop.append({'model': m, 'accuracy': acc, 'size': size})

    pareto_history = []

    for gen in range(generations):
        # Non-dominated sorting
        fronts = nondominated_sort(pop)

        # --- Print Pareto front for this generation ---
        print(f"\nGeneration {gen + 1}")
        if fronts and len(fronts) > 0 and fronts[0]:
            for ind in fronts[0]:
                try:
                    name = ind['model'].getModelName()
                except Exception:
                    name = "Model"
                try:
                    params = ind['model'].getModelParams()
                except Exception:
                    params = {}
                print("Model:", name)
                print("Params:", params)
                print(f"Accuracy: {ind.get('accuracy')}, Size: {ind.get('size')}\n")
        else:
            print("Pareto front: (empty)\n")
        # ----------------------------------------------

        # record Pareto front (gen 0 included)
        pareto_history.append([{'accuracy': ind['accuracy'], 'size': ind['size']} for ind in fronts[0]] if fronts else [])

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
            child1.mutate(pm)
            child2.mutate(pm)
            for child in (child1, child2):
                acc, size = evaluate_model(child, data_path)
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
    pareto_history.append([{'accuracy': ind['accuracy'], 'size': ind['size']} for ind in fronts[0]] if fronts else [])

    # Plot progression
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis
    gens = len(pareto_history)
    for idx, pareto in enumerate(pareto_history):
        if not pareto:
            continue
        accs = [p['accuracy'] for p in pareto]
        sizes = [p['size'] for p in pareto]
        plt.scatter(sizes, accs, color=cmap(idx / max(1, gens - 1)), label=f'gen {idx}', alpha=0.7, s=30)

    plt.xlabel('Size (lower is better)')
    plt.ylabel('Accuracy (higher is better)')
    plt.title('Pareto-front progression across generations')
    if gens <= 12:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass

    return pop


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nsga2.py <data_path> [pop_size] [generations]")
        sys.exit(1)
    data = sys.argv[1]
    ps = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    gens = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    nsga2(pop_size=ps, generations=gens, data_path=data)