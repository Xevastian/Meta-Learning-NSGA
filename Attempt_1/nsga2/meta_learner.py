import json
import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd

# Import with fallback
try:
    from .models import Model
except ImportError:
    from models import Model


def _default_model_stats():
    """Top-level default factory for model statistics (picklable)."""
    return {'wins': 0, 'total': 0, 'avg_accuracy': 0}

class MetaLearner:
    """
    Meta-learning module that learns from previous NSGA-II optimization runs
    to enable faster convergence on new problems.
    """
    
    def __init__(self, meta_db_path='meta_knowledge.pkl', seed=None):
        """
        Initialize meta-learner.
        
        Args:
            meta_db_path: Path to store meta-knowledge database
            seed: Random seed for reproducibility
        """
        self.meta_db_path = meta_db_path
        self.meta_knowledge = {
            'solutions': [],  # List of good solutions
            'model_stats': defaultdict(_default_model_stats),
            'parameter_patterns': defaultdict(list),
            'dataset_signatures': {}  # Map dataset characteristics to solutions
        }
        self.load_meta_knowledge()
    
    def load_meta_knowledge(self):
        """Load existing meta-knowledge from disk."""
        if os.path.exists(self.meta_db_path):
            try:
                with open(self.meta_db_path, 'rb') as f:
                    self.meta_knowledge = pickle.load(f)
                print(f"[OK] Loaded meta-knowledge with {len(self.meta_knowledge['solutions'])} solutions")
            except Exception as e:
                print(f"Error loading meta-knowledge: {e}. Starting fresh.")

    def compute_dataset_signature(self, data_path):
        """Compute lightweight dataset signature to compare dataset similarity."""
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            n_samples, n_features = df.shape
            if 'label' in df.columns:
                labels = df['label']
            else:
                labels = df.iloc[:, -1]
            flag = labels.value_counts(normalize=True)
            # entropy for class balance
            class_entropy = -sum([p * np.log(p + 1e-12) for p in flag.values])
            # number of classes
            n_classes = flag.shape[0]
            signature = {
                'n_samples': float(n_samples),
                'n_features': float(n_features - 1),
                'n_classes': float(n_classes),
                'class_entropy': float(class_entropy),
                'class_balance': float(flag.max())
            }
            return signature
        except Exception as e:
            print(f"Warning: failed to compute dataset signature: {e}")
            return None

    def _signature_distance(self, sig_a, sig_b):
        """Compute normalized distance between two dataset signatures."""
        if not sig_a or not sig_b:
            return float('inf')
        keys = ['n_samples', 'n_features', 'n_classes', 'class_entropy', 'class_balance']
        diffs = []
        for k in keys:
            a = sig_a.get(k, 0.0)
            b = sig_b.get(k, 0.0)
            maxv = max(abs(a), abs(b), 1.0)
            diffs.append(((a - b) / maxv) ** 2)
        return float(np.sqrt(sum(diffs)))
    
    def save_meta_knowledge(self):
        """Save meta-knowledge to disk."""
        try:
            with open(self.meta_db_path, 'wb') as f:
                pickle.dump(self.meta_knowledge, f)
            print(f"[OK] Saved meta-knowledge with {len(self.meta_knowledge['solutions'])} solutions")
        except Exception as e:
            print(f"Error saving meta-knowledge: {e}")

    def compute_dataset_signature(self, data_path):
        """Compute dataset signature for similarity-based warm-start."""
        try:
            df = pd.read_csv(data_path) if isinstance(data_path, str) else data_path.copy()
            if 'label' not in df.columns:
                raise ValueError("Dataset must contain 'label' column")

            X = df.drop('label', axis=1)
            y = df['label']

            value_counts = y.value_counts(normalize=True).to_dict()
            entropy = -sum(p * np.log2(p) for p in value_counts.values() if p > 0)

            signature = {
                'n_samples': int(df.shape[0]),
                'n_features': int(X.shape[1]),
                'n_classes': int(y.nunique()),
                'class_entropy': float(entropy)
            }
            return signature
        except Exception as e:
            print(f"Error computing dataset signature: {e}")
            return None

    @staticmethod
    def _dataset_signature_similarity(sig_a, sig_b):
        """Compute similarity score in [0,1] (1 = identical)."""
        if not sig_a or not sig_b:
            return 0.0

        # Weights for each component
        weights = {
            'n_samples': 0.2,
            'n_features': 0.3,
            'n_classes': 0.3,
            'class_entropy': 0.2
        }
        score = 0.0
        for key, w in weights.items():
            a = sig_a.get(key)
            b = sig_b.get(key)
            if a is None or b is None:
                continue
            if a == b:
                score += w
            else:
                diff = abs(a - b) / max(1.0, max(a, b))
                score += w * max(0.0, 1.0 - diff)

        return min(1.0, score)

    def add_pareto_front(self, pareto_front, dataset_id=None, dataset_signature=None):
        """
        Add solutions from a Pareto front to meta-knowledge.
        
        Args:
            pareto_front: List of individuals {'model': Model, 'accuracy': float, 'size': float}
            dataset_id: Optional identifier for the dataset
        """
        for ind in pareto_front:
            try:
                model = ind['model']
                model_name = model.getModelName()
                params = model.getModelParams()
                
                solution = {
                    'model_name': model_name,
                    'params': params,
                    'accuracy': float(ind['accuracy']),
                    'size': float(ind['size']),
                    'fitness': self._compute_fitness(ind['accuracy'], ind['size']),
                    'dataset_id': dataset_id
                }
                
                self.meta_knowledge['solutions'].append(solution)
                
                # Update model statistics
                stats = self.meta_knowledge['model_stats'][model_name]
                stats['total'] += 1
                stats['wins'] += 1
                stats['avg_accuracy'] = (stats['avg_accuracy'] * (stats['total'] - 1) + 
                                         float(ind['accuracy'])) / stats['total']
                
                # Store parameter patterns
                self.meta_knowledge['parameter_patterns'][model_name].append(params)
                
            except Exception as e:
                print(f"Error adding solution to meta-knowledge: {e}")
        
        # Store dataset signature if available for later similarity matching
        if dataset_id and dataset_signature:
            self.meta_knowledge['dataset_signatures'][dataset_id] = dataset_signature

        # Keep recent solutions (bounded memory)
        if len(self.meta_knowledge['solutions']) > 1000:
            self.meta_knowledge['solutions'] = self.meta_knowledge['solutions'][-1000:]
    
    def get_warm_start_population(self, pop_size, prefer_models=None, dataset_id=None, dataset_signature=None):
        """
        Generate warm-start population from meta-knowledge with dataset-aware filtering.

        Args:
            pop_size: Size of population to generate
            prefer_models: List of model names to prefer (None = all models)
            dataset_id: Identifier for current dataset
            dataset_signature: Numeric signature for current dataset

        Returns:
            List of Model instances with high-fitness parameter configurations
        """
        if not self.meta_knowledge['solutions']:
            return None

        # Filter solutions by preferred models
        solutions = self.meta_knowledge['solutions']
        if prefer_models:
            solutions = [s for s in solutions if s['model_name'] in prefer_models]
        if not solutions:
            solutions = self.meta_knowledge['solutions']

        # Score solutions by fitness + dataset similarity
        scored_solutions = []
        for s in solutions:
            dataset_score = 0.0
            if dataset_id and s.get('dataset_id') == dataset_id:
                dataset_score = 1.0
            elif dataset_signature and s.get('dataset_id') in self.meta_knowledge['dataset_signatures']:
                historical_sig = self.meta_knowledge['dataset_signatures'][s.get('dataset_id')]
                dataset_score = self._dataset_signature_similarity(dataset_signature, historical_sig)

            combined_score = 0.6 * s['fitness'] + 0.4 * dataset_score
            scored_solutions.append((combined_score, s))

        scored_solutions.sort(key=lambda x: x[0], reverse=True)

        top_solutions = [s for _, s in scored_solutions]

        # Sample top solutions with some randomness (explore-exploit balance)
        n_elite = max(1, pop_size // 3)
        elite_solutions = top_solutions[:n_elite]
        other_solutions = top_solutions[n_elite:]

        population = []

        # Add elite solutions
        for _ in range(min(len(elite_solutions), pop_size // 2)):
            sol = np.random.choice(elite_solutions)
            model = self._create_model_from_solution(sol)
            if model:
                population.append(model)

        # Add exploration samples from other solutions
        for _ in range(pop_size - len(population)):
            if np.random.random() < 0.7 and other_solutions:
                sol = np.random.choice(other_solutions)
                model = self._create_model_from_solution(sol, add_noise=True)
            else:
                model = Model()

            if model:
                population.append(model)

        return population[:pop_size]
    
    def get_best_model_type(self, dataset_id=None):
        """
        Get the best performing model type based on meta-knowledge.
        
        Args:
            dataset_id: Optional dataset identifier for dataset-specific recommendation
        
        Returns:
            Model name (string)
        """
        stats = self.meta_knowledge['model_stats']
        if not stats:
            return None
        
        # Sort by average accuracy
        best_model = max(stats.items(), key=lambda x: x[1]['avg_accuracy'])
        return best_model[0]
    
    def get_adaptive_mutation_rate(self, population_diversity):
        """
        Compute adaptive mutation rate based on population diversity.
        Higher diversity -> lower mutation rate (exploit)
        Lower diversity -> higher mutation rate (explore)
        
        Args:
            population_diversity: Diversity metric (0-1)
        
        Returns:
            Adaptive mutation rate
        """
        # Base mutation rate
        base_pm = 0.3
        
        # If diversity is high, reduce mutation (exploit good solutions)
        # If diversity is low, increase mutation (explore new regions)
        if population_diversity > 0.7:
            return base_pm * 0.5  # 0.15 - exploitation
        elif population_diversity < 0.3:
            return base_pm * 2.0  # 0.60 - exploration
        else:
            return base_pm
    
    def compute_population_diversity(self, population):
        """
        Compute diversity of population based on objective space.
        
        Args:
            population: List of individuals with 'accuracy' and 'size'
        
        Returns:
            Diversity metric (0-1)
        """
        if len(population) < 2:
            return 0.0
        
        accuracies = np.array([ind['accuracy'] for ind in population])
        sizes = np.array([ind['size'] for ind in population])
        
        # Normalize
        acc_range = accuracies.max() - accuracies.min() + 1e-6
        size_range = sizes.max() - sizes.min() + 1e-6
        
        accuracies_norm = (accuracies - accuracies.min()) / acc_range
        sizes_norm = (sizes - sizes.min()) / size_range
        
        # Compute average pairwise distance in objective space
        dist_sum = 0
        count = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.sqrt((accuracies_norm[i] - accuracies_norm[j])**2 + 
                              (sizes_norm[i] - sizes_norm[j])**2)
                dist_sum += dist
                count += 1
        
        if count > 0:
            avg_dist = dist_sum / count
            # Normalize to [0, 1] roughly
            diversity = min(1.0, avg_dist / np.sqrt(2))
        else:
            diversity = 0.0
        
        return diversity
    
    def _compute_fitness(self, accuracy, size):
        """
        Compute a unified fitness score (higher is better).
        Weighted combination of accuracy and inverse size.
        """
        # Normalize size (smaller is better)
        inverse_size = 1.0 / (1.0 + size / 1000.0)  # Bounded inverse
        # Weighted average
        fitness = 0.7 * accuracy + 0.3 * inverse_size
        return fitness
    
    def _create_model_from_solution(self, solution, add_noise=False):
        """
        Create a Model instance from a stored solution.

        Args:
            solution: Solution dict with 'model_name' and 'params'
            add_noise: If True, slightly perturb parameters for variation

        Returns:
            Model instance
        """
        try:
            params = solution.get('params', {}) or {}
            params = params.copy()

            if add_noise:
                # Add small perturbation to a few numeric parameters
                for key in list(params.keys())[:2]:
                    if isinstance(params[key], (int, float)) and not isinstance(params[key], bool):
                        params[key] = type(params[key])(max(1e-8, params[key] * np.random.uniform(0.9, 1.1)))

            if params:
                model = Model.from_solution(solution['model_name'], params)
            else:
                model = Model(model_name=solution['model_name'])

            return model
        except Exception as e:
            print(f"Error creating model from solution: {e}")
            return None
    
    def export_meta_knowledge_summary(self, output_path='meta_summary.txt'):
        """Export human-readable summary of meta-knowledge."""
        with open(output_path, 'w') as f:
            f.write("=== META-LEARNING KNOWLEDGE SUMMARY ===\n\n")
            
            f.write(f"Total solutions learned: {len(self.meta_knowledge['solutions'])}\n\n")
            
            f.write("Model Performance Ranking:\n")
            f.write("-" * 50 + "\n")
            stats = sorted(self.meta_knowledge['model_stats'].items(), 
                          key=lambda x: x[1]['avg_accuracy'], reverse=True)
            for model_name, stat in stats:
                f.write(f"{model_name}:\n")
                f.write(f"  Average Accuracy: {stat['avg_accuracy']:.4f}\n")
                f.write(f"  Times in Pareto Front: {stat['wins']}/{stat['total']}\n")
                f.write(f"  Win Rate: {stat['wins']/max(1, stat['total']):.2%}\n\n")
            
            f.write("\nTop 10 Solutions:\n")
            f.write("-" * 50 + "\n")
            sorted_solutions = sorted(self.meta_knowledge['solutions'], 
                                     key=lambda x: x['fitness'], reverse=True)
            for i, sol in enumerate(sorted_solutions[:10], 1):
                f.write(f"{i}. {sol['model_name']}\n")
                f.write(f"   Accuracy: {sol['accuracy']:.4f}, Size: {sol['size']:.0f}\n")
                f.write(f"   Fitness: {sol['fitness']:.4f}\n\n")
        
        print(f"Meta-knowledge summary exported to {output_path}")
