import json
import os
from .nsga2 import nsga2, nondominated_sort
from .trainer import Trainer
import numpy as np

class MetaLearningNSGA2:
    def __init__(self, data_path, priority_accuracy=0.5, priority_size=0.5, pop_size=20, generations=10, seed=None, update_meta_db=True):
        """
        Initialize MetaLearningNSGA2.

        Args:
            data_path: Path to the CSV dataset
            priority_accuracy: Weight for accuracy objective (0-1)
            priority_size: Weight for size objective (0-1)
            pop_size: Population size for NSGA-II
            generations: Number of generations
            seed: Random seed (None for random)
        """
        self.data_path = data_path
        self.priority_acc = priority_accuracy
        self.priority_size = priority_size
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.update_meta_db = update_meta_db
        self.pareto_front = None

    def _split_random_state(self):
        """Match nsga2.py: base_seed = int(seed) % (2**31 - 1), default seed 67."""
        s = self.seed if self.seed is not None else 67
        return int(s) % (2**31 - 1)

    def run(self):
        """
        Run NSGA-II and return the Pareto front.

        Returns:
            List of Pareto optimal models with their metrics
        """
        # Run NSGA-II
        pop = nsga2(
            pop_size=self.pop_size,
            generations=self.generations,
            data_path=self.data_path,
            seed=self.seed,
            update_meta_db=self.update_meta_db,
            save_plot=False,
            show_plot=False
        )

        # Get final Pareto front
        fronts = nondominated_sort(pop)
        self.pareto_front = fronts[0] if fronts else []

        # Save to JSON
        pareto_data = []
        for ind in self.pareto_front:
            model_data = {
                'name': ind['model'].getModelName(),
                'params': ind['model'].getModelParams(),
                'accuracy': ind['accuracy'],
                'size': ind['size']
            }
            pareto_data.append(model_data)

        with open('pareto_front.json', 'w') as f:
            json.dump(pareto_data, f, indent=2)

        # Build ready-to-use trained models from Pareto solutions
        self.ready_to_use_models = []
        for ind in self.pareto_front:
            if 'model' not in ind:
                continue
            base_model = ind['model']
            rs = self._split_random_state()
            trainer = Trainer(base_model, self.data_path, random_state=rs)
            trained_clf = trainer.getModel()
            self.ready_to_use_models.append({
                'pareto_model': base_model,
                'trained_model': trained_clf,
                'scaler': trainer.scaler,
                'impute_medians': trainer.get_impute_medians(),
                'accuracy': ind.get('accuracy', None),
                'size': ind.get('size', None)
            })

        # If ready_to_use_models is empty, keep fallback to raw pareto front items
        if not self.ready_to_use_models:
            self.ready_to_use_models = self.pareto_front

        print(f"Pareto front saved to pareto_front.json with {len(self.pareto_front)} models")
        return self.ready_to_use_models

    def get_confusion_matrix(self, model_index):
        """
        Get confusion matrix for a specific model from the Pareto front.

        Args:
            model_index: Index of the model in the Pareto front

        Returns:
            Confusion matrix as numpy array
        """
        if self.pareto_front is None:
            raise ValueError("Run the algorithm first using run()")
        if model_index < 0 or model_index >= len(self.pareto_front):
            raise IndexError("Model index out of range")

        model = self.pareto_front[model_index]['model']
        trainer = Trainer(model, self.data_path, random_state=self._split_random_state())
        return trainer.getConfusionMatrix()

    def get_model_config(self, model_index):
        """
        Get the configuration code for a specific model from the Pareto front.

        Args:
            model_index: Index of the model in the Pareto front

        Returns:
            String containing the Python code to recreate the model
        """
        if self.pareto_front is None:
            raise ValueError("Run the algorithm first using run()")
        if model_index < 0 or model_index >= len(self.pareto_front):
            raise IndexError("Model index out of range")

        model = self.pareto_front[model_index]['model']
        name = model.getModelName()
        params = model.getModelParams()

        # Generate import and instantiation code
        import_line = f"from sklearn.{name.lower()} import {name}"
        params_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
        model_line = f"model = {name}({params_str})"

        code = f"{import_line}\n{model_line}"
        return code

    def list_pareto_models(self):
        """
        List all models in the Pareto front with their metrics.
        """
        if self.pareto_front is None:
            print("No Pareto front available. Run the algorithm first.")
            return

        print("Pareto Front Models:")
        for i, ind in enumerate(self.pareto_front):
            name = ind['model'].getModelName()
            acc = ind['accuracy']
            size = ind['size']
            print(f"{i}: {name} - Acc={acc:.4f}, Size={size}")

    def get_ready_model(self, model_index):
        """Return trained ready-to-use model dictionary from Pareto index."""
        if not hasattr(self, 'ready_to_use_models') or self.ready_to_use_models is None:
            raise ValueError("Run the algorithm first using run()")
        if model_index < 0 or model_index >= len(self.ready_to_use_models):
            raise IndexError("Model index out of range")
        return self.ready_to_use_models[model_index]

    def predict_pareto(self, model_index, x):
        """Predict for new input vector x using Pareto model at model_index."""
        entry = self.get_ready_model(model_index)
        clf = entry['trained_model']
        scaler = entry['scaler']
        med = entry.get('impute_medians')

        x_arr = np.array(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        # Match Trainer: train-fitted imputation + clip, then scaler (same as training pipeline)
        if med is not None:
            x_arr = np.asarray(x_arr, dtype=np.float64)
            x_arr[~np.isfinite(x_arr)] = np.nan
            if np.isnan(x_arr).any():
                nr, nc = np.where(np.isnan(x_arr))
                x_arr[nr, nc] = med[nc]
            x_arr = np.clip(x_arr, -1e12, 1e12)
        if scaler is not None:
            x_arr = scaler.transform(x_arr)
        return clf.predict(x_arr)