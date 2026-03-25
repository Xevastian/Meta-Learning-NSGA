from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np
import abc

# Defined choices for parameter randomization to make decisions more formulated
MLP_ACTIVATIONS = ['relu', 'tanh', 'logistic', 'identity']
MLP_SOLVERS = ['adam', 'sgd', 'lbfgs']
MLP_LEARNING_RATES = ['constant', 'invscaling', 'adaptive']

RF_CRITERIA = ['gini', 'entropy', 'log_loss']
RF_MAX_FEATURES = ['sqrt', 'log2', None]

HGB_MAX_DEPTHS = [None] + list(range(3, 20))
HGB_MAX_LEAF_NODES = [None] + list(range(10, 200, 10))

LR_REG_TYPES = ['l1', 'l2', 'elasticnet']
LR_SOLVERS = ['lbfgs', 'newton-cg', 'sag', 'saga']

SGD_LOSSES = ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']
SGD_PENALTIES = ['l2', 'l1', 'elasticnet']
SGD_LEARNING_RATES = ['constant', 'optimal', 'invscaling', 'adaptive']

KNN_WEIGHTS = ['uniform', 'distance']
KNN_ALGORITHMS = ['auto', 'ball_tree', 'kd_tree', 'brute']
KNN_METRICS = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']

class ModelType(abc.ABC):
    """Abstract base class for model types."""
    
    def __init__(self, seed=None):
        self.seed = seed
    
    @abc.abstractmethod
    def build(self):
        """Build and return the model instance."""
        pass
    
    @abc.abstractmethod
    def get_params(self, seed=None):
        """Return randomized parameters for the model."""
        pass

class MLPModel(ModelType):
    def build(self):
        return MLPClassifier(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        f1 = random.randint(32, 128)
        f2 = int(f1 / 2)
        params['hidden_layer_sizes'] = (f1, f2)
        params['activation'] = random.choice(MLP_ACTIVATIONS)
        params['solver'] = random.choice(MLP_SOLVERS)
        params['alpha'] = float(10 ** random.uniform(-6, -2))
        params['learning_rate'] = random.choice(MLP_LEARNING_RATES)
        params['learning_rate_init'] = float(10 ** random.uniform(-4, -1))
        params['batch_size'] = random.choice(['auto'] + list(range(8, 256, 8)))
        params['max_iter'] = random.randint(50, 1000)
        params['early_stopping'] = True
        params['validation_fraction'] = 0.1
        params['n_iter_no_change'] = 10
        params['random_state'] = seed
        
        # Validation
        params['hidden_layer_sizes'] = tuple(max(1, int(x)) for x in params['hidden_layer_sizes'])
        params['alpha'] = max(1e-6, min(1e-1, params['alpha']))
        params['learning_rate_init'] = max(1e-5, min(1.0, params['learning_rate_init']))
        params['max_iter'] = max(10, min(2000, params['max_iter']))
        
        return params

class RandomForestModel(ModelType):
    def build(self):
        return RandomForestClassifier(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        params['n_estimators'] = random.randint(10, 500)
        params['max_depth'] = random.choice([None] + list(range(3, 50)))
        params['min_samples_split'] = random.randint(2, 20)
        params['min_samples_leaf'] = random.randint(1, 20)
        params['max_features'] = random.choice(RF_MAX_FEATURES + [random.uniform(0.1, 1.0)])
        params['bootstrap'] = random.choice([True, False])
        params['oob_score'] = random.choice([True, False]) if params['bootstrap'] else False
        params['max_samples'] = random.uniform(0.3, 1.0) if params['bootstrap'] else None
        params['criterion'] = random.choice(RF_CRITERIA)
        params['random_state'] = seed
        # Determinism: avoid nondeterministic parallel execution.
        params['n_jobs'] = 1
        
        # Validation
        params['n_estimators'] = max(1, params['n_estimators'])
        if params['max_depth'] is not None:
            params['max_depth'] = max(1, params['max_depth'])
        params['min_samples_split'] = max(2, params['min_samples_split'])
        params['min_samples_leaf'] = max(1, params['min_samples_leaf'])
        if isinstance(params['max_features'], float):
            params['max_features'] = max(0.1, min(1.0, params['max_features']))
        if params['max_samples'] is not None:
            params['max_samples'] = max(0.1, min(1.0, params['max_samples']))
        
        return params

class HistGradientBoostingModel(ModelType):
    def build(self):
        return HistGradientBoostingClassifier(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        params['max_iter'] = random.randint(50, 1000)
        params['max_depth'] = random.choice(HGB_MAX_DEPTHS)
        params['learning_rate'] = float(10 ** random.uniform(-4, 0))
        params['max_leaf_nodes'] = random.choice(HGB_MAX_LEAF_NODES)
        params['min_samples_leaf'] = random.randint(1, 100)
        params['l2_regularization'] = float(10 ** random.uniform(-8, 2))
        params['max_bins'] = random.choice([63, 127, 255, random.randint(32, 255)])
        params['early_stopping'] = random.choice([True, False])
        if params['early_stopping']:
            params['validation_fraction'] = random.uniform(0.05, 0.3)
            params['n_iter_no_change'] = random.randint(5, 50)
            params['tol'] = float(10 ** random.uniform(-6, -2))
        params['random_state'] = None
        
        # Validation
        params['max_iter'] = max(10, params['max_iter'])
        params['learning_rate'] = max(1e-5, min(1.0, params['learning_rate']))
        params['min_samples_leaf'] = max(1, params['min_samples_leaf'])
        params['l2_regularization'] = max(0.0, params['l2_regularization'])
        params['max_bins'] = max(2, min(255, params['max_bins']))
        if params['early_stopping']:
            params['validation_fraction'] = max(0.01, min(0.5, params['validation_fraction']))
            params['n_iter_no_change'] = max(1, params['n_iter_no_change'])
            params['tol'] = max(1e-6, params['tol'])
        params['random_state'] = seed
        
        return params

class LogisticRegressionModel(ModelType):
    def build(self):
        return LogisticRegression(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        reg = random.choice(LR_REG_TYPES)
        # Map reg to sklearn penalty and solver combinations
        if reg == 'elasticnet':
            params['penalty'] = 'elasticnet'
            params['l1_ratio'] = random.uniform(0.0, 1.0)
            params['solver'] = 'saga'
        elif reg == 'l1':
            params['penalty'] = 'l1'
            params['l1_ratio'] = 1.0
            params['solver'] = 'saga'
        else:  # l2
            params['penalty'] = 'l2'
            params['l1_ratio'] = 0.0
            params['solver'] = random.choice(['newton-cg', 'lbfgs', 'sag', 'saga'])

        params['C'] = float(10 ** random.uniform(-4, 4))

        params['max_iter'] = random.randint(200, 1500)
        params['tol'] = float(10 ** random.uniform(-6, -2))
        params['random_state'] = None
        
        # Validation
        if 'C' in params and params['C'] != float('inf'):
            params['C'] = max(1e-4, params['C'])
        params['max_iter'] = max(10, params['max_iter'])
        params['tol'] = max(1e-6, params['tol'])
        if 'l1_ratio' in params:
            params['l1_ratio'] = max(0.0, min(1.0, params['l1_ratio']))
        params['random_state'] = seed
        # Determinism: avoid internal parallelism.
        params['n_jobs'] = 1
        
        return params

class SGDModel(ModelType):
    def build(self):
        return SGDClassifier(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        params['loss'] = random.choice(SGD_LOSSES)
        params['penalty'] = random.choice(SGD_PENALTIES)
        params['alpha'] = float(10 ** random.uniform(-8, -1))
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = random.uniform(0.0, 1.0)
        else:
            # Elastic net ratio is ignored unless elasticnet penalty
            params['l1_ratio'] = 0.15
        params['learning_rate'] = random.choice(SGD_LEARNING_RATES)
        if params['learning_rate'] in ['constant', 'invscaling', 'adaptive']:
            params['eta0'] = float(10 ** random.uniform(-5, -1))
        params['max_iter'] = random.randint(400, 3000)
        params['tol'] = max(1e-5, float(10 ** random.uniform(-6, -2)))
        params['early_stopping'] = random.choice([True, False])
        if params['early_stopping']:
            params['validation_fraction'] = random.uniform(0.05, 0.2)
            params['n_iter_no_change'] = random.randint(5, 50)
        params['random_state'] = None
        
        # Validation
        params['alpha'] = max(1e-8, min(1e-1, params['alpha']))
        if 'l1_ratio' in params:
            params['l1_ratio'] = max(0.0, min(1.0, params['l1_ratio']))
        if 'eta0' in params:
            params['eta0'] = max(1e-6, params['eta0'])
        params['max_iter'] = max(10, params['max_iter'])
        params['tol'] = max(1e-6, params['tol'])
        if params['early_stopping']:
            params['n_iter_no_change'] = max(1, params['n_iter_no_change'])
        params['random_state'] = seed
        
        return params

class KNeighborsModel(ModelType):
    def build(self):
        return KNeighborsClassifier(**self.get_params(seed=self.seed))
    
    def get_params(self, seed=None):
        params = {}
        params['n_neighbors'] = random.randint(1, 50)
        params['weights'] = random.choice(KNN_WEIGHTS)
        params['algorithm'] = random.choice(KNN_ALGORITHMS)
        params['leaf_size'] = random.randint(5, 100)
        params['p'] = random.choice([1, 2, random.uniform(1, 10)])
        params['metric'] = random.choice(KNN_METRICS)
        # Determinism: avoid nondeterministic parallel execution.
        params['n_jobs'] = 1
        
        # Validation
        params['n_neighbors'] = max(1, params['n_neighbors'])
        params['leaf_size'] = max(1, params['leaf_size'])
        if isinstance(params['p'], float):
            params['p'] = max(1.0, params['p'])
        
        return params

class Model:
    
    def __init__(self, model_name=None, seed=None, params=None):
        self.__seed = seed
        self.registry = {
            'MLP': MLPModel,
            'RandomForest': RandomForestModel,
            'HistGradientBoosting': HistGradientBoostingModel,
            'LogisticRegression': LogisticRegressionModel,
            'SGD': SGDModel,
            'KNeighbors': KNeighborsModel
        }
        self.__modelName = model_name if model_name and model_name in self.registry else random.choice(list(self.registry.keys()))
        self.__modelParams = params.copy() if isinstance(params, dict) else {}
        self.__model = self.__builder(self.__modelName, self.__modelParams)
    
    @staticmethod
    def _build_sklearn_model(model_name, params):
        """Construct a sklearn model instance from params"""
        if model_name == 'MLP':
            return MLPClassifier(**params)
        elif model_name == 'RandomForest':
            # Defensive: sklearn expects max_features within acceptable ranges,
            # but we only sample valid values here. No-op.
            if 'n_estimators' in params:
                params['n_estimators'] = max(1, int(params['n_estimators']))
            if 'max_depth' in params and params['max_depth'] is not None:
                params['max_depth'] = max(1, int(params['max_depth']))
            if 'min_samples_split' in params:
                params['min_samples_split'] = max(2, int(params['min_samples_split']))
            if 'min_samples_leaf' in params:
                params['min_samples_leaf'] = max(1, int(params['min_samples_leaf']))
            if isinstance(params.get('max_features'), float):
                params['max_features'] = max(0.1, min(1.0, float(params['max_features'])))
            if isinstance(params.get('max_samples'), float):
                params['max_samples'] = max(0.1, min(1.0, float(params['max_samples'])))
            return RandomForestClassifier(**params)
        elif model_name == 'HistGradientBoosting':
            if 'max_iter' in params:
                params['max_iter'] = max(10, int(params['max_iter']))
            if 'max_depth' in params and params['max_depth'] is not None:
                params['max_depth'] = max(1, int(params['max_depth']))
            if 'max_leaf_nodes' in params and params['max_leaf_nodes'] is not None:
                params['max_leaf_nodes'] = max(2, int(params['max_leaf_nodes']))
            if 'min_samples_leaf' in params:
                params['min_samples_leaf'] = max(1, int(params['min_samples_leaf']))
            if 'l2_regularization' in params:
                params['l2_regularization'] = max(0.0, float(params['l2_regularization']))
            if 'max_bins' in params:
                params['max_bins'] = max(2, min(255, int(params['max_bins'])))
            if 'learning_rate' in params:
                params['learning_rate'] = max(1e-5, min(1.0, float(params['learning_rate'])))
            if params.get('early_stopping', False):
                if 'validation_fraction' in params:
                    params['validation_fraction'] = max(0.01, min(0.5, float(params['validation_fraction'])))
                if 'n_iter_no_change' in params:
                    params['n_iter_no_change'] = max(1, int(params['n_iter_no_change']))
                if 'tol' in params:
                    params['tol'] = max(1e-6, float(params['tol']))
            return HistGradientBoostingClassifier(**params)
        elif model_name == 'LogisticRegression':
            # Meta-learning may add noise to stored params, which can violate
            # sklearn constraints (e.g., l1_ratio must be in [0, 1]).
            if 'l1_ratio' in params and params['l1_ratio'] is not None:
                try:
                    params['l1_ratio'] = float(params['l1_ratio'])
                except Exception:
                    params['l1_ratio'] = None
            if 'l1_ratio' in params and params['l1_ratio'] is not None:
                params['l1_ratio'] = max(0.0, min(1.0, params['l1_ratio']))

            # Keep penalty/l1_ratio consistent enough to avoid sklearn checks.
            penalty = params.get('penalty')
            if penalty == 'l1':
                params['l1_ratio'] = 1.0
            elif penalty == 'l2':
                params['l1_ratio'] = 0.0

            return LogisticRegression(**params)
        elif model_name == 'SGD':
            return SGDClassifier(**params)
        elif model_name == 'KNeighbors':
            if 'n_neighbors' in params:
                params['n_neighbors'] = max(1, int(params['n_neighbors']))
            if 'leaf_size' in params:
                params['leaf_size'] = max(1, int(params['leaf_size']))
            if 'p' in params and params['p'] is not None:
                params['p'] = max(1.0, float(params['p']))
            return KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def setModelParams(self, params):
        self.__modelParams = params.copy() if isinstance(params, dict) else {}
        self.__model = self._build_sklearn_model(self.__modelName, self.__modelParams)

    @classmethod
    def from_solution(cls, model_name, params, seed=None):
        model = cls(model_name=model_name, seed=seed, params=params)
        return model

    def mutate(self, p=0.5):
        """
        Mutation operator.

        Args:
            p: Probability of applying *any* mutation to this individual.
        """
        if random.random() >= p:
            # No mutation: keep individual unchanged.
            return

        # If we mutate, choose between "parameter mutation" vs "type switch".
        if random.random() < 0.5:
            # Mutate current model parameters
            self.__model = self.__builder(self.__modelName)
        else:
            # Switch to completely new model type
            self._mutateNew()

    def _mutateNew(self):
        model = random.choice(list(self.registry.keys()))
        self.__modelName = model
        # new model with random params
        self.__modelParams = {}
        self.__model = self.__builder(model)

    def __builder(self, model, params=None):
        if params:
            # Use provided params for warm-start or restoration.
            self.__modelParams = params.copy()
            return self._build_sklearn_model(model, self.__modelParams)
        # random generation path
        model_class = self.registry[model](seed=self.__seed)
        self.__modelParams = model_class.get_params(seed=self.__seed)
        return model_class.build()
    
    def getModel(self):
        return self.__model
    
    def getModelName(self):
        return self.__modelName
    
    def getModelParams(self):
        return self.__modelParams