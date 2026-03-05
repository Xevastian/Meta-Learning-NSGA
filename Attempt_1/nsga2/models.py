from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np

class Model:
    
    def __init__(self, model_name=None):
        self.__modelName = model_name if model_name else random.choice(['MLP', 'RandomForest', 'HistGradientBoosting', 'LogisticRegression', 'SGD', 'KNeighbors'])
        self.__modelParams = {}
        self.__model = self.__builder(self.__modelName)
    
    def mutate(self, p=0.5):
        if random.random() < p:
            # Adjust current model parameters by ±20%
            self.__adjustParams()
            self.__model = self.__model.__class__(**self.__modelParams)
        else:
            # Switch to completely new model type
            self._mutateNew()

    def _mutateNew(self):
        model = random.choice(['MLP', 'RandomForest', 'HistGradientBoosting', 'LogisticRegression', 'SGD', 'KNeighbors'])
        self.__modelName = model
        self.__model = self.__builder(model)

    def __adjustParams(self):
        """Adjust existing parameters by ±20% for numerical values."""
        numerical_params = {
            'MLP': ['alpha', 'learning_rate_init'],
            'RandomForest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
            'HistGradientBoosting': ['max_iter', 'max_depth', 'learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization', 'max_bins', 'validation_fraction', 'n_iter_no_change', 'tol'],
            'LogisticRegression': ['C', 'max_iter', 'tol'],
            'SGD': ['alpha', 'l1_ratio', 'eta0', 'max_iter', 'tol', 'n_iter_no_change'],
            'KNeighbors': ['n_neighbors', 'leaf_size']
        }
        
        if self.__modelName in numerical_params:
            for param in numerical_params[self.__modelName]:
                if param in self.__modelParams and isinstance(self.__modelParams[param], (int, float)):
                    # Adjust by ±20%
                    factor = random.uniform(0.8, 1.2)
                    self.__modelParams[param] = self.__modelParams[param] * factor
                    # Clamp to reasonable bounds
                    if param in ['l1_ratio', 'validation_fraction']:
                        self.__modelParams[param] = max(0.0, min(1.0, self.__modelParams[param]))
                    elif param == 'max_depth' and self.__modelParams[param] is not None:
                        self.__modelParams[param] = max(1, int(self.__modelParams[param]))
                    elif param in ['n_estimators', 'max_iter', 'n_neighbors', 'leaf_size', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'max_bins', 'n_iter_no_change']:
                        self.__modelParams[param] = max(1, int(self.__modelParams[param]))
                    elif param in ['alpha', 'learning_rate', 'learning_rate_init', 'eta0', 'tol', 'l2_regularization', 'C']:
                        self.__modelParams[param] = max(1e-10, self.__modelParams[param])

    def __builder(self, model):
        build = {
            'MLP': self.__MLP, 
            'RandomForest': self.__RandomForest, 
            'HistGradientBoosting': self.__HistGradientBoosting, 
            'LogisticRegression': self.__LogisticRegression, 
            'SGD': self.__SGD, 
            'KNeighbors': self.__KNeighbors
        }
        return build[model]()
    
    def getModel(self):
        return self.__model
    
    def getModelName(self):
        return self.__modelName
    
    def getModelParams(self):
        return self.__modelParams
    
#                ==================== 6 MODELS INITIATOR ====================

    def __MLP(self):
        self.__MLPParameter()
        return MLPClassifier(**self.__modelParams)

    def __RandomForest(self):
        self.__RandomForestParameter()
        return RandomForestClassifier(**self.__modelParams)

    def __HistGradientBoosting(self):
        self.__HistGradientBoostingParameter()
        return HistGradientBoostingClassifier(**self.__modelParams)

    def __LogisticRegression(self):
        self.__LogisticRegressionParameter()
        return LogisticRegression(**self.__modelParams)

    def __SGD(self):
        self.__SGDParameter()
        return SGDClassifier(**self.__modelParams)

    def __KNeighbors(self):
        self.__KNeighborsParameter()
        return KNeighborsClassifier(**self.__modelParams)

    
#                ==================== 6 MODELS PARAMETER ====================
    
    def __SGDParameter(self):
        """SGDClassifier configuration"""
        params = {}
        params['loss'] = random.choice(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'])
        params['penalty'] = random.choice(['l2', 'l1', 'elasticnet', None])
        params['alpha'] = float(10 ** random.uniform(-6, -1))  # 1e-6 to 0.1
        
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = float(random.uniform(0.0, 1.0))
        else:
            params['l1_ratio'] = 0.15
        
        params['learning_rate'] = random.choice(['constant', 'optimal', 'invscaling', 'adaptive'])
        
        # `eta0` is only required for certain learning_rate schedules.
        # Do not pass `eta0=0.0` (invalid). Only set eta0 for schedules
        # that use an explicit initial learning rate.
        if params['learning_rate'] in ['constant', 'invscaling', 'adaptive']:
            params['eta0'] = float(10 ** random.uniform(-4, -1))  # 1e-4 to 0.1
        else:
            # For 'optimal' learning_rate, omit `eta0` so sklearn uses its default
            # (avoid passing zero which causes validation errors).
            pass
        
        params['max_iter'] = random.choice([100, 200, 500, 1000])
        params['tol'] = float(10 ** random.uniform(-5, -2))  # 1e-5 to 1e-2
        params['early_stopping'] = random.choice([True, False])
        
        if params['early_stopping']:
            params['validation_fraction'] = 0.1
            params['n_iter_no_change'] = random.choice([5, 10, 20])
        
        params['random_state'] = 42
        self.__modelParams = params
    
    def __MLPParameter(self):
        """MLPClassifier configuration"""
        params = {}
        
        # Hidden layer sizes (2 layers)
        f1 = random.randint(32, 128)
        f2 = int(f1 / 2)
        params['hidden_layer_sizes'] = (f1, f2)
        
        params['activation'] = random.choice(['relu', 'tanh', 'logistic', 'identity'])
        params['solver'] = random.choice(['adam', 'sgd', 'lbfgs'])
        params['alpha'] = float(np.random.uniform(1e-6, 1e-2))  # L2 regularization
        params['learning_rate'] = random.choice(['constant', 'invscaling', 'adaptive'])
        params['learning_rate_init'] = float(np.random.uniform(1e-4, 1e-2))
        params['batch_size'] = random.choice(['auto', 16, 32, 64, 128])
        params['max_iter'] = random.choice([100, 200, 300])
        params['early_stopping'] = True
        params['validation_fraction'] = 0.1
        params['n_iter_no_change'] = 10
        params['random_state'] = 42
        
        self.__modelParams = params
    
    def __RandomForestParameter(self):
        """RandomForestClassifier configuration"""
        params = {}
        
        params['n_estimators'] = random.choice([10, 25, 50, 100, 150, 200])
        params['max_depth'] = random.choice([None, 5, 10, 15, 20, 30, 40])
        params['min_samples_split'] = random.choice([2, 5, 10, 15, 20])
        params['min_samples_leaf'] = random.choice([1, 2, 4, 8, 10])
        params['max_features'] = random.choice(['sqrt', 'log2', None])
        params['bootstrap'] = random.choice([True, False])
        
        if params['bootstrap']:
            params['oob_score'] = random.choice([True, False])
        else:
            params['oob_score'] = False
        
        params['max_samples'] = random.choice([None, 0.5, 0.7, 0.9]) if params['bootstrap'] else None
        params['criterion'] = random.choice(['gini', 'entropy', 'log_loss'])
        params['random_state'] = 42
        params['n_jobs'] = -1  # Use all cores
        
        self.__modelParams = params
    
    def __HistGradientBoostingParameter(self):
        """HistGradientBoostingClassifier configuration"""
        params = {}
        
        params['max_iter'] = random.choice(range(50,301))
        params['max_depth'] = random.choice([None, 3, 5, 7, 10, 15])
        params['learning_rate'] = float(10 ** random.uniform(-3, 0))  # 0.001 to 1.0
        params['max_leaf_nodes'] = random.choice([None, 15, 31, 63, 127])
        params['min_samples_leaf'] = random.choice([1, 5, 10, 20, 50])
        params['l2_regularization'] = float(10 ** random.uniform(-6, 1))  # 1e-6 to 10
        params['max_bins'] = random.choice([63, 127, 255])
        params['early_stopping'] = random.choice([True, False])
        
        if params['early_stopping']:
            params['validation_fraction'] = random.choice([0.1, 0.15, 0.2])
            params['n_iter_no_change'] = random.choice([5, 10, 15, 20])
            params['tol'] = float(10 ** random.uniform(-5, -2))
        
        params['random_state'] = 42
        
        self.__modelParams = params
    
    def __LogisticRegressionParameter(self):
        """LogisticRegression configuration without using deprecated 'penalty' key."""
        params = {}
        reg = random.choice(['l1', 'l2', 'elasticnet', 'none'])

        if reg == 'elasticnet':
            params['l1_ratio'] = float(random.uniform(0.0, 1.0))
            params['solver'] = 'saga'  # saga supports elasticnet and l1_ratio
        elif reg == 'l1':
            params['l1_ratio'] = 1.0
            # prefer 'saga' which supports multiclass; avoid 'liblinear'
            params['solver'] = 'saga'
        elif reg == 'l2':
            params['l1_ratio'] = 0.0
            params['solver'] = random.choice(['lbfgs', 'newton-cg', 'sag', 'saga'])
        else:  # 'none' -> mimic no regularization by setting C to infinity
            params['l1_ratio'] = 0.0
            params['C'] = float('inf')
            params['solver'] = random.choice(['lbfgs', 'newton-cg', 'sag', 'saga'])

        if 'C' not in params:
            params['C'] = float(10 ** random.uniform(-3, 3))  # 0.001 to 1000

        params['max_iter'] = random.choice([100, 200, 500, 1000])
        params['tol'] = float(10 ** random.uniform(-5, -2))
        params['random_state'] = 42

        # Do NOT set 'penalty' key to avoid FutureWarning / UserWarning
        self.__modelParams = params
    
    def __KNeighborsParameter(self):
        """KNeighborsClassifier configuration"""
        params = {}
        
        params['n_neighbors'] = random.choice([3, 5, 7, 9, 11, 15, 21, 31])
        params['weights'] = random.choice(['uniform', 'distance'])
        params['algorithm'] = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])
        params['leaf_size'] = random.choice([10, 20, 30, 40, 50])
        params['p'] = random.choice([1, 2])  # 1=manhattan, 2=euclidean
        params['metric'] = random.choice(['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
        params['n_jobs'] = -1
        
        self.__modelParams = params