import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Trainer:
    def __init__(self, model, data, target_column='label', test_size=0.2, random_state=42, scale_data=True):
        
        """
        Train and evaluate a model
        
        Args:
            model: Model instance (from Model class)
            data: pandas DataFrame or path to CSV file
            target_column: name of the target column (default: 'label')
            test_size: proportion of test set (default: 0.2)
            random_state: random seed (default: 42)
            scale_data: whether to standardize features (default: True)
        """
        # Store model
        self.model_instance = model
        self.model_name = model.getModelName()
        self.model_params = model.getModelParams()
        
        # Load data if it's a file path
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data
        
        # Prepare data
        self.target_column = target_column
        self.X = self.df.drop(target_column, axis=1).values
        self.y = self.df[target_column].values
        
        # Scale data if requested
        if scale_data:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = None
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.trained_model = self.__train()
        
        # Calculate metrics
        self.result = self.__calculateResult()
        self.accuracy = self.__calculateAccuracy()
        self.size = self.__calculateSize()
    
    def __train(self):
        """Train the model"""
        clf = self.model_instance.getModel()
        clf.fit(self.X_train, self.y_train)
        return clf
    
    def __calculateResult(self):
        """Get predictions on test set"""
        y_pred = self.trained_model.predict(self.X_test)
        return y_pred
    
    def __calculateAccuracy(self):
        """Calculate accuracy on test set"""
        return accuracy_score(self.y_test, self.result)
    
    def __calculateSize(self):
        """Calculate model size based on model type"""
        model_type = self.model_name
        model = self.trained_model
        
        if model_type == 'MLP':
            # Total parameters (weights + biases)
            size = sum(w.size + b.size for w, b in zip(model.coefs_, model.intercepts_))
        
        elif model_type == 'RandomForest':
            # Total nodes across all trees
            size = sum(tree.tree_.node_count for tree in model.estimators_)
        
        elif model_type == 'HistGradientBoosting':
            # Total nodes across all trees
            try:
                size = sum(predictor[0].get_n_leaf_nodes() for predictor in model._predictors)
            except:
                # Fallback: estimate based on iterations
                size = model.n_iter_ * 100  # Rough estimate
        
        elif model_type in ['LogisticRegression', 'SGD']:
            # Coefficients + intercept
            size = model.coef_.size
            if hasattr(model, 'intercept_') and model.intercept_ is not None:
                size += model.intercept_.size
        
        elif model_type == 'KNeighbors':
            # Stores training data (k * n_features)
            size = model.n_samples_fit_ * model.n_features_in_
        
        else:
            # Default: return 0 if unknown
            size = None
        
        return int(size)
    
    def getAccuracy(self):
        return self.accuracy
    
    def getSize(self):
        return self.size
    
    def getModel(self):
        return self.trained_model
    
    def getModelName(self):
        return self.model_name
    
    def getPredictions(self):
        return self.result
    
    def getSummary(self):
        return {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'accuracy': self.accuracy,
            'model_size': self.size,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
