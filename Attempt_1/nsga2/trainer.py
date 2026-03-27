import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Trainer:
    def __init__(self, model, data, target_column='label', test_size=0.2, random_state=42, scale_data=True, stratify=True):
        try:
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

            # Split data deterministically; stratify if requested and feasible
            stratify_vals = self.y if stratify and len(np.unique(self.y)) > 1 else None
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_vals
            )

            # Sanitize features: fit medians on TRAIN only, then apply to train and test (no leakage)
            self._impute_medians = None
            self.X_train = self.__fit_sanitize(self.X_train)
            self.X_test = self.__transform_sanitize(self.X_test)

            # Scale data if requested
            if scale_data:
                self.scaler = StandardScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
            else:
                self.scaler = None

            # Train the model
            self.trained_model = self.__train()
            
            # Calculate metrics
            self.result = self.__calculateResult()
            self.accuracy = self.__calculateAccuracy()
            self.size = self.__calculateSize()
            self.confusion_matrix = self.__calculateConfusionMatrix()
        except Exception as e:
            print(f"Trainer initialization failed: {e}")
            # Set defaults safely
            self.model_instance = model
            self.model_name = 'Unknown'
            try:
                self.model_name = model.getModelName()
            except Exception:
                pass
            self.model_params = {}
            try:
                self.model_params = model.getModelParams()
            except Exception:
                pass
            self.trained_model = None
            self.result = None
            self.accuracy = 0.0
            self.size = float('inf')
            self.confusion_matrix = None

    _CLIP_LOW = -1e12
    _CLIP_HIGH = 1e12

    def __fit_sanitize(self, X_train):
        """
        Learn per-column medians from training data only; impute and clip train.
        """
        X = np.asarray(X_train, dtype=np.float64)
        X[~np.isfinite(X)] = np.nan
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        self._impute_medians = col_medians
        if np.isnan(X).any():
            nan_rows, nan_cols = np.where(np.isnan(X))
            X[nan_rows, nan_cols] = self._impute_medians[nan_cols]
        return np.clip(X, self._CLIP_LOW, self._CLIP_HIGH)

    def __transform_sanitize(self, X):
        """
        Apply training medians to test (and any future data): no test statistics used.
        """
        X = np.asarray(X, dtype=np.float64)
        X[~np.isfinite(X)] = np.nan
        if self._impute_medians is None:
            raise RuntimeError("__transform_sanitize called before __fit_sanitize")
        if X.shape[1] != self._impute_medians.shape[0]:
            raise ValueError(
                f"Feature count mismatch: got {X.shape[1]}, expected {self._impute_medians.shape[0]}"
            )
        if np.isnan(X).any():
            nan_rows, nan_cols = np.where(np.isnan(X))
            X[nan_rows, nan_cols] = self._impute_medians[nan_cols]
        return np.clip(X, self._CLIP_LOW, self._CLIP_HIGH)

    def apply_feature_preprocessing(self, X):
        """
        Same pipeline as training inputs: train-fitted imputation + clip + scaler (if any).
        Use for prediction on new rows so it matches fit().
        """
        X = self.__transform_sanitize(np.asarray(X, dtype=np.float64))
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def get_impute_medians(self):
        """Expose train-fitted medians for callers that need manual preprocessing."""
        return None if self._impute_medians is None else self._impute_medians.copy()
    
    def __train(self):
        """Train the model"""
        clf = self.model_instance.getModel()
        clf.fit(self.X_train, self.y_train)
        return clf
    
    def __calculateResult(self):
        """Get predictions on test set"""
        if self.trained_model is None:
            return None
        y_pred = self.trained_model.predict(self.X_test)
        return y_pred
    
    def __calculateAccuracy(self):
        """Calculate accuracy on test set"""
        if self.result is None:
            return 0.0
        return accuracy_score(self.y_test, self.result)
    
    def __calculateSize(self):
        """Calculate model size as serialized byte length"""
        if self.trained_model is None:
            return float('inf')
        try:
            import pickle
            return len(pickle.dumps(self.trained_model))
        except Exception:
            return float('inf')
        
    def __calculateConfusionMatrix(self):
        """Calculate confusion matrix on test set"""
        if self.result is None:
            return None
        return confusion_matrix(self.y_test, self.result)
    
    def getAccuracy(self):
        return self.accuracy if self.accuracy is not None else 0.0
    
    def getSize(self):
        return self.size if self.size is not None else float('inf')
    
    def getModel(self):
        return self.trained_model
    
    def getModelName(self):
        return self.model_name
    
    def getPredictions(self):
        return self.result
    
    def getConfusionMatrix(self):
        return self.confusion_matrix
    
    def getSummary(self):
        return {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'accuracy': self.accuracy,
            'model_size': self.size,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
