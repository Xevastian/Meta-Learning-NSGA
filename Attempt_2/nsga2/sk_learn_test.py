import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Edit these values or pass CSV path as first CLI arg ---
DATA_PATH = r"C:\Users\suman\OneDrive\Documents\4th year\Research\Notebooks\#2 NSGA-II\Attempt_2\Spam.csv"
TARGET_COLUMN = 'label'
TEST_SIZE = 0.2
RANDOM_STATE = 42
SCALE = True

# Paste model name and params here:
MODEL_NAME = "HistGradientBoosting"
MODEL_PARAMS = {'max_iter': 300, 'max_depth': 5, 'learning_rate': 0.03549799328527889, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'l2_regularization': 0.027440740954483653, 'max_bins': 63, 'early_stopping': True, 'validation_fraction': 0.15, 'n_iter_no_change': 10, 'tol': 0.00286524552290574, 'random_state': 42}

MODEL_MAP = {
    'MLP': MLPClassifier,
    'RandomForest': RandomForestClassifier,
    'HistGradientBoosting': HistGradientBoostingClassifier,
    'LogisticRegression': LogisticRegression,
    'SGD': SGDClassifier,
    'KNeighbors': KNeighborsClassifier
}


def build_model(name: str, params: dict):
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model name: {name}. Supported: {list(MODEL_MAP.keys())}")
    cls = MODEL_MAP[name]
    # Remove None values to avoid sklearn warnings
    params_clean = {k: v for k, v in params.items() if v is not None}
    return cls(**params_clean)


def test_model(data_path: str, model_name: str, params: dict,
               target_column: str = TARGET_COLUMN,
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE,
               scale: bool = SCALE,
               sample_rows: int = 20):
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None
    )

    # keep a copy of X_test with original columns for printing samples
    if isinstance(X_test, pd.DataFrame):
        X_test_df = X_test.reset_index(drop=True).copy()
    else:
        X_test_df = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = build_model(model_name, params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nModel:", model_name)
    print("Parameters:")
    print(params)
    print(f"\nAccuracy: {acc:.6f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Print sample rows: compact with leading cols, '...' and trailing cols plus actual/predicted
    try:
        n = min(len(X_test_df), sample_rows)
        sample_df = X_test_df.reset_index(drop=True).head(n).copy()
        actual_series = pd.Series(y_test).reset_index(drop=True).iloc[:n]
        pred_series = pd.Series(y_pred).reset_index(drop=True).iloc[:n]
        sample_df['actual'] = actual_series.values
        sample_df['predicted'] = pred_series.values

        # compact column display: show first/last cols with '...' in middle
        def compact_display(df, left=3, right=3):
            cols = list(df.columns)
            # if few columns, return as-is
            if len(cols) <= left + right + 2:  # +2 for actual/predicted
                return df
            # ensure actual/predicted included at end
            end_cols = ['actual', 'predicted']
            mid_cols = [c for c in cols if c not in end_cols]
            if len(mid_cols) <= left + right:
                return df[end_cols + mid_cols] if False else df  # fallback
            left_cols = mid_cols[:left]
            right_cols = mid_cols[-right:]
            ellipsis_col = ['...'] * len(df)
            compact = pd.concat(
                [df[left_cols].reset_index(drop=True),
                 pd.DataFrame({'...': ellipsis_col}),
                 df[right_cols].reset_index(drop=True),
                 df[end_cols].reset_index(drop=True)],
                axis=1
            )
            return compact

        compact_df = compact_display(sample_df, left=3, right=3)
        # round numeric for compactness
        compact_df = compact_df.round(4)

        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(f"\nSample predictions (first {n} rows):")
            print(compact_df.to_string(index=False))
    except Exception as e:
        print(f"\nCould not print sample predictions: {e}")

    return model, acc, y_test, y_pred


if __name__ == "__main__":
    # Allow passing data path as first arg
    if len(sys.argv) >= 2:
        DATA_PATH = sys.argv[1]
    if DATA_PATH is None:
        print("Edit DATA_PATH at top of file or pass CSV path as first argument.")
        sys.exit(1)

    test_model(DATA_PATH, MODEL_NAME, MODEL_PARAMS)