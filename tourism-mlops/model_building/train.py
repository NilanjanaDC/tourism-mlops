"""
Model Training Script with Hyperparameter Tuning
Author: MLOps Team
Description: Trains RandomForest & XGBoost models, logs to MLflow, uploads best model to HF
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =============== Configuration ===============
HF_DATASET_REPO_ID = "nilanjanadevc/tourism-wellness-dataset"
HF_MODEL_REPO_ID = "nilanjanadevc/tourism-wellness-model"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# =============== Initialize APIs ===============
api = HfApi(token=os.getenv("HF_TOKEN"))
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tourism_product_purchase_prediction")

def load_train_test():
    """Load preprocessed train/test splits from HF Dataset Hub."""
    print("="*60)
    print("LOADING DATA FROM HUGGING FACE")
    print("="*60)
    
    base_path = f"hf://datasets/{HF_DATASET_REPO_ID}"
    X_train = pd.read_csv(f"{base_path}/X_train.csv")
    X_test = pd.read_csv(f"{base_path}/X_test.csv")
    y_train = pd.read_csv(f"{base_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{base_path}/y_test.csv").squeeze()
    
    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ y_train distribution:\n{y_train.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train):
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\nNumeric Features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def train_random_forest(preprocessor, X_train, y_train):
    """Train RandomForest with GridSearchCV."""
    print("\n" + "="*60)
    print("RANDOM FOREST - GridSearchCV")
    print("="*60)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }
    
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", rf),
    ])
    
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
    )
    gs.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {gs.best_params_}")
    print(f"✓ Best CV Score: {gs.best_score_:.4f}")
    
    return gs

def train_xgboost(preprocessor, X_train, y_train):
    """Train XGBoost with RandomizedSearchCV."""
    print("\n" + "="*60)
    print("XGBOOST - RandomizedSearchCV")
    print("="*60)
    
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    
    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.9, 1.0],
    }
    
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", xgb_clf),
    ])
    
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        random_state=42,
        verbose=1,
    )
    rs.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {rs.best_params_}")
    print(f"✓ Best CV Score: {rs.best_score_:.4f}")
    
    return rs

def evaluate_and_log(model_name, search_obj, X_train, y_train, X_test, y_test):
    """Evaluate model and log metrics to MLflow."""
    best_model = search_obj.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  - Train Accuracy: {train_acc:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(search_obj.best_params_)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=None,
        )
    
    return {
        "name": model_name,
        "best_model": best_model,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }

def ensure_model_repo():
    """Create HF Model repository if it doesn't exist."""
    try:
        api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type="model")
        print(f"✓ Model repo '{HF_MODEL_REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"✗ Model repo not found. Creating '{HF_MODEL_REPO_ID}'...")
        create_repo(
            repo_id=HF_MODEL_REPO_ID,
            repo_type="model",
            private=False,
        )
        print(f"✓ Model repo created.")

def upload_best_model(best_model, filename="best_tourism_model.joblib"):
    """Save and upload best model to HF Model Hub."""
    joblib.dump(best_model, filename)
    ensure_model_repo()
    
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"✓ Model uploaded to: https://huggingface.co/models/{HF_MODEL_REPO_ID}/{filename}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL TRAINING & HYPERPARAMETER TUNING PIPELINE")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_train_test()
    
    # Build preprocessor
    preprocessor = build_preprocessor(X_train)
    
    # Train models
    rf_search = train_random_forest(preprocessor, X_train, y_train)
    xgb_search = train_xgboost(preprocessor, X_train, y_train)
    
    # Evaluate and log
    rf_result = evaluate_and_log("RandomForest_GridSearch", rf_search, X_train, y_train, X_test, y_test)
    xgb_result = evaluate_and_log("XGBoost_RandomizedSearch", xgb_search, X_train, y_train, X_test, y_test)
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    results = [rf_result, xgb_result]
    comparison_df = pd.DataFrame(results)
    print(comparison_df[["name", "train_accuracy", "test_accuracy"]].to_string(index=False))
    
    # Select best model
    best_result = sorted(results, key=lambda x: x["test_accuracy"], reverse=True)[0]
    print(f"\n✓ Best Model: {best_result['name']} with Test Accuracy: {best_result['test_accuracy']:.4f}")
    
    # Upload best model
    print("\n" + "="*60)
    print("UPLOADING BEST MODEL")
    print("="*60)
    upload_best_model(best_result["best_model"])
    
    print("\n" + "="*60)
    print("✓ TRAINING PIPELINE COMPLETE!")
    print("="*60)
