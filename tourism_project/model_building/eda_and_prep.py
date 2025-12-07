"""
Data Preparation & EDA Script for Tourism Dataset
Author: MLOps Team
Description: Loads data from HF, performs EDA, cleans, preprocesses, and uploads processed splits
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# =============== Configuration ===============
HF_DATASET_REPO_ID = "nilanjanadevc/tourism-wellness-dataset"

# =============== Initialize HF API ===============
api = HfApi(token=os.getenv("HF_TOKEN"))

def load_data():
    """Load the raw tourism dataset from HF Datasets Hub."""
    try:
        raw_data_path = f"hf://datasets/{HF_DATASET_REPO_ID}/tourism.csv"
        df = pd.read_csv(raw_data_path)
        print(f"✓ Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def basic_eda(df):
    """
    Perform basic EDA and log insights.
    This helps understand data characteristics before preprocessing.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    print("\n1. Data Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")

    print("\n2. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   - No missing values found ✓")

    print("\n3. Target Distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).to_string())

    print("\n4. Data Types:")
    print(df.dtypes.to_string())

def clean_data(df):
    """
    Clean the dataset by:
    - Removing duplicates
    - Handling missing values
    - Removing non-predictive columns
    - Fixing data inconsistencies
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {before_dup - len(df)} duplicate rows")

    # Drop rows with missing target
    df = df.dropna(subset=["ProdTaken"])
    print("✓ Removed rows with missing target variable")

    # Drop non-predictive columns
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("✓ Dropped CustomerID (non-predictive identifier)")

    # Drop redundant index column if present
    # This column often appears when a CSV is saved with the DataFrame index.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("\n'Unnamed: 0' column dropped (was likely a redundant index).")
    else:
        cols_lower = df.columns.str.lower()
        if "unnamed: 0" in cols_lower.values:
            unnamed_col = df.columns[cols_lower == "unnamed: 0"][0]
            df = df.drop(columns=[unnamed_col])
            print(f"\n'{unnamed_col}' column dropped (was likely a redundant index).")

    # Fix inconsistencies
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({
            "Fe Male": "Female",
            "fe male": "Female"
        })

    print(f"\n✓ Cleaned dataset shape: {df.shape}")
    return df

def split_and_save(df):
    """
    Split data into train and test sets and save to CSV files.
    Uses stratified splitting to maintain class balance.
    """
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    target_col = "ProdTaken"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs("tourism_project/model_building", exist_ok=True)

    X_train.to_csv("tourism_project/model_building/X_train.csv", index=False)
    X_test.to_csv("tourism_project/model_building/X_test.csv", index=False)
    y_train.to_csv("tourism_project/model_building/y_train.csv", index=False)
    y_test.to_csv("tourism_project/model_building/y_test.csv", index=False)

    print(f"✓ Train set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print("\n✓ All splits saved to tourism_project/model_building/")

def upload_splits_to_hf():
    """Upload preprocessed train and test splits to HF Dataset Hub."""
    print("\n" + "=" * 60)
    print("UPLOADING SPLITS TO HUGGING FACE")
    print("=" * 60)

    output_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    for file_name in output_files:
        file_path = f"tourism_project/model_building/{file_name}"
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset"
            )
            print(f"✓ Uploaded {file_name}")
        except Exception as e:
            print(f"✗ Error uploading {file_name}: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATA PREPARATION & EDA PIPELINE")
    print("=" * 60)

    df_raw = load_data()
    if df_raw is None:
        print("Failed to load data. Exiting...")
        exit(1)

    basic_eda(df_raw)

    df_clean = clean_data(df_raw)

    split_and_save(df_clean)

    upload_splits_to_hf()

    print("\n" + "=" * 60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("=" * 60)
