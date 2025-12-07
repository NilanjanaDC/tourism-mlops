"""
Script to register the tourism dataset on Hugging Face Datasets Hub
Author: MLOps Team
Description: Uploads the raw tourism.csv dataset to HF for version control and accessibility
"""

import os
import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =============== Configuration ===============
DATASET_REPO_ID = "nilanjanadevc/tourism-wellness-dataset"
DATASET_REPO_TYPE = "dataset"
LOCAL_DATA_FOLDER = "tourism_project/data"

# =============== Initialize HF API ===============
api = HfApi(token=os.getenv("HF_TOKEN"))

def ensure_dataset_repo_exists():
    """
    Check if the dataset repository exists on HF.
    If not, create a new dataset repository.
    """
    try:
        repo_info = api.repo_info(repo_id=DATASET_REPO_ID, repo_type=DATASET_REPO_TYPE)
        print(f"✓ Dataset repo '{DATASET_REPO_ID}' already exists.")
        return True
    except RepositoryNotFoundError:
        print(f"✗ Dataset repo '{DATASET_REPO_ID}' not found. Creating...")
        try:
            create_repo(
                repo_id=DATASET_REPO_ID,
                repo_type=DATASET_REPO_TYPE,
                private=False,
            )
            print(f"✓ Dataset repo '{DATASET_REPO_ID}' created successfully!")
            return True
        except Exception as e:
            print(f"✗ Error creating dataset repo: {e}")
            return False

def upload_raw_data():
    """
    Upload the raw data folder containing tourism.csv to HF Dataset Hub.
    """
    try:
        api.upload_folder(
            folder_path=LOCAL_DATA_FOLDER,
            repo_id=DATASET_REPO_ID,
            repo_type=DATASET_REPO_TYPE,
        )
        print(f"✓ Successfully uploaded raw data to HF Dataset Hub!")
        print(f"✓ Dataset Repository: https://huggingface.co/datasets/{DATASET_REPO_ID}")
    except Exception as e:
        print(f"✗ Error uploading data: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("DATA REGISTRATION ON HUGGING FACE DATASETS HUB")
    print("="*60)
    
    # Step 1: Ensure dataset repo exists
    if ensure_dataset_repo_exists():
        # Step 2: Upload raw data
        print("\nUploading raw dataset...")
        upload_raw_data()
    else:
        print("Failed to ensure dataset repository. Exiting...")
    
    print("="*60)
    print("DATA REGISTRATION COMPLETE")
    print("="*60)
