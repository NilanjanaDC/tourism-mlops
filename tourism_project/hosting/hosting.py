"""
Script to upload deployment files to Hugging Face Space
Author: MLOps Team
Description: Automates the upload of the Streamlit app to HF Space for deployment
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError



# =============== Configuration ===============
HF_SPACE_REPO_ID = "nilanjanadevc/tourism-wellness-space"
DEPLOYMENT_FOLDER = "tourism_project/deployment"

# =============== Initialize HF API ===============
api = HfApi(token=os.getenv('HF_TOKEN'))

def ensure_space_exists():
    """
    Check if the Hugging Face Space exists.
    If not, create a new Space.
    """
    try:
        repo_info = api.repo_info(repo_id=HF_SPACE_REPO_ID, repo_type="space")
        print(f"✓ Space '{HF_SPACE_REPO_ID}' already exists.")
        return True
    except RepositoryNotFoundError:
        print(f"✗ Space '{HF_SPACE_REPO_ID}' not found. Creating a new Space...")
        try:
            create_repo(
                repo_id=HF_SPACE_REPO_ID,
                repo_type="space",
                space_sdk="docker",
                private=False,
            )
            print(f"✓ Space '{HF_SPACE_REPO_ID}' created successfully!")
            return True
        except Exception as e:
            print(f"✗ Error creating Space: {e}")
            return False

def upload_deployment_files():
    """
    Upload all deployment files (app.py, Dockerfile, requirements.txt) to HF Space.
    """
    try:
        api.upload_folder(
            folder_path=DEPLOYMENT_FOLDER,
            repo_id=HF_SPACE_REPO_ID,
            repo_type="space",
            path_in_repo="",
        )
        print(f"✓ Successfully uploaded deployment files to HF Space!")
        print(f"✓ Space URL: https://huggingface.co/spaces/{HF_SPACE_REPO_ID}")
    except Exception as e:
        print(f"✗ Error uploading files: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("HOSTING DEPLOYMENT TO HUGGING FACE SPACE")
    print("="*60)
    
    # Step 1: Ensure Space exists
    if ensure_space_exists():
        # Step 2: Upload deployment files
        print("\nUploading deployment files...")
        upload_deployment_files()
    else:
        print("Failed to ensure Space exists. Exiting...")
    
    print("="*60)
    print("HOSTING PROCESS COMPLETE")
    print("="*60)
