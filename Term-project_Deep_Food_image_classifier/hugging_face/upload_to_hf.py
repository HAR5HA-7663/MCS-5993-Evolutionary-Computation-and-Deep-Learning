"""
Upload American Food Classifier to Hugging Face Hub
Repository: HAR5HA-YELLELA/american_food_classifier

Usage:
    1. First, copy your model file to hugging_face folder:
       cp optimized_model_best.keras hugging_face/
       
    2. Run this script:
       cd hugging_face
       python upload_to_hf.py
"""

import os
import shutil
from pathlib import Path

# Check if huggingface_hub is installed
try:
    from huggingface_hub import login, upload_folder, HfApi
except ImportError:
    print("❌ huggingface_hub not installed!")
    print("   Run: pip install huggingface_hub")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
REPO_ID = "HAR5HA-YELLELA/american_food_classifier"
MODEL_FILE = "optimized_model_best.keras"

def check_files():
    """Check if all required files exist"""
    print("📋 Checking required files...")
    
    required_files = [
        "README.md",
        "class_indices.json",
        "classes.json",
        MODEL_FILE
    ]
    
    missing = []
    for f in required_files:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024 * 1024)  # MB
            print(f"   ✓ {f} ({size:.2f} MB)" if size > 1 else f"   ✓ {f}")
        else:
            print(f"   ❌ {f} - MISSING!")
            missing.append(f)
    
    return missing

def main():
    print("=" * 60)
    print("🤗 Hugging Face Model Upload Script")
    print(f"   Repository: {REPO_ID}")
    print("=" * 60)
    
    # Check files
    missing = check_files()
    
    if MODEL_FILE in missing:
        print(f"\n⚠️  Model file '{MODEL_FILE}' not found!")
        print("\n   Please copy your model file first:")
        print(f"   cp ../optimized_model_best.keras .")
        print("\n   Or if running from term_project folder:")
        print(f"   cp optimized_model_best.keras hugging_face/")
        return
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return
    
    print("\n" + "=" * 60)
    
    # Login to Hugging Face
    print("\n🔑 Logging in to Hugging Face...")
    print("   (A browser window may open for authentication)\n")
    
    try:
        login()
        print("   ✓ Login successful!")
    except Exception as e:
        print(f"   ❌ Login failed: {e}")
        print("\n   Alternative: Set HF_TOKEN environment variable")
        print("   export HF_TOKEN='your_token_here'")
        return
    
    # Upload to Hugging Face
    print("\n" + "=" * 60)
    print("📤 Uploading to Hugging Face Hub...")
    print(f"   Repository: https://huggingface.co/{REPO_ID}")
    print("   This may take a few minutes for large model files...\n")
    
    try:
        upload_folder(
            folder_path=".",
            repo_id=REPO_ID,
            repo_type="model",
            ignore_patterns=["*.py", "__pycache__", ".git", "*.pyc"],
            commit_message="Upload American Food Classifier model with ES(1+1) optimization"
        )
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Model uploaded to Hugging Face!")
        print("=" * 60)
        print(f"\n🔗 View your model at:")
        print(f"   https://huggingface.co/{REPO_ID}")
        print("\n📖 Usage:")
        print("   from huggingface_hub import hf_hub_download")
        print(f'   model_path = hf_hub_download(repo_id="{REPO_ID}", filename="{MODEL_FILE}")')
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify you have write access to the repository")
        print("3. Create the repository first at: https://huggingface.co/new")
        print(f"   Repository name: {REPO_ID.split('/')[1]}")

if __name__ == "__main__":
    main()
