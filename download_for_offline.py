"""
Offline Model & Dataset Downloader
===================================
This script downloads the FLAN-T5 model and Alpaca dataset for offline use.
Run this on a PC with internet connection.

Usage:
    python download_for_offline.py
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import transformers
        import datasets
        import torch
        print("✓ Required packages found")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e.name}")
        print("\nPlease install required packages first:")
        print("pip install transformers datasets torch")
        return False

def download_model(model_name="google/flan-t5-base", output_dir="./offline_models"):
    """Download the model and tokenizer"""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    model_dir = Path(output_dir) / "flan-t5-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DOWNLOADING MODEL: {model_name}")
    print(f"{'='*80}")
    print(f"Saving to: {model_dir.absolute()}")
    print("\nThis may take several minutes (model is ~1 GB)...")
    
    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        print("✓ Tokenizer saved")
        
        # Download model
        print("\n[2/2] Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        print("✓ Model saved")
        
        print(f"\n✓ Model successfully downloaded to: {model_dir.absolute()}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False

def download_dataset(dataset_name="tatsu-lab/alpaca", output_dir="./offline_datasets"):
    """Download the dataset"""
    from datasets import load_dataset
    
    dataset_dir = Path(output_dir) / "alpaca"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DOWNLOADING DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Saving to: {dataset_dir.absolute()}")
    print("\nThis may take a few minutes...")
    
    try:
        print("\nDownloading dataset...")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_dir)
        
        print(f"\n✓ Dataset successfully downloaded to: {dataset_dir.absolute()}")
        print(f"Dataset splits: {list(dataset.keys())}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return False

def get_directory_size(path):
    """Calculate total size of directory"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except:
        pass
    return total

def format_size(size_bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    print(f"\n{'='*80}")
    print("OFFLINE MODEL & DATASET DOWNLOADER")
    print(f"{'='*80}")
    print("\nThis script will download:")
    print("  1. google/flan-t5-base model (~1 GB)")
    print("  2. tatsu-lab/alpaca dataset (~50-100 MB)")
    print("\nMake sure you have:")
    print("  - Active internet connection")
    print("  - At least 2 GB free disk space")
    print("  - Required Python packages installed")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("\n" + "="*80)
    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        sys.exit(0)
    
    # Download model
    model_success = download_model()
    
    # Download dataset
    dataset_success = download_dataset()
    
    # Final summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    
    if model_success:
        model_path = Path("./offline_models/flan-t5-base")
        if model_path.exists():
            size = format_size(get_directory_size(model_path))
            print(f"✓ Model: {model_path.absolute()} ({size})")
        else:
            print(f"✓ Model: Downloaded")
    else:
        print("✗ Model: Failed")
    
    if dataset_success:
        dataset_path = Path("./offline_datasets/alpaca")
        if dataset_path.exists():
            size = format_size(get_directory_size(dataset_path))
            print(f"✓ Dataset: {dataset_path.absolute()} ({size})")
        else:
            print(f"✓ Dataset: Downloaded")
    else:
        print("✗ Dataset: Failed")
    
    if model_success and dataset_success:
        print(f"\n{'='*80}")
        print("SUCCESS! All files downloaded.")
        print(f"{'='*80}")
        print("\nNext steps:")
        print("  1. Copy 'offline_models' folder to your pendrive")
        print("  2. Copy 'offline_datasets' folder to your pendrive")
        print("  3. On offline PC, use local_files_only=True when loading")
        print("\nExample code for offline PC:")
        print("-" * 80)
        print("""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk

# Load model (change path to your location)
tokenizer = AutoTokenizer.from_pretrained(
    "./offline_models/flan-t5-base", 
    local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./offline_models/flan-t5-base",
    local_files_only=True
)

# Load dataset
dataset = load_from_disk("./offline_datasets/alpaca")
        """)
        print("-" * 80)
    else:
        print(f"\n{'='*80}")
        print("FAILED! Some downloads were unsuccessful.")
        print("Please check the errors above and try again.")
        print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
