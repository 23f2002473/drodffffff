# Modified Code for Offline Training
# ====================================
# Replace the relevant sections in your notebook with these code snippets

# ============================================================================
# SECTION 1: Model and Dataset Loading (OFFLINE VERSION)
# ============================================================================

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from datasets import load_from_disk  # Changed from load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm

# ============================================================================
# IMPORTANT: Set paths to your local model and dataset directories
# ============================================================================
# Change these paths to where you copied the files on your offline PC
MODEL_PATH = "C:/ml_models/flan-t5-base"  # Update this path!
DATASET_PATH = "C:/ml_datasets/alpaca"      # Update this path!

# Optional: Force offline mode (prevents accidental internet access)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("="*80)
print("OFFLINE TRAINING MODE")
print("="*80)
print(f"Model path: {MODEL_PATH}")
print(f"Dataset path: {DATASET_PATH}")
print("="*80)

# ============================================================================
# Load Model and Tokenizer (OFFLINE)
# ============================================================================
print("\nLoading model and tokenizer from local directory...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True  # IMPORTANT: Prevents internet access
    )
    print("✓ Tokenizer loaded")
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True  # IMPORTANT: Prevents internet access
    )
    print("✓ Model loaded")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if MODEL_PATH is correct")
    print("  2. Verify all model files were copied from pendrive")
    print("  3. Make sure you downloaded the model on internet PC first")
    raise

# ============================================================================
# Load Dataset (OFFLINE)
# ============================================================================
print("\nLoading dataset from local directory...")

try:
    raw_datasets = load_from_disk(DATASET_PATH)  # Use load_from_disk for offline
    print("✓ Dataset loaded")
    print(f"Dataset splits: {list(raw_datasets.keys())}")
    
    # Show sample
    if 'train' in raw_datasets:
        print(f"Training samples: {len(raw_datasets['train'])}")
        print(f"\nSample entry:")
        print(raw_datasets['train'][0])
    
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("\nTroubleshooting:")
    print("  1. Check if DATASET_PATH is correct")
    print("  2. Verify dataset folder was copied from pendrive")
    print("  3. Make sure you downloaded the dataset on internet PC first")
    raise

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Move model to device
base_model.to(device)

# Get model parameters info
total_params = sum(p.numel() for p in base_model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================================
# Rest of your training code continues here...
# (The LoRA setup, training loop, evaluation remain the same)
# ============================================================================


# ============================================================================
# SECTION 2: Loading Trained Model (OFFLINE VERSION)
# ============================================================================
# When loading your fine-tuned model after training:

from peft import PeftModel

print("Loading fine-tuned model...")

# Load base model
base_model_for_inference = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,  # Still use local path
    local_files_only=True
)

# Load LoRA weights (these are saved locally during training)
model = PeftModel.from_pretrained(
    base_model_for_inference, 
    "./flan-t5-alpaca-lora-best"  # Local path where you saved during training
)

model.to(device)
model.eval()

print("✓ Fine-tuned model loaded and ready for inference")


# ============================================================================
# SECTION 3: Environment Variables (Add at the very beginning of notebook)
# ============================================================================
# Add this cell at the very beginning of your notebook to ensure offline mode

import os

# Force offline mode - prevents any accidental downloads
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Optional: Set cache directory if you want to use a specific location
# os.environ['HF_HOME'] = 'C:/huggingface_cache'
# os.environ['TRANSFORMERS_CACHE'] = 'C:/huggingface_cache/transformers'

print("✓ Offline mode enabled")
print("✓ No internet connection will be used")


# ============================================================================
# SECTION 4: Verification Script (Run this to test offline setup)
# ============================================================================
"""
Quick Verification Script
Run this in a separate cell to verify everything is set up correctly
"""

import os
from pathlib import Path

def verify_offline_setup():
    print("="*80)
    print("OFFLINE SETUP VERIFICATION")
    print("="*80)
    
    # Paths to check
    model_path = "C:/ml_models/flan-t5-base"  # Update this!
    dataset_path = "C:/ml_datasets/alpaca"     # Update this!
    
    errors = []
    
    # Check model directory
    print("\n[1/4] Checking model directory...")
    model_dir = Path(model_path)
    if not model_dir.exists():
        errors.append(f"Model directory not found: {model_path}")
        print(f"  ✗ Not found: {model_path}")
    else:
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing = [f for f in required_files if not (model_dir / f).exists()]
        
        # Check for model.safetensors as alternative
        if 'pytorch_model.bin' in missing and (model_dir / 'model.safetensors').exists():
            missing.remove('pytorch_model.bin')
        
        if missing:
            errors.append(f"Missing model files: {', '.join(missing)}")
            print(f"  ✗ Missing files: {', '.join(missing)}")
        else:
            print(f"  ✓ All model files found")
    
    # Check dataset directory
    print("\n[2/4] Checking dataset directory...")
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        errors.append(f"Dataset directory not found: {dataset_path}")
        print(f"  ✗ Not found: {dataset_path}")
    else:
        # Check for dataset files
        if (dataset_dir / 'dataset_info.json').exists() or (dataset_dir / 'state.json').exists():
            print(f"  ✓ Dataset files found")
        else:
            errors.append("Dataset directory exists but appears empty")
            print(f"  ✗ Dataset directory appears incomplete")
    
    # Check packages
    print("\n[3/4] Checking Python packages...")
    required_packages = ['transformers', 'datasets', 'torch', 'peft']
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} installed")
        except ImportError:
            errors.append(f"Package not installed: {package}")
            print(f"  ✗ {package} not installed")
    
    # Check CUDA
    print("\n[4/4] Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA not available - will use CPU (slower)")
    except:
        print(f"  ✗ Cannot check CUDA")
    
    # Final summary
    print("\n" + "="*80)
    if not errors:
        print("✓ ALL CHECKS PASSED - Ready for offline training!")
        print("="*80)
        return True
    else:
        print("✗ ISSUES FOUND:")
        for error in errors:
            print(f"  • {error}")
        print("="*80)
        print("\nPlease fix these issues before starting training.")
        return False

# Run verification
verify_offline_setup()


# ============================================================================
# NOTES AND TIPS
# ============================================================================
"""
IMPORTANT NOTES:

1. ALWAYS use local_files_only=True when loading models/tokenizers
2. Use load_from_disk() instead of load_dataset() for datasets
3. Set environment variables at the start to force offline mode
4. Test the verification script before starting long training jobs
5. Keep a backup of downloaded files

COMMON ERRORS AND SOLUTIONS:

Error: "Can't find model configuration"
→ Solution: Check MODEL_PATH is correct and contains config.json

Error: "Dataset not found" 
→ Solution: Use load_from_disk() instead of load_dataset()

Error: "No module named 'transformers'"
→ Solution: Install packages offline first using pip with --no-index

Error: Connection errors even with local_files_only=True
→ Solution: Set HF_HUB_OFFLINE=1 environment variable

FOLDER STRUCTURE ON OFFLINE PC:

C:/ml_models/
└── flan-t5-base/
    ├── config.json
    ├── pytorch_model.bin (or model.safetensors)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ... (other files)

C:/ml_datasets/
└── alpaca/
    ├── dataset_info.json
    ├── train/
    └── ... (other splits)

DISK SPACE REQUIREMENTS:
- Model files: ~1 GB
- Dataset files: ~50-100 MB
- Training checkpoints: ~500 MB - 1 GB
- Total recommended: At least 5 GB free space
"""
