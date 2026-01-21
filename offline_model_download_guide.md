# Guide: Download Model & Dataset for Offline Training

## Your Model: `google/flan-t5-base`
## Dataset: `tatsu-lab/alpaca` (as seen in your notebook)

---

## Method 1: Using huggingface-cli (Recommended)

### Step 1: Install huggingface-hub on Internet-Connected PC

```bash
pip install huggingface-hub
```

### Step 2: Download the Model

```bash
huggingface-cli download google/flan-t5-base --local-dir ./flan-t5-base --local-dir-use-symlinks False
```

This downloads ALL model files to `./flan-t5-base/` directory.

### Step 3: Download the Dataset

```bash
huggingface-cli download tatsu-lab/alpaca --repo-type dataset --local-dir ./alpaca-dataset --local-dir-use-symlinks False
```

### Step 4: Transfer to Pendrive

Copy both folders:
- `flan-t5-base/` (model)
- `alpaca-dataset/` (dataset)

---

## Method 2: Using Python Script

Create this Python script on your internet-connected PC:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import os

# Create directories
os.makedirs("./offline_models/flan-t5-base", exist_ok=True)
os.makedirs("./offline_datasets/alpaca", exist_ok=True)

print("Downloading model and tokenizer...")
# Download model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Save locally
model.save_pretrained("./offline_models/flan-t5-base")
tokenizer.save_pretrained("./offline_models/flan-t5-base")

print("Model saved!")

print("\nDownloading dataset...")
# Download dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Save locally
dataset.save_to_disk("./offline_datasets/alpaca")

print("Dataset saved!")
print("\nAll files ready for transfer!")
```

Run it:
```bash
python download_offline.py
```

---

## Step 5: Transfer Everything to Offline PC

Copy these folders to your pendrive:
1. `offline_models/flan-t5-base/`
2. `offline_datasets/alpaca/`
3. `python_packages/` (from earlier package download)

---

## Using the Model on Offline PC

### Modified Code for Your Notebook

Replace the model and dataset loading parts in your notebook with:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk

# OFFLINE MODE - Load from local directory
model_path = "E:/offline_models/flan-t5-base"  # Change E: to your pendrive letter
dataset_path = "E:/offline_datasets/alpaca"

print("Loading model from local directory...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

print("Loading dataset from local directory...")
raw_datasets = load_from_disk(dataset_path)

print("Model and dataset loaded successfully!")
```

---

## Complete Offline Setup Checklist

### On Internet-Connected PC:

- [ ] Download Python packages:
  ```bash
  pip download -r requirements.txt -d ./python_packages
  ```

- [ ] Download model:
  ```bash
  huggingface-cli download google/flan-t5-base --local-dir ./flan-t5-base --local-dir-use-symlinks False
  ```

- [ ] Download dataset:
  ```bash
  huggingface-cli download tatsu-lab/alpaca --repo-type dataset --local-dir ./alpaca-dataset --local-dir-use-symlinks False
  ```

### Transfer to Pendrive:
- [ ] Copy `python_packages/` folder
- [ ] Copy `flan-t5-base/` folder
- [ ] Copy `alpaca-dataset/` folder

### On Offline PC:

- [ ] Install packages:
  ```bash
  pip install -r requirements.txt --no-index --find-links E:/python_packages
  ```

- [ ] Copy model and dataset folders to a permanent location (e.g., `C:/ml_models/`)

- [ ] Update notebook paths to point to local directories

- [ ] Add `local_files_only=True` to all `from_pretrained()` calls

---

## File Sizes (Approximate)

- **flan-t5-base model**: ~1 GB
- **alpaca dataset**: ~50-100 MB
- **Python packages**: ~2-3 GB (depending on versions)

**Total**: ~4-5 GB (make sure your pendrive has enough space!)

---

## Troubleshooting

### Error: "Can't find model files"
- Make sure `local_files_only=True` is set
- Check the path is correct (use absolute paths)
- Verify all files were copied from pendrive

### Error: "No module named 'transformers'"
- Install packages first using the offline method
- Check Python version matches (3.12.0)

### Error: "Dataset not found"
- Use `load_from_disk()` instead of `load_dataset()`
- Check dataset path is correct

---

## Quick Commands Summary

**Download everything:**
```bash
# Packages
pip download -r requirements.txt -d ./python_packages

# Model
huggingface-cli download google/flan-t5-base --local-dir ./flan-t5-base --local-dir-use-symlinks False

# Dataset
huggingface-cli download tatsu-lab/alpaca --repo-type dataset --local-dir ./alpaca-dataset --local-dir-use-symlinks False
```

**Install offline:**
```bash
pip install -r requirements.txt --no-index --find-links ./python_packages
```

**Load offline in Python:**
```python
# Model
tokenizer = AutoTokenizer.from_pretrained("./path/to/flan-t5-base", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("./path/to/flan-t5-base", local_files_only=True)

# Dataset
dataset = load_from_disk("./path/to/alpaca-dataset")
```

---

## Additional Tips

1. **Test on internet PC first**: Before transferring, test loading the model/dataset from local directory on your internet-connected PC to make sure everything downloaded correctly.

2. **Keep folder structure**: Don't rename folders or move files around inside them.

3. **Use absolute paths**: On offline PC, use full paths like `C:/ml_models/flan-t5-base` to avoid confusion.

4. **Backup**: Keep a backup copy of downloaded files in case transfer fails.

5. **Cache directory**: You can also set environment variable to prevent accidental downloads:
   ```bash
   set HF_HUB_OFFLINE=1
   ```
   (Windows Command Prompt) or
   ```bash
   export HF_HUB_OFFLINE=1
   ```
   (Linux/Mac/WSL)
