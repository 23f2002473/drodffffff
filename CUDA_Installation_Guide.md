# Complete CUDA Installation Guide for Offline Windows PC
## For NVIDIA Quadro P6000 with T5 Model Fine-tuning

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Download Phase (Internet-Connected PC)](#download-phase-internet-connected-pc)
3. [Installation Phase (Offline PC)](#installation-phase-offline-pc)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

**Hardware:**
- NVIDIA Quadro P6000 GPU ✅
- At least 50 GB free disk space
- 32+ GB pen drive (or external HDD recommended)

**Software:**
- Windows 10/11 (64-bit)
- Python 3.12 (should be installed already or download offline installer)

---

## 📥 DOWNLOAD PHASE (Internet-Connected PC)

### Step 1: Download Visual Studio 2022 Community

**1.1 Download the Bootstrapper**
- Go to: https://visualstudio.microsoft.com/downloads/
- Click **"Free download"** under "Community 2022"
- Save `vs_community.exe` to a folder (e.g., `C:\VSDownload`)

**1.2 Create Offline Installer**

Open Command Prompt as Administrator and run:

```cmd
cd C:\VSDownload
vs_community.exe --layout C:\VSOffline --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22000 --includeRecommended --lang en-US
```

**What this does:**
- `--layout C:\VSOffline` → Downloads to this folder
- `--add Microsoft.VisualStudio.Workload.NativeDesktop` → C++ development tools
- `--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64` → MSVC compiler (required for CUDA)
- `--add Microsoft.VisualStudio.Component.Windows11SDK.22000` → Windows SDK
- `--includeRecommended` → Recommended components

**⏱️ Time Required:** 30-60 minutes  
**📦 Size:** ~8-10 GB

**1.3 Verify Download**
Check that `C:\VSOffline` contains:
- `vs_setup.exe`
- `certificates` folder
- Multiple package folders

---

### Step 2: Download NVIDIA Quadro P6000 Driver

**2.1 Get the Driver**
- Go to: https://www.nvidia.com/download/index.aspx
- Fill in:
  - Product Type: **Quadro**
  - Product Series: **Quadro Series**
  - Product: **Quadro P6000**
  - Operating System: **Windows 10 64-bit** or **Windows 11 64-bit**
  - Download Type: **Game Ready Driver (GRD)** or **Studio Driver**
- Click **Search** → **Download**

**📦 File:** `xxx.xx-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe` (~700 MB)

---

### Step 3: Download CUDA Toolkit 11.8

**3.1 Download CUDA 11.8**
- Go to: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Select:
  - Operating System: **Windows**
  - Architecture: **x86_64**
  - Version: **10** or **11** (your Windows version)
  - Installer Type: **exe (local)**
- Click **Download**

**📦 File:** `cuda_11.8.0_522.06_windows.exe` (~3.5 GB)

**⚠️ IMPORTANT:** Download the **local** installer, NOT network installer!

---

### Step 4: Download cuDNN

**4.1 Create NVIDIA Developer Account** (if you don't have one)
- Go to: https://developer.nvidia.com/cudnn
- Click **"Download cuDNN"**
- Sign up or log in (free account)

**4.2 Download cuDNN for CUDA 11.x**
- After login, accept terms
- Find: **"cuDNN v8.x.x for CUDA 11.x"**
- Download: **"Local Installer for Windows (Zip)"**

**📦 File:** `cudnn-windows-x86_64-8.x.x.x_cuda11-archive.zip` (~600-800 MB)

---

### Step 5: Download Python (if needed)

**5.1 Download Python 3.12**
- Go to: https://www.python.org/downloads/
- Download **Python 3.12.x** - Windows installer (64-bit)

**📦 File:** `python-3.12.x-amd64.exe` (~25 MB)

---

### Step 6: Download PyTorch with CUDA Support

**6.1 Visit PyTorch Wheel Repository**
- Go to: https://download.pytorch.org/whl/torch_stable.html

**6.2 Download Required Wheels for Python 3.12 + CUDA 11.8**

Search for and download these files (use Ctrl+F):

**For PyTorch 2.4.0:**
```
torch-2.4.0+cu118-cp312-cp312-win_amd64.whl
torchvision-0.19.0+cu118-cp312-cp312-win_amd64.whl
torchaudio-2.4.0+cu118-cp312-cp312-win_amd64.whl
```

**Explanation:**
- `cu118` = CUDA 11.8
- `cp312` = Python 3.12
- `win_amd64` = Windows 64-bit

**📦 Size:** ~2-3 GB total

---

### Step 7: Download All Python Dependencies

**7.1 Create Requirements File**

Save this as `requirements.txt`:
```txt
transformers>=4.30.0
peft>=0.5.0
datasets>=2.0.0
evaluate>=0.4.0
bert-score>=0.3.13
rouge-score>=0.1.2
sacrebleu>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.5.0
huggingface-hub>=0.7.0
accelerate>=0.20.0
```

**7.2 Download All Packages**

Open Command Prompt and run:
```cmd
mkdir C:\PythonPackages
pip download -r requirements.txt -d C:\PythonPackages
```

**⏱️ Time Required:** 10-20 minutes  
**📦 Size:** ~2-3 GB

---

### Step 8: Organize Files for Transfer

**8.1 Create Folder Structure**

Create this structure on your internet PC:
```
OfflineInstall/
├── 1_VisualStudio/
│   └── VSOffline/ (entire folder)
├── 2_Drivers/
│   └── xxx-quadro-driver.exe
├── 3_CUDA/
│   ├── cuda_11.8.0_522.06_windows.exe
│   └── cudnn-windows-x86_64-8.x.x.x_cuda11-archive.zip
├── 4_Python/
│   └── python-3.12.x-amd64.exe (if needed)
├── 5_PyTorch/
│   ├── torch-2.4.0+cu118-cp312-cp312-win_amd64.whl
│   ├── torchvision-0.19.0+cu118-cp312-cp312-win_amd64.whl
│   └── torchaudio-2.4.0+cu118-cp312-cp312-win_amd64.whl
├── 6_Packages/
│   └── (all downloaded packages from pip download)
└── requirements.txt
```

**8.2 Copy to Pen Drive**
- Copy the entire `OfflineInstall` folder to your pen drive
- **Total Size:** ~20-25 GB

---

## 🔧 INSTALLATION PHASE (Offline PC)

### **⚠️ CRITICAL: Installation Order Matters!**

Follow this exact order:
1. ✅ Python (if needed)
2. ✅ Visual Studio
3. ✅ NVIDIA Driver
4. ✅ **RESTART** 🔄
5. ✅ CUDA Toolkit
6. ✅ cuDNN
7. ✅ PyTorch
8. ✅ Python Packages

---

### Step 1: Install Python (if not already installed)

**1.1 Run Python Installer**
```
E:\OfflineInstall\4_Python\python-3.12.x-amd64.exe
```

**1.2 Installation Options:**
- ✅ Check **"Add Python 3.12 to PATH"** (IMPORTANT!)
- Click **"Customize installation"**
- ✅ Check all optional features
- Click **Next**
- ✅ Check **"Install for all users"**
- Click **Install**

**1.3 Verify Python**
Open Command Prompt:
```cmd
python --version
```
Should show: `Python 3.12.x`

---

### Step 2: Install Visual Studio 2022

**2.1 Install Certificates First (CRITICAL!)**

Navigate to pen drive in Command Prompt as Administrator:
```cmd
cd E:\OfflineInstall\1_VisualStudio\VSOffline\certificates
installCertificates.bat
```

Wait for completion message.

**2.2 Run Visual Studio Installer**
```cmd
cd E:\OfflineInstall\1_VisualStudio\VSOffline
vs_setup.exe
```

**2.3 Select Workloads**
- ✅ Check **"Desktop development with C++"**
- On the right side panel, verify these are selected:
  - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
  - ✅ Windows 11 SDK (or Windows 10 SDK)
  - ✅ C++ CMake tools for Windows
  - ✅ C++ core features

**2.4 Install**
- Click **Install** button
- ⏱️ Time Required: 20-40 minutes
- **DO NOT restart yet** when installation completes

**2.5 Verify Installation**

Open Command Prompt (new window):
```cmd
where cl
```
Should show path like:
```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.xx.xxxxx\bin\Hostx64\x64\cl.exe
```

---

### Step 3: Install NVIDIA Quadro Driver

**3.1 Run Driver Installer**
```
E:\OfflineInstall\2_Drivers\xxx-quadro-driver.exe
```

**3.2 Installation Options:**
- Choose **"Custom (Advanced)"** installation
- ✅ Check:
  - Graphics Driver
  - PhysX
  - CUDA components (if available)
- Uncheck:
  - GeForce Experience (not needed for Quadro)
  - 3D Vision (optional)
- Click **Next** → **Install**

**⏱️ Time Required:** 5-10 minutes

**3.3 RESTART YOUR PC NOW** 🔄

**⚠️ IMPORTANT:** After restart, Windows might take a few minutes to configure the new driver. Wait before proceeding.

---

### Step 4: Install CUDA Toolkit 11.8

**4.1 Run CUDA Installer**

After restart:
```
E:\OfflineInstall\3_CUDA\cuda_11.8.0_522.06_windows.exe
```

**4.2 Installation Options**

When prompted:
- Select **"Custom (Advanced)"** installation
- You'll see components:
  - ✅ CUDA Toolkit (keep all sub-components)
  - ✅ CUDA Development (keep all)
  - ✅ CUDA Runtime (keep all)
  - ✅ CUDA Documentation (optional - uncheck to save space)
  - ✅ CUDA Samples (optional - uncheck to save space)
  - ⚠️ Driver components → Check if version is OLDER than what you just installed
    - If older: Uncheck driver components
    - If newer or same: Keep checked

**4.3 Installation Location**

Default is fine:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

**4.4 Complete Installation**
- Click **Next** → **Install**
- ⏱️ Time Required: 10-15 minutes
- Allow firewall permissions if prompted

**4.5 Verify CUDA Installation**

Open NEW Command Prompt:
```cmd
nvcc --version
```

Should show:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

**4.6 Check Environment Variables**

```cmd
echo %CUDA_PATH%
```
Should show: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

If not set, manually add:
- Open **System Properties** → **Environment Variables**
- Under System variables, click **New**:
  - Variable name: `CUDA_PATH`
  - Variable value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

Also verify PATH includes:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

---

### Step 5: Install cuDNN

**5.1 Extract cuDNN Archive**

Right-click on:
```
E:\OfflineInstall\3_CUDA\cudnn-windows-x86_64-8.x.x.x_cuda11-archive.zip
```
Extract to a temporary location (e.g., `C:\Temp\cudnn`)

**5.2 Copy cuDNN Files to CUDA Directory**

You'll see folders: `bin`, `include`, `lib`

**Copy files manually:**

Open two File Explorer windows:

**Window 1 (Source):** `C:\Temp\cudnn`  
**Window 2 (Destination):** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

**Copy operations:**

1. **bin folder:**
   - Copy all `.dll` files from `C:\Temp\cudnn\bin\`
   - To: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`

2. **include folder:**
   - Copy all `.h` files from `C:\Temp\cudnn\include\`
   - To: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`

3. **lib folder:**
   - Copy all files from `C:\Temp\cudnn\lib\x64\`
   - To: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

**⚠️ IMPORTANT:** You may need Administrator permissions. If prompted, click "Continue" or "Replace".

**5.3 Verify cuDNN Files**

Check that these files exist:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\cudnn.h
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\cudnn.lib
```

---

### Step 6: Install PyTorch with CUDA Support

**6.1 Install PyTorch Wheels**

Open Command Prompt and navigate to PyTorch folder:
```cmd
cd E:\OfflineInstall\5_PyTorch
```

Install in this order:
```cmd
pip install torch-2.4.0+cu118-cp312-cp312-win_amd64.whl
pip install torchvision-0.19.0+cu118-cp312-cp312-win_amd64.whl
pip install torchaudio-2.4.0+cu118-cp312-cp312-win_amd64.whl
```

**⏱️ Time Required:** 2-5 minutes

**6.2 Verify PyTorch Installation**

Open Python:
```cmd
python
```

Run these commands:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Expected Output:**
```
PyTorch version: 2.4.0+cu118
CUDA available: True
CUDA version: 11.8
GPU device: Quadro P6000
Number of GPUs: 1
```

**⚠️ If CUDA available shows False, see Troubleshooting section!**

Type `exit()` to quit Python.

---

### Step 7: Install Python Packages

**7.1 Install from Offline Packages**

```cmd
cd E:\OfflineInstall\6_Packages
pip install --no-index --find-links=. -r E:\OfflineInstall\requirements.txt
```

**What this does:**
- `--no-index` → Don't use PyPI (offline mode)
- `--find-links=.` → Look for packages in current directory
- `-r requirements.txt` → Install all requirements

**⏱️ Time Required:** 5-10 minutes

**7.2 Verify Key Packages**

```cmd
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print('PEFT installed successfully')"
python -c "import datasets; print('Datasets installed successfully')"
python -c "import accelerate; print('Accelerate installed successfully')"
```

---

## ✅ VERIFICATION

### Complete System Check

**Create a test script:** `cuda_test.py`

```python
import torch
import transformers
import sys

print("="*50)
print("CUDA & PyTorch Environment Check")
print("="*50)

# Python version
print(f"\n1. Python version: {sys.version}")

# PyTorch
print(f"\n2. PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version()}")

# GPU info
if torch.cuda.is_available():
    print(f"\n3. GPU Information:")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Memory info
    print(f"\n4. GPU Memory:")
    print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Simple tensor test
    print(f"\n5. GPU Tensor Test:")
    try:
        x = torch.rand(5, 3).cuda()
        print(f"   ✅ Successfully created tensor on GPU")
        print(f"   Tensor device: {x.device}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("\n❌ CUDA is NOT available!")

# Transformers
print(f"\n6. Transformers version: {transformers.__version__}")

# Test T5 model loading (small test)
print(f"\n7. Testing T5 model loading...")
try:
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small", local_files_only=False)
    print(f"   ✅ T5 tokenizer loaded successfully")
except Exception as e:
    print(f"   ⚠️  Note: {e}")
    print(f"   (This is normal if you haven't downloaded T5 model yet)")

print("\n" + "="*50)
print("Environment check complete!")
print("="*50)
```

**Run the test:**
```cmd
python cuda_test.py
```

---

## 🎯 Quick GPU Performance Test

**Create:** `gpu_benchmark.py`

```python
import torch
import time

print("GPU Benchmark Test")
print("="*50)

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    exit()

device = torch.device("cuda")
print(f"Using device: {torch.cuda.get_device_name(0)}\n")

# Matrix multiplication benchmark
sizes = [1000, 2000, 4000, 8000]

for size in sizes:
    # Create random matrices on GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Matrix {size}x{size}: {(end-start)*1000:.2f} ms")

print("\n✅ GPU is working correctly!")
```

**Run:**
```cmd
python gpu_benchmark.py
```

---

## 🔍 TROUBLESHOOTING

### Issue 1: "CUDA available: False" in PyTorch

**Causes & Solutions:**

**A. Wrong PyTorch version installed**
```cmd
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Should show: 2.4.0+cu118
# If it shows just "2.4.0" (no +cu118), you installed CPU version
```

**Solution:** Reinstall correct wheel:
```cmd
pip uninstall torch torchvision torchaudio
pip install E:\OfflineInstall\5_PyTorch\torch-2.4.0+cu118-cp312-cp312-win_amd64.whl
```

**B. NVIDIA driver not properly installed**
```cmd
nvidia-smi
```

Should show GPU info. If error:
- Reinstall NVIDIA driver
- Restart PC

**C. CUDA not in PATH**

Check environment variables:
```cmd
echo %CUDA_PATH%
echo %PATH%
```

Should include CUDA paths. If missing, add manually.

---

### Issue 2: "nvcc: command not found"

**Solution:**

Add to PATH:
1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Edit **PATH** (System variables)
3. Add:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```
4. Click OK
5. **Close and reopen** Command Prompt

---

### Issue 3: cuDNN not found errors

**Error message:** `Could not load dynamic library 'cudnn64_8.dll'`

**Solution:**

1. Verify file exists:
   ```cmd
   dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll"
   ```

2. If missing, re-copy cuDNN files (Step 5 of installation)

3. Add to PATH if needed:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```

---

### Issue 4: Visual Studio not found during CUDA install

**Error:** "Visual Studio integration will be disabled"

**This is usually OK if:**
- You already installed Visual Studio
- CUDA Toolkit still installs successfully

**To fix properly:**
1. Ensure Visual Studio installed FIRST
2. Verify `cl.exe` exists:
   ```cmd
   where cl
   ```
3. Reinstall CUDA if needed

---

### Issue 5: Out of memory errors when training

**Error:** `CUDA out of memory`

**Solutions:**

**A. Reduce batch size in your training script**

**B. Clear cache:**
```python
import torch
torch.cuda.empty_cache()
```

**C. Check memory usage:**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

**D. Use gradient accumulation:**
- Train with smaller batches
- Accumulate gradients over multiple steps

---

### Issue 6: Package compatibility issues

**Error:** Version conflicts during pip install

**Solution:**

Install packages one by one to identify conflicts:
```cmd
pip install --no-index --find-links=E:\OfflineInstall\6_Packages transformers
pip install --no-index --find-links=E:\OfflineInstall\6_Packages peft
# ... and so on
```

Or create virtual environment:
```cmd
python -m venv t5_env
t5_env\Scripts\activate
# Then install packages
```

---

### Issue 7: T5 model download fails (offline mode)

Since you're offline, you'll need to download models separately.

**On internet PC:**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"  # or t5-base, t5-large, etc.

# Download model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("./t5-small-local")
tokenizer.save_pretrained("./t5-small-local")
```

Copy `t5-small-local` folder to pen drive, then on offline PC:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("E:\OfflineInstall\models\t5-small-local")
tokenizer = T5Tokenizer.from_pretrained("E:\OfflineInstall\models\t5-small-local")
```

---

## 📊 Final Verification Checklist

Run through this checklist:

- [ ] Python 3.12 installed and in PATH
- [ ] Visual Studio 2022 with C++ tools installed
- [ ] NVIDIA driver installed (check with `nvidia-smi`)
- [ ] CUDA 11.8 installed (check with `nvcc --version`)
- [ ] cuDNN files copied correctly
- [ ] PyTorch installed with CUDA support (`torch.cuda.is_available()` returns `True`)
- [ ] All Python packages installed
- [ ] GPU recognized by PyTorch (`torch.cuda.get_device_name(0)` shows "Quadro P6000")
- [ ] Simple GPU tensor operations work
- [ ] No errors in `cuda_test.py` script

---

## 🚀 Next Steps: Training T5

Once everything is verified, you can start training:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Your training code here...
```

---

## 📝 Summary of Installation Sizes

| Component | Download Size | Installed Size |
|-----------|--------------|----------------|
| Visual Studio | 8-10 GB | 20-25 GB |
| NVIDIA Driver | 700 MB | 1.5 GB |
| CUDA Toolkit | 3.5 GB | 5 GB |
| cuDNN | 800 MB | 1.5 GB |
| PyTorch | 2.5 GB | 4 GB |
| Python Packages | 2-3 GB | 4-5 GB |
| **Total** | **~20 GB** | **~40 GB** |

---

## ⏱️ Total Time Estimate

- Download phase: 2-4 hours (depending on internet speed)
- Installation phase: 1-2 hours
- **Total: 3-6 hours**

---

## 💾 Recommended Pen Drive Size

**Minimum:** 32 GB  
**Recommended:** 64 GB or external HDD  
(Allows space for models, datasets, and future updates)

---

## ✨ Tips for Success

1. **Follow the exact order** - Don't skip steps
2. **Restart when instructed** - Essential for drivers
3. **Run Command Prompt as Administrator** when needed
4. **Keep installation files** - In case you need to reinstall
5. **Document any errors** - Screenshot error messages for troubleshooting
6. **Test each step** - Don't proceed if verification fails
7. **Be patient** - Installations take time, especially Visual Studio

---

## 📞 Support Resources

If you encounter issues:
- NVIDIA Developer Forums: https://forums.developer.nvidia.com/
- PyTorch Forums: https://discuss.pytorch.org/
- Stack Overflow: Search for specific error messages
- Hugging Face Forums: https://discuss.huggingface.co/

---

## 📜 License & Attribution

This guide is provided as-is for educational purposes. All software mentioned:
- Visual Studio: Microsoft Corporation
- CUDA & cuDNN: NVIDIA Corporation
- PyTorch: PyTorch Foundation
- Transformers: Hugging Face Inc.

Please refer to their respective licenses for usage terms.

---

**Good luck with your T5 model fine-tuning! Your Quadro P6000 with 24GB memory is excellent for this task.** 🎉

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Target System:** Windows 10/11 with NVIDIA Quadro P6000  
**CUDA Version:** 11.8  
**PyTorch Version:** 2.4.0
