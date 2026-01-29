# Complete CUDA Installation Guide for Ubuntu System
## For NVIDIA Quadro P6000 with T5 Model Fine-tuning (With Internet Access)

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-Installation Steps](#pre-installation-steps)
3. [Installation Phase](#installation-phase)
4. [Python Environment Setup](#python-environment-setup)
5. [PyTorch and Dependencies Installation](#pytorch-and-dependencies-installation)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

**Hardware:**
- NVIDIA Quadro P6000 GPU ✅
- At least 50 GB free disk space
- 16+ GB RAM (recommended)

**Software:**
- Ubuntu 20.04 LTS or Ubuntu 22.04 LTS (64-bit)
- Internet connection (required)
- Sudo/root access

---

## 🔍 PRE-INSTALLATION STEPS

### Step 1: Check System Information

**1.1 Verify Ubuntu Version**

```bash
lsb_release -a
```

Expected output:
```
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.x LTS
Release:        22.04
Codename:       jammy
```

**1.2 Check if NVIDIA GPU is Detected**

```bash
lspci | grep -i nvidia
```

Should show something like:
```
01:00.0 VGA compatible controller: NVIDIA Corporation GP102 [Quadro P6000] (rev a1)
```

**1.3 Check Kernel Version**

```bash
uname -r
```

**1.4 Check Disk Space**

```bash
df -h
```

Ensure you have at least 50 GB free on your root partition.

---

### Step 2: Update System

**2.1 Update Package Lists**

```bash
sudo apt update
```

**2.2 Upgrade Existing Packages**

```bash
sudo apt upgrade -y
```

**2.3 Install Essential Tools**

```bash
sudo apt install -y wget curl git vim nano
```

---

### Step 3: Remove Old NVIDIA Drivers (If Any)

**3.1 Check for Existing NVIDIA Drivers**

```bash
nvidia-smi
```

If this works, you have drivers installed. If you want a fresh install:

**3.2 Remove Old NVIDIA Drivers**

```bash
# Remove old drivers
sudo apt purge -y nvidia* libnvidia*

# Remove CUDA if previously installed
sudo apt purge -y cuda*

# Autoremove unused packages
sudo apt autoremove -y

# Clean up
sudo apt autoclean
```

**3.3 Disable Nouveau Driver (Open-source NVIDIA driver)**

```bash
# Create blacklist file
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

# Verify the file
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
```

Should show:
```
blacklist nouveau
options nouveau modeset=0
```

**3.4 Update initramfs**

```bash
sudo update-initramfs -u
```

**3.5 Reboot System**

```bash
sudo reboot
```

**⚠️ IMPORTANT:** After reboot, continue to the next step.

---

## 🔧 INSTALLATION PHASE

### **⚠️ CRITICAL: Installation Order**

Follow this exact order:
1. ✅ Build Essential Tools
2. ✅ NVIDIA Driver
3. ✅ **REBOOT** 🔄
4. ✅ CUDA Toolkit
5. ✅ cuDNN
6. ✅ Python 3.12 and pip
7. ✅ PyTorch with CUDA
8. ✅ Python Packages

---

### Step 1: Install Build Essential Tools

**1.1 Install gcc, g++, make**

```bash
sudo apt update
sudo apt install -y build-essential
```

**1.2 Install Linux Headers**

```bash
sudo apt install -y linux-headers-$(uname -r)
```

**1.3 Install DKMS (Dynamic Kernel Module Support)**

```bash
sudo apt install -y dkms
```

**1.4 Verify Installation**

```bash
gcc --version
make --version
```

Should show version information for both.

---

### Step 2: Install NVIDIA Driver

**2.1 Add NVIDIA PPA Repository**

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
```

**2.2 Find Available Drivers**

```bash
ubuntu-drivers devices
```

This will show recommended drivers. Look for something like:
```
driver   : nvidia-driver-550 - distro non-free recommended
driver   : nvidia-driver-545 - distro non-free
```

**2.3 Install Recommended Driver**

**Option A: Automatic Installation (Recommended)**

```bash
sudo ubuntu-drivers autoinstall
```

**Option B: Manual Installation (Choose specific version)**

```bash
# Install specific driver version (e.g., 550)
sudo apt install -y nvidia-driver-550
```

**⏱️ Time Required:** 5-10 minutes

**2.4 REBOOT SYSTEM** 🔄

```bash
sudo reboot
```

**⚠️ CRITICAL:** You MUST reboot after driver installation!

---

### Step 3: Verify NVIDIA Driver Installation

**3.1 Check Driver with nvidia-smi**

After reboot, run:

```bash
nvidia-smi
```

**Expected Output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54       Driver Version: 550.54       CUDA Version: 12.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro P6000        Off  | 00000000:01:00.0 Off |                  Off |
| 26%   35C    P8    15W / 250W |      0MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**3.2 Check Driver Version**

```bash
cat /proc/driver/nvidia/version
```

---

### Step 4: Install CUDA Toolkit 11.8

**4.1 Download CUDA 11.8 Installer**

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

**4.2 Make Installer Executable**

```bash
chmod +x cuda_11.8.0_520.61.05_linux.run
```

**4.3 Run CUDA Installer**

```bash
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**4.4 Installation Options**

During installation:
1. Accept the license agreement (type `accept`)
2. **IMPORTANT:** When asked to install driver, select **NO** (you already installed it)
3. Select these options:
   - ✅ CUDA Toolkit 11.8
   - ✅ CUDA Samples 11.8
   - ✅ CUDA Demo Suite 11.8
   - ✅ CUDA Documentation 11.8
   - ❌ Driver (uncheck - already installed)

4. Accept default installation path: `/usr/local/cuda-11.8`

**⏱️ Time Required:** 10-15 minutes

**4.5 Set Up Environment Variables**

Add CUDA to your PATH:

```bash
# Open bashrc
nano ~/.bashrc
```

Add these lines at the end:

```bash
# CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

Save and exit (Ctrl+O, Enter, Ctrl+X)

**4.6 Apply Changes**

```bash
source ~/.bashrc
```

**4.7 Verify CUDA Installation**

```bash
nvcc --version
```

**Expected Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

**4.8 Verify CUDA Path**

```bash
echo $CUDA_HOME
echo $PATH | grep cuda
```

---

### Step 5: Install cuDNN

**5.1 Download cuDNN (Requires NVIDIA Account)**

**Manual Download:**
1. Go to: https://developer.nvidia.com/cudnn
2. Sign in/Sign up (free)
3. Download: **cuDNN v8.x.x for CUDA 11.x** (Linux x86_64)
4. Choose: **"Local Installer for Linux x86_64 (Tar)"**

**5.2 Extract cuDNN Archive**

Assuming you downloaded to `~/Downloads`:

```bash
cd ~/Downloads
tar -xvf cudnn-linux-x86_64-8.*.tar.xz
```

**5.3 Copy cuDNN Files to CUDA Directory**

```bash
# Copy include files
sudo cp cudnn-linux-x86_64-8.*/include/cudnn*.h /usr/local/cuda-11.8/include

# Copy library files
sudo cp cudnn-linux-x86_64-8.*/lib/libcudnn* /usr/local/cuda-11.8/lib64

# Set permissions
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*
```

**5.4 Verify cuDNN Installation**

```bash
# Check if files exist
ls -la /usr/local/cuda-11.8/include/cudnn*.h
ls -la /usr/local/cuda-11.8/lib64/libcudnn*
```

**5.5 Update Linker Cache**

```bash
sudo ldconfig
```

---

## 🐍 PYTHON ENVIRONMENT SETUP

### Step 6: Install Python 3.12

**6.1 Add deadsnakes PPA (for Python 3.12)**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
```

**6.2 Install Python 3.12**

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

**6.3 Install pip for Python 3.12**

```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip
python3.12 get-pip.py

# Verify installation
python3.12 --version
python3.12 -m pip --version
```

**Expected Output:**
```
Python 3.12.x
pip 24.x from /home/username/.local/lib/python3.12/site-packages/pip (python 3.12)
```

**6.4 Create Virtual Environment (Recommended)**

```bash
# Create project directory
mkdir ~/t5-finetuning
cd ~/t5-finetuning

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

Your prompt should now show `(venv)`.

**6.5 Upgrade pip, setuptools, wheel**

```bash
pip install --upgrade pip setuptools wheel
```

---

## 📦 PYTORCH AND DEPENDENCIES INSTALLATION

### Step 7: Install PyTorch with CUDA 11.8 Support

**7.1 Install PyTorch, torchvision, torchaudio**

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**⏱️ Time Required:** 5-10 minutes

**7.2 Verify PyTorch Installation**

```bash
python3.12 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3.12 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3.12 -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python3.12 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected Output:**
```
PyTorch: 2.4.0+cu118
CUDA Available: True
CUDA Version: 11.8
GPU: Quadro P6000
```

**⚠️ If CUDA Available shows False, see Troubleshooting section!**

---

### Step 8: Install Python Dependencies

**8.1 Create requirements.txt**

```bash
cat > requirements.txt << 'EOF'
# Core ML/DL frameworks
torch>=2.4.0
transformers>=4.30.0

# PEFT for LoRA
peft>=0.5.0

# Dataset handling
datasets>=2.0.0

# Evaluation metrics
evaluate>=0.4.0
bert-score>=0.3.13
rouge-score>=0.1.2
sacrebleu>=2.0.0

# Data processing and utilities
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Progress bars and visualization
tqdm>=4.62.0
matplotlib>=3.5.0

# Hugging Face Hub
huggingface-hub>=0.7.0

# Additional dependencies
accelerate>=0.20.0
EOF
```

**8.2 Install All Dependencies**

```bash
pip install -r requirements.txt
```

**⏱️ Time Required:** 10-20 minutes

**8.3 Verify Key Packages**

```bash
python3.12 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3.12 -c "import peft; print('PEFT: OK')"
python3.12 -c "import datasets; print('Datasets: OK')"
python3.12 -c "import accelerate; print('Accelerate: OK')"
```

---

## ✅ VERIFICATION

### Complete System Check

**Create verification script:** `cuda_test.py`

```python
#!/usr/bin/env python3.12
import torch
import transformers
import sys
import subprocess

print("=" * 60)
print("CUDA & PyTorch Environment Check for Ubuntu")
print("=" * 60)

# System info
print("\n1. System Information:")
try:
    result = subprocess.run(['lsb_release', '-d'], capture_output=True, text=True)
    print(f"   OS: {result.stdout.split(':')[1].strip()}")
except:
    print("   OS: Unable to determine")

try:
    result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
    print(f"   Kernel: {result.stdout.strip()}")
except:
    print("   Kernel: Unable to determine")

# Python version
print(f"\n2. Python version: {sys.version}")

# PyTorch
print(f"\n3. PyTorch Information:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version()}")
print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")

# NVIDIA Driver
print(f"\n4. NVIDIA Driver:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    print(f"   Driver Version: {result.stdout.strip()}")
except:
    print("   Driver: Unable to query")

# GPU info
if torch.cuda.is_available():
    print(f"\n5. GPU Information:")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Memory info
    print(f"\n6. GPU Memory:")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    print(f"   Total: {total_mem:.2f} GB")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Cached: {cached:.2f} GB")
    print(f"   Free: {total_mem - allocated:.2f} GB")
    
    # Simple tensor test
    print(f"\n7. GPU Tensor Test:")
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print(f"   ✅ Successfully created and computed tensors on GPU")
        print(f"   Tensor device: {x.device}")
        print(f"   Computation result shape: {z.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("\n❌ CUDA is NOT available!")
    print("   Please check:")
    print("   - NVIDIA driver installation")
    print("   - CUDA installation")
    print("   - PyTorch CUDA build")

# Transformers
print(f"\n8. Transformers version: {transformers.__version__}")

# Environment variables
print(f"\n9. Environment Variables:")
import os
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# CUDA compiler
print(f"\n10. CUDA Compiler:")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
    print(f"   {version_line.strip()}")
except:
    print("   nvcc not found in PATH")

print("\n" + "=" * 60)
print("Environment check complete!")
print("=" * 60)
```

**Run the verification:**

```bash
# Make script executable
chmod +x cuda_test.py

# Run it
python3.12 cuda_test.py
```

---

### GPU Performance Benchmark

**Create benchmark script:** `gpu_benchmark.py`

```python
#!/usr/bin/env python3.12
import torch
import time

print("=" * 60)
print("GPU Performance Benchmark")
print("=" * 60)

if not torch.cuda.is_available():
    print("❌ CUDA not available! Cannot run benchmark.")
    exit(1)

device = torch.device("cuda")
print(f"\nUsing device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# Warm up GPU
print("Warming up GPU...")
dummy = torch.randn(1000, 1000, device=device)
_ = torch.matmul(dummy, dummy)
torch.cuda.synchronize()
del dummy
torch.cuda.empty_cache()

print("\nRunning matrix multiplication benchmarks...\n")

# Matrix multiplication benchmark
sizes = [1000, 2000, 4000, 8000]
results = []

for size in sizes:
    # Create random matrices on GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up for this size
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark (average of 10 runs)
    times = []
    for _ in range(10):
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    gflops = (2 * size**3) / (avg_time / 1000) / 1e9  # GFLOPS calculation
    
    print(f"Matrix {size:4d}x{size:4d}: {avg_time:7.2f} ms (avg) | {gflops:6.2f} GFLOPS")
    results.append((size, avg_time, gflops))
    
    # Clean up
    del a, b, c
    torch.cuda.empty_cache()

# Memory bandwidth test
print("\nRunning memory bandwidth test...\n")
mem_sizes_mb = [100, 500, 1000, 2000]

for size_mb in mem_sizes_mb:
    size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
    
    # Allocate memory
    src = torch.randn(size_elements, device=device)
    
    # Warm up
    for _ in range(3):
        dst = src.clone()
    torch.cuda.synchronize()
    
    # Benchmark (average of 10 runs)
    times = []
    for _ in range(10):
        start = time.time()
        dst = src.clone()
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    bandwidth = (size_mb / avg_time) / 1024  # GB/s
    
    print(f"Memory Copy {size_mb:4d} MB: {avg_time*1000:7.2f} ms | {bandwidth:6.2f} GB/s")
    
    # Clean up
    del src, dst
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("✅ GPU benchmark complete!")
print("=" * 60)
```

**Run the benchmark:**

```bash
chmod +x gpu_benchmark.py
python3.12 gpu_benchmark.py
```

---

### Quick CUDA Sample Test

**Compile and run CUDA samples:**

```bash
# Navigate to CUDA samples
cd /usr/local/cuda-11.8/samples/1_Utilities/deviceQuery

# Compile
sudo make

# Run
./deviceQuery
```

Expected output should show your Quadro P6000 information.

---

## 🔍 TROUBLESHOOTING

### Issue 1: "CUDA available: False" in PyTorch

**A. Check NVIDIA Driver**

```bash
nvidia-smi
```

If this fails:
- Reinstall NVIDIA driver
- Reboot system

**B. Check CUDA Installation**

```bash
nvcc --version
ls -la /usr/local/cuda-11.8
```

**C. Check PyTorch Build**

```bash
python3.12 -c "import torch; print(torch.__version__)"
```

Should show `2.4.0+cu118` (note the `+cu118`)

If it shows just `2.4.0`:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**D. Check Environment Variables**

```bash
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $PATH
```

Should show CUDA paths. If missing, add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
```

Then:
```bash
source ~/.bashrc
```

---

### Issue 2: "nvidia-smi" not found or fails

**A. Check if driver is loaded**

```bash
lsmod | grep nvidia
```

Should show several nvidia modules. If not:

```bash
sudo modprobe nvidia
```

**B. Reinstall driver**

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

**C. Check for nouveau interference**

```bash
lsmod | grep nouveau
```

If nouveau is loaded, disable it (see Step 3.3 in Pre-Installation).

---

### Issue 3: "nvcc: command not found"

**Solution:**

```bash
# Check if CUDA is installed
ls -la /usr/local/cuda-11.8/bin/nvcc

# If exists, add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

---

### Issue 4: cuDNN library not found

**Error:** `Could not load dynamic library 'libcudnn.so.8'`

**A. Check if cuDNN files exist**

```bash
ls -la /usr/local/cuda-11.8/lib64/libcudnn*
```

**B. Update library cache**

```bash
sudo ldconfig
```

**C. Verify library path**

```bash
echo $LD_LIBRARY_PATH
```

Should include `/usr/local/cuda-11.8/lib64`

**D. Manually add if missing**

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### Issue 5: Permission denied errors

**A. Fix CUDA directory permissions**

```bash
sudo chmod -R 755 /usr/local/cuda-11.8
```

**B. Add user to video group**

```bash
sudo usermod -a -G video $USER
```

Log out and log back in for changes to take effect.

---

### Issue 6: Out of memory errors

**Error:** `CUDA out of memory`

**A. Check GPU memory usage**

```bash
nvidia-smi
```

**B. Clear GPU cache in Python**

```python
import torch
torch.cuda.empty_cache()
```

**C. Monitor memory in real-time**

```bash
watch -n 1 nvidia-smi
```

**D. Reduce batch size in your training script**

**E. Use gradient accumulation**

---

### Issue 7: GCC version incompatibility

**Error:** `unsupported GNU version! gcc versions later than X are not supported!`

**Solution: Install compatible GCC version**

```bash
# Install gcc-11 (compatible with CUDA 11.8)
sudo apt install -y gcc-11 g++-11

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Verify
gcc --version
```

---

### Issue 8: Virtual environment issues

**A. Recreate virtual environment**

```bash
cd ~/t5-finetuning
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**B. Check Python in venv**

```bash
which python
python --version
```

Should point to venv Python.

---

### Issue 9: Slow pip downloads

**A. Use faster mirror**

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

**B. Increase timeout**

```bash
pip install --timeout=1000 <package>
```

---

### Issue 10: Transformers/Model download fails

**For offline model usage:**

```python
# Download model on internet PC
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("./t5-small-local")
tokenizer.save_pretrained("./t5-small-local")
```

Transfer folder to Ubuntu system, then:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("/path/to/t5-small-local")
tokenizer = T5Tokenizer.from_pretrained("/path/to/t5-small-local")
```

---

## 📊 Final Verification Checklist

Run through this checklist:

```bash
# System checks
- [ ] Ubuntu version confirmed (lsb_release -a)
- [ ] GPU detected (lspci | grep -i nvidia)
- [ ] Nouveau disabled (lsmod | grep nouveau should be empty)

# NVIDIA components
- [ ] NVIDIA driver installed (nvidia-smi works)
- [ ] CUDA 11.8 installed (nvcc --version)
- [ ] cuDNN files copied correctly
- [ ] Environment variables set (CUDA_HOME, PATH, LD_LIBRARY_PATH)

# Python environment
- [ ] Python 3.12 installed (python3.12 --version)
- [ ] Virtual environment created and activated
- [ ] pip upgraded (pip --version)

# PyTorch and packages
- [ ] PyTorch with CUDA installed (torch.cuda.is_available() == True)
- [ ] GPU recognized (torch.cuda.get_device_name(0) shows "Quadro P6000")
- [ ] All dependencies installed (transformers, peft, datasets, etc.)
- [ ] Simple GPU operations work

# Verification scripts
- [ ] cuda_test.py runs without errors
- [ ] gpu_benchmark.py shows good performance
- [ ] deviceQuery sample works
```

---

## 🚀 Next Steps: Training T5

**Create a simple test script:** `test_t5.py`

```python
#!/usr/bin/env python3.12
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

print("Testing T5 Model with CUDA")
print("=" * 60)

# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load model
print("\nLoading T5-small model...")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Test inference
print("\nTesting inference...")
input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids, max_length=50)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {decoded}")

# Check model parameters
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Memory usage
if device == "cuda":
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print("\n" + "=" * 60)
print("✅ T5 model test complete!")
print("=" * 60)
```

**Run the test:**

```bash
chmod +x test_t5.py
python3.12 test_t5.py
```

---

## 📝 Useful Commands Reference

### System Monitoring

```bash
# GPU monitoring
nvidia-smi

# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -q

# GPU processes
nvidia-smi pmon

# Check CUDA version
nvcc --version

# Check driver version
cat /proc/driver/nvidia/version

# System resources
htop

# Disk usage
df -h

# Check running processes
ps aux | grep python
```

### Virtual Environment

```bash
# Create venv
python3.12 -m venv venv

# Activate
source venv/bin/activate

# Deactivate
deactivate

# Delete venv
rm -rf venv
```

### Package Management

```bash
# Install package
pip install package_name

# Install specific version
pip install package_name==version

# List installed packages
pip list

# Show package info
pip show package_name

# Freeze requirements
pip freeze > requirements.txt

# Uninstall package
pip uninstall package_name

# Update package
pip install --upgrade package_name
```

### CUDA Management

```bash
# Check CUDA version
nvcc --version

# List CUDA installations
ls -la /usr/local/ | grep cuda

# Check library path
echo $LD_LIBRARY_PATH

# Update library cache
sudo ldconfig

# Find CUDA libraries
ldconfig -p | grep cuda
```

---

## 📂 Directory Structure

**Recommended project structure:**

```
~/t5-finetuning/
├── venv/                    # Virtual environment
├── data/                    # Your datasets
│   ├── train/
│   ├── val/
│   └── test/
├── models/                  # Saved models and checkpoints
│   ├── t5-small-local/      # Local model cache
│   └── checkpoints/
├── scripts/                 # Training and evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/               # Jupyter notebooks (optional)
├── logs/                    # Training logs
├── results/                 # Output results
├── requirements.txt         # Python dependencies
├── cuda_test.py            # CUDA verification script
├── gpu_benchmark.py        # GPU benchmark script
└── test_t5.py              # T5 test script
```

---

## 💡 Performance Tips

### 1. Monitor GPU Usage

```bash
# Create GPU monitoring script
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    nvidia-smi
    sleep 1
done
EOF

chmod +x monitor_gpu.sh
./monitor_gpu.sh
```

### 2. Set CUDA Device

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
```

### 3. Enable cuDNN Benchmarking

```python
import torch
torch.backends.cudnn.benchmark = True  # Faster, but uses more memory
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5. Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 🔐 Security Best Practices

```bash
# Update system regularly
sudo apt update && sudo apt upgrade -y

# Don't run training scripts as root
# Use virtual environment
# Keep backups of important data

# Check for suspicious processes
ps aux | grep python
```

---

## 📊 Installation Summary

| Component | Version | Installation Time | Disk Space |
|-----------|---------|------------------|------------|
| Ubuntu | 20.04/22.04 LTS | - | - |
| Build Tools | Latest | 2-5 min | 500 MB |
| NVIDIA Driver | 550+ | 5-10 min | 1.5 GB |
| CUDA Toolkit | 11.8 | 10-15 min | 5 GB |
| cuDNN | 8.x | 2-3 min | 1.5 GB |
| Python 3.12 | 3.12.x | 3-5 min | 200 MB |
| PyTorch | 2.4.0+cu118 | 5-10 min | 4 GB |
| Dependencies | Latest | 10-20 min | 3-4 GB |
| **Total** | - | **40-70 min** | **~16 GB** |

---

## ⏱️ Total Time Estimate

- System preparation: 10-15 minutes
- Driver installation: 10-15 minutes (including reboot)
- CUDA installation: 15-20 minutes
- Python setup: 10-15 minutes
- PyTorch and packages: 20-30 minutes
- **Total: 65-95 minutes**

---

## 🎯 Common Training Workflows

### Basic Fine-tuning Template

```python
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "t5-small"

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load and preprocess data
dataset = load_dataset("your_dataset")

def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], 
                      max_length=512, 
                      truncation=True, 
                      padding="max_length")
    targets = tokenizer(examples["target_text"], 
                       max_length=512, 
                       truncation=True, 
                       padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    fp16=True,  # Mixed precision training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train
trainer.train()

# Save
trainer.save_model("./final_model")
```

---

## 📞 Support and Resources

**Official Documentation:**
- NVIDIA CUDA: https://docs.nvidia.com/cuda/
- PyTorch: https://pytorch.org/docs/
- Transformers: https://huggingface.co/docs/transformers/
- Ubuntu: https://help.ubuntu.com/

**Community Support:**
- NVIDIA Forums: https://forums.developer.nvidia.com/
- PyTorch Forums: https://discuss.pytorch.org/
- Hugging Face Forums: https://discuss.huggingface.co/
- Stack Overflow: Search for specific errors

**Useful Links:**
- CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive
- PyTorch Installation: https://pytorch.org/get-started/locally/

---

## 🎉 Congratulations!

You now have a fully configured Ubuntu system with:
- ✅ NVIDIA Quadro P6000 driver
- ✅ CUDA 11.8 Toolkit
- ✅ cuDNN libraries
- ✅ Python 3.12 environment
- ✅ PyTorch 2.4.0 with CUDA support
- ✅ All ML/DL dependencies for T5 fine-tuning

Your Quadro P6000 with 24GB VRAM is excellent for training T5 models!

**Happy Training! 🚀**

---

## 📝 Quick Start Commands

```bash
# Activate environment
cd ~/t5-finetuning
source venv/bin/activate

# Check GPU
nvidia-smi

# Verify CUDA
python3.12 -c "import torch; print(torch.cuda.is_available())"

# Start training
python3.12 scripts/train.py

# Monitor GPU (in another terminal)
watch -n 1 nvidia-smi
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Target System:** Ubuntu 20.04/22.04 LTS with NVIDIA Quadro P6000  
**CUDA Version:** 11.8  
**PyTorch Version:** 2.4.0  
**Installation Type:** Online (with internet access)
