# Running SEAL on Google Colab T4: Complete Setup Guide

## 🎯 What This Guide Covers

How to run SEAL experiments on Google Colab with T4 GPU (free or Pro tier) using the 1B Llama model.

---

## 📋 Prerequisites

1. Google account
2. GitHub account (optional, but recommended)
3. Basic Python knowledge
4. Patience (T4 is slower than A100)

---

## 🚀 Quick Start (Copy-Paste Ready)

### Step 1: Create New Colab Notebook

1. Go to: https://colab.research.google.com
2. Click: `File` → `New notebook`
3. Click: `Runtime` → `Change runtime type` → Select `T4 GPU` → `Save`

### Step 2: Setup Environment

```python
# Cell 1: Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_properties(0).name}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

Expected output:
```
GPU Available: True
GPU Name: Tesla T4
GPU Memory: 15.0GB
```

### Step 3: Clone Repository

```python
# Cell 2: Clone SEAL repo
!git clone https://github.com/Continual-Intelligence/SEAL.git
%cd SEAL
!ls
```

### Step 4: Install Dependencies

```python
# Cell 3: Install required packages
!pip install -q torch transformers accelerate peft datasets vllm
!pip install -q numpy scipy matplotlib tqdm rich

# Verify vLLM installation
import vllm
print(f"vLLM version: {vllm.__version__}")
```

### Step 5: Download Model

```python
# Cell 4: Pre-download the model (saves time later)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
print(f"Downloading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

print("Model downloaded successfully!")
print(f"Model size: ~{model.get_memory_footprint() / 1e9:.2f}GB")

# Free memory
del model, tokenizer
torch.cuda.empty_cache()
```

---

## 🧪 Experiment 1: Baseline Evaluation

### Run Baseline ARC Evaluation (20 tasks, ~20 minutes)

```python
# Cell 5: Baseline evaluation with vLLM
!python few-shot/eval-self-edits-baseline.py \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
    --solution_file=few-shot/data/arc-agi_evaluation_solutions.json \
    --num_examples=20 \
    --temperature=0.0 \
    --n_sample=1 \
    --experiment_folder=/content/experiments/baseline \
    --new_format
```

Expected output:
```
Processing 20 tasks...
[1/20] Processing task 00576224...
...
Accuracy: 12.5% (2/20 correct)
```

### View Results

```python
# Cell 6: Check results
import json

with open('/content/experiments/baseline/results.json') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))
```

---

## 🧪 Experiment 2: LoRA Test-Time Training

### Train LoRA on 5 Tasks (~15-20 minutes)

```python
# Cell 7: Generate self-edits with LoRA training
!python few-shot/self-edit.py \
    --experiment_name=t4_test \
    --challenge_file=few-shot/data/arc-agi_training_challenges.json \
    --solution_file=few-shot/data/arc-agi_training_solutions.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --n_tasks=5 \
    --n_self_edits_per_task=3 \
    --lora_rank=16 \
    --lora_alpha=16 \
    --num_train_epochs=2 \
    --batch_size=1
```

**Note:** Reduced parameters for T4:
- `lora_rank=16` (instead of 128)
- `n_self_edits_per_task=3` (instead of 15)
- `num_train_epochs=2` (instead of 8)
- `batch_size=1` (instead of 5)

### Evaluate with LoRA

```python
# Cell 8: Evaluate with trained LoRAs
!python few-shot/eval-self-edits.py \
    --experiment_folder=/content/experiments/lora_eval \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=/content/SEAL/loras/self-edit/t4_test \
    --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
    --solution_file=few-shot/data/arc-agi_evaluation_solutions.json \
    --temperature=0.0 \
    --n_sample=1 \
    --max_lora_rank=16 \
    --include_n=1 \
    --new_format \
    --num_examples=10
```

---

## 🛠️ T4-Optimized Configuration

### Memory-Efficient Settings

```python
# Cell 9: Create T4-optimized config file
config_t4 = {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "max_model_len": 4096,  # Reduced from 8192
    "gpu_memory_utilization": 0.85,  # Leave headroom
    "lora_rank": 16,  # Reduced from 64-128
    "lora_alpha": 16,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "max_lora_rank": 16,
}

import json
with open('/content/config_t4.json', 'w') as f:
    json.dump(config_t4, f, indent=2)

print("T4 config saved to /content/config_t4.json")
```

### Monitor GPU Usage

```python
# Cell 10: GPU monitoring function
import torch

def print_gpu_stats():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"GPU Memory:")
    print(f"  Allocated: {allocated:.2f}GB / {total:.1f}GB")
    print(f"  Reserved:  {reserved:.2f}GB / {total:.1f}GB")
    print(f"  Free:      {total - reserved:.2f}GB")

# Call anytime
print_gpu_stats()
```

---

## 🔧 Common T4 Issues & Fixes

### Issue 1: Out of Memory

```python
# Solution: Reduce memory usage
!python few-shot/eval-self-edits-baseline.py \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --max_tokens=2048 \  # Reduced
    --num_examples=10 \  # Smaller batch
    --quantization=bitsandbytes  # Use 4-bit quantization
```

### Issue 2: Colab Disconnects

```python
# Solution: Save checkpoints frequently
# Add to your scripts:
import os

def save_checkpoint(data, name):
    checkpoint_dir = '/content/drive/MyDrive/SEAL_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    import json
    with open(f'{checkpoint_dir}/{name}.json', 'w') as f:
        json.dump(data, f)
    
    print(f"Checkpoint saved: {name}")

# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')
```

### Issue 3: vLLM Errors

```python
# Solution: Use HuggingFace backend instead
# Modify script to use engine_no_lora.py instead of engine_vllm.py

# Or install specific vLLM version
!pip install vllm==0.6.0
```

---

## 📊 Expected Performance on T4

### Inference Speed
- **Baseline:** ~45-60 seconds per task
- **With LoRA:** ~60-90 seconds per task
- **vs Paper (A100):** 5-10 seconds per task

### Accuracy
- **Baseline:** 10-15% on ARC
- **With LoRA (small scale):** 15-20%
- **vs Paper:** 35-40%

### Limitations
- Can't run 7B models (OOM)
- Can't use full LoRA rank (use 8-16, not 128)
- Can't process 400 tasks in one go
- Colab Free: 12-hour session limit

---

## 💾 Save Results to Google Drive

```python
# Cell 11: Mount Drive and save results
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!mkdir -p /content/drive/MyDrive/SEAL_results
!cp -r /content/experiments/* /content/drive/MyDrive/SEAL_results/

print("Results saved to Google Drive!")
```

---

## 🎯 Recommended Workflows

### Workflow 1: Quick Test (30 min)
```python
# 1. Setup (5 min)
# Cells 1-4

# 2. Baseline eval (20 min)
# Cell 5 with num_examples=10

# 3. Check results (5 min)
# Cell 6
```

### Workflow 2: LoRA Training (1.5 hours)
```python
# 1. Setup (5 min)
# 2. Train LoRA (45 min)
# Cell 7 with n_tasks=5
# 3. Evaluate (30 min)
# Cell 8 with num_examples=10
# 4. Save results (10 min)
# Cell 11
```

### Workflow 3: Full Day Run (Colab Pro)
```python
# Morning: Setup + baseline (50 tasks)
# Afternoon: LoRA training (20 tasks)
# Evening: Evaluation + analysis
# Total: 6-8 hours
```

---

## 🔄 Upgrading to Colab Pro

### When to Upgrade

Upgrade if you need:
- ✅ Longer sessions (24h vs 12h)
- ✅ More RAM (25GB vs 12GB)
- ✅ Priority GPU access
- ✅ Background execution

### Cost
- **Colab Pro:** $10/month
- **Colab Pro+:** $50/month (includes A100 access)

### Getting A100 on Colab Pro+

```python
# In Colab Pro+ notebook
# Runtime → Change runtime type → Select "A100 GPU"

# Verify
import torch
print(torch.cuda.get_device_properties(0).name)
# Should show: "NVIDIA A100-SXM4-40GB"
```

---

## 📈 Scaling Strategy

### Phase 1: T4 Free (This Guide)
- ✅ Learn concepts
- ✅ Test 10-20 tasks
- ✅ Understand workflow
- Cost: $0

### Phase 2: T4 Pro
- ✅ 50-100 tasks
- ✅ Full LoRA experiments
- ✅ Longer sessions
- Cost: $10/month

### Phase 3: A100 Pro+
- ✅ 100-400 tasks
- ✅ Near paper results
- ✅ 7B models possible
- Cost: $50/month

### Phase 4: Dedicated GPU
- ✅ Lambda Labs / RunPod
- ✅ Full replication
- Cost: $1-2/hour

---

## 🎓 Tips & Best Practices

### 1. Start Small
```python
# Always test with num_examples=2 first
# Then scale up to 10, 20, 50...
```

### 2. Save Frequently
```python
# Mount Drive at start
# Save after each major step
```

### 3. Monitor Memory
```python
# Call print_gpu_stats() regularly
# If >90% used, reduce batch sizes
```

### 4. Use Checkpointing
```python
# For long runs, save progress every N tasks
# Can resume if disconnected
```

### 5. Optimize Settings
```python
# Use config_t4.json
# Reduce unnecessary parameters
# Use gradient accumulation for larger effective batch sizes
```

---

## 🐛 Debugging Guide

### Problem: "CUDA out of memory"
```python
# Solutions:
torch.cuda.empty_cache()  # Clear cache
# Use smaller model
# Reduce max_model_len
# Use quantization
# Reduce batch_size to 1
```

### Problem: "Colab disconnected"
```python
# Solutions:
# 1. Keep browser tab active
# 2. Use Colab Pro for longer sessions
# 3. Add checkpointing
# 4. Run overnight
```

### Problem: "vLLM import error"
```python
# Solutions:
!pip uninstall vllm -y
!pip install vllm==0.6.0  # Specific version
# Or use HF backend instead
```

---

## 📚 Additional Resources

### Documentation
- This guide: `COLAB_T4_SETUP.md`
- Feasibility: `COLAB_T4_FEASIBILITY.md`
- Local Ollama: `README_OLLAMA.md`
- Original paper: https://arxiv.org/abs/2506.10943

### Colab Notebooks
```
You can save this as a Colab notebook:
File → Save a copy in Drive
Share the link with others
```

### Community
- GitHub Issues: https://github.com/Continual-Intelligence/SEAL/issues
- Paper website: https://jyopari.github.io/posts/seal

---

## ✅ Checklist

Before starting, verify:
- [ ] Colab notebook created
- [ ] GPU set to T4
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Model downloaded
- [ ] Google Drive mounted (optional)
- [ ] Started with small num_examples

---

## 🎉 Success!

If you made it here and ran experiments successfully:

1. ✅ You understand SEAL concepts
2. ✅ You can run LoRA training
3. ✅ You can evaluate on ARC
4. ✅ You're ready to scale up

**Next steps:**
- Run more tasks (20 → 50 → 100)
- Experiment with different prompts
- Try different LoRA configurations
- Upgrade to Colab Pro if needed
- Consider A100 for full replication

**Have fun experimenting!** 🚀
