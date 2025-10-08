# SEAL T4 - Command Cheatsheet 📋

Quick reference for all SEAL T4 GPU commands.

---

## 🚀 Setup

```bash
# Install
pip install -q -r requirements_colab_t4.txt

# Setup script
bash setup_t4.sh  # Linux/Mac
powershell -File setup_t4.ps1  # Windows
```

---

## 🧪 Test-Time Training

```bash
# Quick test
python few-shot/ttt_t4.py --num_tasks=2

# Full run
python few-shot/ttt_t4.py \
    --num_tasks=10 \
    --lora_rank=128 \
    --num_train_epochs=3
```

---

## 🔄 Self-Edit RL

```bash
# Iteration 1
python few-shot/self_edit_t4.py \
    --experiment_name=iter1 \
    --n_tasks=12 \
    --n_self_edits_per_task=5 \
    --challenge_file=few-shot/data/arc-agi_training_challenges.json \
    --solution_file=few-shot/data/arc-agi_training_solutions.json

# Iteration 2 (use RL model)
python few-shot/self_edit_t4.py \
    --experiment_name=iter2 \
    --model_name=models_t4/rl_iteration_1 \
    --n_tasks=12 \
    --n_self_edits_per_task=5
```

---

## 📊 Evaluation

```bash
# Basic eval
python few-shot/eval_self_edits_t4.py \
    --experiment_folder=results_t4/eval_iter1 \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=loras_t4/self_edit/iter1 \
    --data_file=few-shot/data/arc-agi_training_challenges.json \
    --solution_file=few-shot/data/arc-agi_training_solutions.json

# Check results
python -c "import json; r=json.load(open('results_t4/eval_iter1/eval_results.json')); print(f'Accuracy: {r[\"accuracy\"]:.2%}')"
```

---

## 🎓 RestEM

```bash
# Train on correct predictions
python few-shot/restem_t4.py \
    --configs_and_indices=loras_t4/self_edit/iter1/final_configs_and_indices.json \
    --results=results_t4/eval_iter1/eval_results.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --output_dir=models_t4/rl_iteration_1 \
    --num_train_epochs=8
```

---

## 🔁 Full Pipeline

```bash
# Complete RL iteration
python few-shot/self_edit_t4.py --experiment_name=iter1 --n_tasks=12
python few-shot/eval_self_edits_t4.py --experiment_folder=results_t4/iter1
python few-shot/restem_t4.py --output_dir=models_t4/rl_iter1
```

---

## 🔍 Monitoring

```bash
# GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Python memory
python -c "import torch; print(f'{torch.cuda.memory_allocated()/1024**3:.2f} GB')"

# List adapters
find loras_t4/ -name "adapter_config.json"
```

---

## 💾 Save/Load

```python
# Save to Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')
!cp -r models_t4/ /content/drive/MyDrive/SEAL/

# Zip & download
!zip -r results.zip loras_t4/ results_t4/ models_t4/
from google.colab import files
files.download('results.zip')
```

---

## ⚙️ Memory Presets

```bash
# Low VRAM
--batch_size=1 --gradient_accumulation_steps=8 --lora_rank=64 --use_8bit

# Balanced
--batch_size=1 --gradient_accumulation_steps=2 --lora_rank=128 --use_8bit

# Fast
--batch_size=2 --gradient_accumulation_steps=1 --lora_rank=128
```

---

## 📈 Analysis

```python
# Compare iterations
import json
for i in [1,2,3]:
    r = json.load(open(f'results_t4/eval_iter{i}/eval_results.json'))
    print(f"Iter {i}: {r['accuracy']:.2%}")
```

---

See [COLAB_T4_COMPLETE_GUIDE.md](COLAB_T4_COMPLETE_GUIDE.md) for full documentation.
