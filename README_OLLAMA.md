# Running SEAL with Local Ollama (Laptop Edition)

This guide shows you how to run SEAL experiments using your local Llama 3.2 model via Ollama, without needing OpenAI API, SLURM clusters, or multiple GPUs.

## ⚠️ Important Limitations

This Ollama adaptation provides:
- ✅ **Conceptual understanding** of SEAL's workflow
- ✅ **Basic evaluation** capabilities
- ✅ **No external API costs**

But with limitations:
- ❌ **No LoRA adaptation** (core SEAL technique removed)
- ❌ **Much slower** (CPU inference, hours instead of minutes)
- ❌ **Lower accuracy** (no test-time training)
- ❌ **Small scale only** (5-10 tasks max recommended)

## 🚀 Quick Start

### 1. Prerequisites

Make sure Ollama is installed and running:

```powershell
# Check if Ollama is running
ollama list

# If not installed, download from: https://ollama.ai
# Then pull your model
ollama pull llama3.2:latest

# Start Ollama server (if not running)
ollama serve
```

### 2. Install Dependencies

```powershell
# Install base requirements
pip install -r requirements.txt

# Install additional Ollama support
pip install requests
```

### 3. Run Simple Check

```powershell
python run_ollama_simple.py --mode=check
```

This checks if Ollama is running and your model is available.

### 4. Run ARC Evaluation (Baseline)

```powershell
# Evaluate on 3 ARC tasks (start small!)
python run_ollama_simple.py --mode=arc --num_examples=3

# Or run directly:
python few-shot/eval_ollama_simple.py \
    --model_name=llama3.2:latest \
    --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
    --solution_file=few-shot/data/arc-agi_evaluation_solutions.json \
    --num_examples=5 \
    --temperature=0.0 \
    --experiment_folder=experiments/ollama_test
```

**Expected time:** 5-15 minutes per task on CPU

### 5. Test Knowledge Incorporation

```powershell
python run_ollama_simple.py --mode=knowledge
```

## 📁 New Files Created

### Core Inference Engine
- **`few-shot/inference/engine_ollama.py`** - Replaces vLLM with Ollama API
  - Lightweight engine for local inference
  - No LoRA support (technical limitation)
  - Works on CPU/laptop GPU

### Grading System
- **`few-shot/utils/ollama_grader.py`** - Replaces OpenAI GPT-4 grading
  - Uses local Llama for yes/no grading
  - Less accurate than GPT-4 but free

### Utilities
- **`general-knowledge/src/utils_ollama.py`** - Ollama utilities for knowledge tasks
  - Question answering
  - Answer grading
  - Training data preparation

### Evaluation Scripts
- **`few-shot/eval_ollama_simple.py`** - Simplified ARC evaluation
  - No LoRA, no vLLM dependencies
  - Works on single GPU or CPU
  - Baseline evaluation only

### Launcher
- **`run_ollama_simple.py`** - Main entry point
  - Checks Ollama connectivity
  - Runs different experiment modes
  - Provides guidance

## 🎯 What You Can Do

### 1. ARC Baseline Evaluation
Test the base model on ARC tasks without any adaptation:

```powershell
python few-shot/eval_ollama_simple.py \
    --model_name=llama3.2:latest \
    --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
    --num_examples=5 \
    --temperature=0.0
```

**Expected accuracy:** 5-15% (random guessing is ~0%)

### 2. Knowledge Q&A Testing
Test basic question answering:

```python
from general_knowledge.src.utils_ollama import init_ollama_client

client = init_ollama_client("llama3.2:latest")
response = client.generate("What is the capital of France?")
print(response)
```

### 3. Custom Prompting Experiments
Modify prompts to test different approaches:

```python
# Edit few-shot/arclib/messagers.py to try different prompt formats
# Then re-run evaluation
```

## 🔧 Configuration

### Change Ollama Model

```powershell
# Use different model
python run_ollama_simple.py --model=llama3.1:8b --mode=arc
```

### Adjust Generation Parameters

Edit `few-shot/eval_ollama_simple.py`:

```python
parser.add_argument("--temperature", type=float, default=0.0)  # Change here
parser.add_argument("--max_tokens", type=int, default=4096)    # Change here
parser.add_argument("--n_sample", type=int, default=1)         # Samples per task
```

### Change Ollama URL

If running Ollama on different port/host:

```powershell
python few-shot/eval_ollama_simple.py \
    --ollama_url=http://192.168.1.100:11434 \
    --model_name=llama3.2:latest \
    --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
    --num_examples=3
```

## 📊 Expected Performance

### On Laptop (CPU/Integrated GPU)
- **Inference speed:** ~2-10 minutes per task
- **Memory usage:** ~4-8GB RAM
- **ARC accuracy:** 5-15% (baseline, no adaptation)
- **Practical limit:** 5-10 tasks per run

### With Dedicated GPU (RTX 3060+)
- **Inference speed:** ~30-60 seconds per task
- **Memory usage:** ~8-12GB VRAM
- **ARC accuracy:** 10-20% (baseline)
- **Practical limit:** 20-50 tasks per run

### Original Paper (2x A100)
- **Inference speed:** ~5-10 seconds per task
- **Memory usage:** ~40GB VRAM
- **ARC accuracy:** 30-40% (with LoRA adaptation)
- **Scale:** 100+ tasks

## 🚫 What's NOT Supported

### Test-Time Training (LoRA)
The core SEAL technique requires:
- Multiple GPUs
- vLLM server
- LoRA training infrastructure
- High-speed inference

**Workaround:** None for laptop. Use cloud GPU for real TTT.

### Multi-Document CPT
Continual Pre-Training across documents requires:
- ZMQ server coordination
- Parallel training/inference
- Large memory for multiple LoRAs

**Workaround:** Test on single documents only.

### Full RL Training (ReST-EM)
Requires:
- Large-scale data generation
- Multiple training iterations
- GPU cluster

**Workaround:** Use pre-generated data from paper.

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
```powershell
# Make sure Ollama is running
ollama serve

# Check in browser: http://localhost:11434
```

### "Model not found"
```powershell
# Pull the model first
ollama pull llama3.2:latest
ollama list  # Verify it's there
```

### "Out of memory"
- Reduce `--num_examples` to 1-2
- Reduce `--max_tokens` to 2048
- Close other applications
- Use smaller model: `ollama pull llama3.2:1b`

### "Takes too long"
- This is expected on CPU!
- Start with just 1-2 examples
- Consider using cloud GPU for larger runs
- Expected: 5-15 min per task on CPU

### "Low accuracy"
- Expected! Baseline without LoRA is ~10-15%
- Original paper: ~35-40% with full SEAL
- You're running a stripped-down version

## 📈 Scaling Up

When you're ready for full SEAL capabilities:

### Option 1: Google Colab Pro
```python
# In Colab notebook with A100
!pip install vllm peft transformers
!git clone https://github.com/Continual-Intelligence/SEAL
# Run original scripts with vLLM
```

### Option 2: Cloud GPU
- **Lambda Labs:** ~$1.10/hour for A100
- **RunPod:** ~$0.79/hour for RTX 4090
- **AWS EC2:** g5.12xlarge instance

### Option 3: Local GPU Server
- Minimum: 1x RTX 4090 (24GB)
- Recommended: 2x RTX 4090 or 1x A100

## 🔗 Related Files

### Original Scripts (Require vLLM/Multi-GPU)
- `few-shot/eval-self-edits.py` - Full LoRA evaluation
- `few-shot/self-edit.py` - Generate self-edits
- `few-shot/ttt.py` - Test-time training
- `general-knowledge/scripts/*.sh` - SLURM scripts

### Modified for Ollama
- `few-shot/eval_ollama_simple.py` - Simplified evaluation
- `few-shot/inference/engine_ollama.py` - Ollama inference
- `general-knowledge/src/utils_ollama.py` - Ollama utilities

## 💡 Tips

1. **Start small:** Test with 1-2 examples first
2. **Be patient:** CPU inference is slow (minutes per task)
3. **Monitor resources:** Check Task Manager for memory/CPU usage
4. **Save often:** Script saves results after each task
5. **Iterate prompts:** Experiment with different prompt formats
6. **Use cloud for scale:** When ready, move to GPU cloud

## 📚 Learning Path

1. ✅ **Week 1:** Run baseline with 5 tasks, understand workflow
2. ✅ **Week 2:** Experiment with prompts, analyze outputs
3. ✅ **Week 3:** Try different Ollama models
4. ⚠️ **Week 4:** Move to cloud GPU for full SEAL

## ❓ FAQ

**Q: Can I run full SEAL on my laptop?**
A: No. You can run simplified baseline evaluation only.

**Q: Will results match the paper?**
A: No. Paper uses LoRA adaptation (~35% accuracy), you'll get ~10-15% baseline.

**Q: Is this useful for learning?**
A: Yes! Understand the concepts, workflows, and data formats.

**Q: How much does cloud GPU cost?**
A: ~$5-10 for a few hours to run full experiments.

**Q: Can I use other Ollama models?**
A: Yes! Try `llama3.1:8b`, `mistral`, `qwen2.5:7b`, etc.

---

For full replication, see original `README.md` and use cloud/cluster GPUs.
