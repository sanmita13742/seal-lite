#!/bin/bash
# T4 GPU Setup Script for SEAL
# Run this on Google Colab or any Ubuntu system with T4 GPU

set -e  # Exit on error

echo "=================================="
echo "SEAL T4 GPU Setup"
echo "=================================="

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you running on a GPU instance?"
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install Python dependencies
echo ""
echo "Installing Python packages..."
pip install -q --upgrade pip
pip install -q -r requirements_colab_t4.txt

echo "✓ Dependencies installed"

# Create directory structure
echo ""
echo "Creating directories..."
mkdir -p loras_t4/{ttt,self_edit}
mkdir -p models_t4
mkdir -p results_t4
mkdir -p few-shot/data

echo "✓ Directories created"

# Download ARC data if not exists
echo ""
echo "Checking ARC dataset..."
if [ ! -f "few-shot/data/arc-agi_training_challenges.json" ]; then
    echo "Downloading ARC dataset..."
    wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/training/challenges.json \
        -O few-shot/data/arc-agi_training_challenges.json
    wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/training/solutions.json \
        -O few-shot/data/arc-agi_training_solutions.json
    wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/evaluation/challenges.json \
        -O few-shot/data/arc-agi_evaluation_challenges.json
    wget -q https://github.com/fchollet/ARC-AGI/raw/master/data/evaluation/solutions.json \
        -O few-shot/data/arc-agi_evaluation_solutions.json
    echo "✓ ARC dataset downloaded"
else
    echo "✓ ARC dataset already exists"
fi

# Test import
echo ""
echo "Testing imports..."
python3 << END
import torch
import transformers
import peft
from transformers import AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
END

echo "✓ All imports successful"

# Create quick test script
echo ""
echo "Creating quick test script..."
cat > test_t4.py << 'END'
"""Quick test of T4 setup"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing T4 setup...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print("✓ Model loaded successfully!")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Test generation
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nTest generation:")
    print(f"Input: {prompt}")
    print(f"Output: {result}")
    print("\n✓ Setup test passed!")
else:
    print("ERROR: CUDA not available!")
END

echo "✓ Test script created"

# Run test
echo ""
echo "Running setup test..."
python3 test_t4.py

echo ""
echo "=================================="
echo "Setup Complete! ✓"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run quick TTT test:"
echo "   python few-shot/ttt_t4.py --num_tasks=2"
echo ""
echo "2. Run full self-edit RL:"
echo "   python few-shot/self_edit_t4.py --experiment_name=test --n_tasks=5 --n_self_edits_per_task=3"
echo ""
echo "3. Check COLAB_T4_COMPLETE_GUIDE.md for full instructions"
echo ""
echo "=================================="
