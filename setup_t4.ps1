# Windows Setup for T4 GPU (Colab/Cloud)
# Run this in PowerShell on Windows VM with T4 GPU

Write-Host "==================================" -ForegroundColor Green
Write-Host "SEAL T4 GPU Setup (Windows)" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Check if NVIDIA GPU exists
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "ERROR: nvidia-smi not found. Are you running on a GPU instance?" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ GPU detected:" -ForegroundColor Green
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Python found:" -ForegroundColor Green
python --version

# Install dependencies
Write-Host "`nInstalling Python packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
python -m pip install -r requirements_colab_t4.txt -q

Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "loras_t4\ttt" | Out-Null
New-Item -ItemType Directory -Force -Path "loras_t4\self_edit" | Out-Null
New-Item -ItemType Directory -Force -Path "models_t4" | Out-Null
New-Item -ItemType Directory -Force -Path "results_t4" | Out-Null
New-Item -ItemType Directory -Force -Path "few-shot\data" | Out-Null

Write-Host "✓ Directories created" -ForegroundColor Green

# Check ARC data
Write-Host "`nChecking ARC dataset..." -ForegroundColor Yellow
$arcFile = "few-shot\data\arc-agi_training_challenges.json"
if (-not (Test-Path $arcFile)) {
    Write-Host "Downloading ARC dataset..." -ForegroundColor Yellow
    $urls = @{
        "few-shot\data\arc-agi_training_challenges.json" = "https://github.com/fchollet/ARC-AGI/raw/master/data/training/challenges.json"
        "few-shot\data\arc-agi_training_solutions.json" = "https://github.com/fchollet/ARC-AGI/raw/master/data/training/solutions.json"
        "few-shot\data\arc-agi_evaluation_challenges.json" = "https://github.com/fchollet/ARC-AGI/raw/master/data/evaluation/challenges.json"
        "few-shot\data\arc-agi_evaluation_solutions.json" = "https://github.com/fchollet/ARC-AGI/raw/master/data/evaluation/solutions.json"
    }
    
    foreach ($file in $urls.Keys) {
        Invoke-WebRequest -Uri $urls[$file] -OutFile $file -UseBasicParsing
    }
    Write-Host "✓ ARC dataset downloaded" -ForegroundColor Green
} else {
    Write-Host "✓ ARC dataset already exists" -ForegroundColor Green
}

# Test imports
Write-Host "`nTesting imports..." -ForegroundColor Yellow
python -c @"
import torch
import transformers
import peft

print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PEFT version: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"@

Write-Host "✓ All imports successful" -ForegroundColor Green

Write-Host "`n==================================" -ForegroundColor Green
Write-Host "Setup Complete! ✓" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "1. Run quick TTT test:"
Write-Host "   python few-shot\ttt_t4.py --num_tasks=2" -ForegroundColor Cyan
Write-Host "`n2. Run full self-edit RL:"
Write-Host "   python few-shot\self_edit_t4.py --experiment_name=test --n_tasks=5" -ForegroundColor Cyan
Write-Host "`n3. Check COLAB_T4_COMPLETE_GUIDE.md for full instructions" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Green
