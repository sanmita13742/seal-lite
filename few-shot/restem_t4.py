"""
T4-Optimized RestEM (Reinforcement Learning) for SEAL
Implements the full RestEM behavioral cloning pipeline on T4 GPU

Features:
- Trains on correct predictions from self-edit evaluation
- LoRA fine-tuning with gradient accumulation
- Memory-efficient for T4
- Merges LoRA into base model
"""

import os
import json
import torch
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def create_dataset(training_data, tokenizer):
    """Create dataset with proper labels for causal LM."""
    dataset = Dataset.from_dict({"text": training_data})
    
    def tokenize_function(examples):
        # Tokenize the full text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=8192,
            padding="longest",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids initially)
        labels = tokenized["input_ids"].clone()
        
        # For each example, mask the prompt part (only train on response)
        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Find assistant header tokens (128007, 271)
            special_indices = []
            for j in range(len(input_ids) - 1):
                if input_ids[j] == 128007 and input_ids[j + 1] == 271:
                    special_indices.append(j + 1)
            
            # Use second-to-last occurrence (the final assistant response)
            if len(special_indices) >= 2:
                mask_until = special_indices[-2]
                labels[i, :mask_until] = -100  # Mask prompt
            else:
                # Fallback: mask first 80% as prompt
                mask_until = int(len(input_ids) * 0.8)
                labels[i, :mask_until] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description='T4-Optimized RestEM Training')
    
    # Input arguments
    parser.add_argument(
        '--configs_and_indices',
        type=str,
        required=True,
        help='Path to configs_and_indices.json from self-edit'
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to eval_results.json from evaluation'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Base model or previous RL checkpoint'
    )
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=16)
    
    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=8)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for merged model'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T4-Optimized RestEM Training for SEAL")
    print("=" * 80)
    print(f"Base model: {args.model_name}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)
    
    # Load configs and results
    print("\nLoading data...")
    with open(args.configs_and_indices, 'r') as f:
        configs_and_indices = json.load(f)
    
    with open(args.results, 'r') as f:
        eval_results = json.load(f)
        results = eval_results.get('results', {})
    
    print(f"Loaded configs for {len(configs_and_indices)} tasks")
    print(f"Loaded results for {len(results)} tasks")
    
    # Extract correct prompts and responses
    print("\nExtracting correct examples...")
    correct_prompts_responses = []
    
    for task_name, task_result in results.items():
        if not task_result.get('correct', False):
            continue
        
        # Get task ID
        task_id = task_name.split("-")[0] if "-" in task_name else task_name
        
        if task_id not in configs_and_indices:
            continue
        
        # Get all configs for this task
        task_configs = configs_and_indices[task_id]
        
        for config_idx_str, config_data in task_configs.items():
            if 'prompt' in config_data and 'response' in config_data:
                correct_prompts_responses.append({
                    'prompt': config_data['prompt'],
                    'response': config_data['response'],
                    'task_id': task_id,
                    'config_idx': config_idx_str
                })
    
    print(f"Found {len(correct_prompts_responses)} correct examples")
    
    if len(correct_prompts_responses) == 0:
        print("ERROR: No correct examples found!")
        return
    
    # Create training data
    training_data = []
    for item in correct_prompts_responses:
        # Format as complete sequence with end token
        full_text = item['prompt'] + item['response'] + '<|eot_id|>'
        training_data.append(full_text)
    
    print(f"Created {len(training_data)} training examples")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization for memory efficiency
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    print("\nCreating dataset...")
    train_dataset = create_dataset(training_data, tokenizer)
    print(f"Dataset size: {len(train_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./tmp_restem_training",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.train()
    print("=" * 80)
    print("Training complete!")
    
    # Merge LoRA weights into base model
    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save to temporary directory first
    temp_dir = "./temp_merged_model"
    model.save_pretrained(
        temp_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(temp_dir)
    
    # Move to final location
    for file in os.listdir(temp_dir):
        src = os.path.join(temp_dir, file)
        dst = os.path.join(args.output_dir, file)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    
    # Cleanup temp directories
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree("./tmp_restem_training", ignore_errors=True)
    
    print("\n" + "=" * 80)
    print("RestEM Training Complete!")
    print("=" * 80)
    print(f"Trained on {len(correct_prompts_responses)} correct examples")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use this model as the base for next RL iteration")
    print("2. Run self-edit again with this model")
    print("3. Evaluate and repeat RestEM")
    print("=" * 80)


if __name__ == "__main__":
    main()
