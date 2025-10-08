"""
T4-Optimized Test-Time Training (TTT) for SEAL
Implements complete LoRA-based TTT on T4 GPU with 16GB VRAM

Features:
- LoRA fine-tuning at test time
- Memory-efficient training
- Works on single T4 GPU
- No vLLM dependency
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import urllib.request

from transformers import AutoTokenizer
from peft import LoraConfig

from arclib.arc import read_tasks_from_single_file
from arclib.representers import (
    TextTaskRepresenter,
    TextExampleRepresenter,
    PythonListGridRepresenter,
)
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.update_model import TTT
from arclib.augmenters import *


def download_arc_data(data_dir: str = "data"):
    """Download ARC dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    
    files = {
        "arc-agi_training_challenges.json": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/challenges.json",
        "arc-agi_training_solutions.json": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/solutions.json",
        "arc-agi_evaluation_challenges.json": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/challenges.json",
        "arc-agi_evaluation_solutions.json": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/solutions.json",
    }
    
    for filename, url in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
        else:
            print(f"✓ {filename} already exists")


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True
) -> List:
    """Get list of augmenters based on configuration."""
    
    basic_augmenters = (
        [
            Rotate(90), Rotate(270), Rotate(180),
            Flip(0), Flip(1),
            Reflect(0, reverse=True), Reflect(1, reverse=True),
            Reflect(0, reverse=False), Reflect(1, reverse=False),
            RandomTranslateXY(), Transpose(),
        ]
        if include_basic else []
    )
    
    size_augmenters = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size else []
    )
    
    chain_augmenters = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain else []
    )
    
    repeat_augmenters = (
        [Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)]
        if include_repeat else []
    )
    
    return (
        basic_augmenters +
        size_augmenters +
        chain_augmenters +
        repeat_augmenters
    )


def get_test_time_train_data(
    task, augmenters, n: int = 1, permute_n: int = 1, seed: int = 0
):
    """Generate test-time training data with augmentation and permutations."""
    from arclib.augmenters import IdentityAugmenter
    import numpy as np
    
    rng = np.random.RandomState(seed)
    train_data = []
    augmenters_with_identity = [IdentityAugmenter()] + augmenters
    
    for augmenter in augmenters_with_identity:
        # CRITICAL: Pass rng to augmenter (required for RandomTranslateXY and others)
        augmented_task = augmenter.apply_to_task(task, rng=rng, to_input=True, to_output=True)
        
        # Leave out n training examples for validation
        if len(augmented_task.train_examples) > n:
            examples_to_use = augmented_task.train_examples[:-n]
            
            # Apply permutations
            for _ in range(permute_n):
                permuted_indices = rng.permutation(len(examples_to_use))
                permuted_examples = [examples_to_use[i] for i in permuted_indices]
                
                permuted_task = type(augmented_task)(
                    train_examples=permuted_examples,
                    test_example=augmented_task.test_example,
                    name=augmented_task.name
                )
                train_data.append(permuted_task)
    
    return train_data


def format_and_filter(formatter, tokenizer, task, max_tokens: int = 8192):
    """Format task for training and filter by token count."""
    # Encode task using the message representer
    formatted = formatter.encode(task)
    data = {"input": formatted[0], "output": formatted[1]}
    
    # Apply chat template
    task_text = tokenizer.apply_chat_template(
        formatted[0] + [formatted[1]],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize to get token count
    tokens = tokenizer(task_text, return_tensors="pt")
    total_tokens = tokens['input_ids'].shape[1]
    
    data["total_tokens"] = total_tokens
    data["full_text"] = task_text
    
    return data if total_tokens < max_tokens else None


def get_formatted_data(
    task, augmenters, formatter, tokenizer, leave_n: int = 1,
    permute_n: int = 1, seed: int = 0, max_tokens: int = 8192
):
    """Get formatted training data for a task."""
    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )
    
    formatted_data = []
    for task_item in train_data:
        formatted = format_and_filter(formatter, tokenizer, task_item, max_tokens=max_tokens)
        if formatted is not None:
            formatted_data.append(formatted)
    
    return formatted_data


def process_task(
    task,
    augmenters,
    formatter,
    tokenizer,
    leave_n: List[int] = [1, 2],
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0
):
    """Process task to create training data (matches original ttt.py logic)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    
    train = []
    # Generate training data for each n in leave_n
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer,
            leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)
    
    # Shuffle and limit the total number of examples if needed
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]
    
    return train


def main():
    parser = argparse.ArgumentParser(description='T4-Optimized TTT for SEAL')
    
    # Data arguments
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/arc-agi_training_challenges.json',
        help='Path to ARC challenges file'
    )
    parser.add_argument(
        '--solution_file',
        type=str,
        default='data/arc-agi_training_solutions.json',
        help='Path to ARC solutions file'
    )
    parser.add_argument(
        '--num_tasks',
        type=int,
        default=10,
        help='Number of tasks to process'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-3.2-1B-Instruct',
        help='Base model name'
    )
    parser.add_argument(
        '--use_8bit',
        action='store_true',
        help='Use 8-bit quantization'
    )
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    
    # Augmentation arguments
    parser.add_argument('--include_basic_aug', action='store_true', default=True)
    parser.add_argument('--include_size_aug', action='store_true', default=True)
    parser.add_argument('--include_chain_aug', action='store_true', default=True)
    parser.add_argument('--include_repeat_aug', action='store_true', default=True)
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='loras_t4/ttt',
        help='Output directory for LoRA adapters'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T4-Optimized Test-Time Training for SEAL")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"8-bit: {args.use_8bit}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download ARC data if needed
    print("\nChecking ARC dataset...")
    data_dir = os.path.dirname(args.data_file) or "data"
    download_arc_data(data_dir)
    

    # Load tasks with fallback
    print("\nLoading tasks...")
    try:
        tasks = read_tasks_from_single_file(
            challenge_file=args.data_file,
            solution_file=args.solution_file
        )
        tasks = tasks[:args.num_tasks]
        print(f"Loaded {len(tasks)} tasks from single JSON file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Single JSON file not found or invalid: {e}")
        print("Falling back to public ARC directory format...")
        # Use public ARC format - files are directly in training/ directory
        from arclib.arc import read_tasks_from_arc_directory
        challenge_dir = os.path.join(data_dir, "training")
        solution_dir = os.path.join(data_dir, "training")  # Same dir in public ARC
        tasks = read_tasks_from_arc_directory(challenge_dir, solution_dir, max_tasks=args.num_tasks)
        print(f"Loaded {len(tasks)} tasks from ARC directory.")
    
    # Setup tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup formatters
    print("Setting up formatters...")
    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)
    
    # Setup LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    # Initialize TTT
    print("\nInitializing TTT model...")
    ttt = TTT(
        model_name=args.model_name,
        state_dict_path=None,
        lora_config=lora_config
    )
    print("TTT model initialized")
    
    # Setup augmenters
    augmenters = get_augmenters(
        include_basic=args.include_basic_aug,
        include_size=args.include_size_aug,
        include_chain=args.include_chain_aug,
        include_repeat=args.include_repeat_aug
    )
    print(f"Using {len(augmenters)} augmenters")
    
    # Process each task
    results = []
    print("\nProcessing tasks...")
    
    for i, task in enumerate(tqdm(tasks, desc="Tasks")):
        print(f"\n--- Task {i+1}/{len(tasks)}: {task.name} ---")
        
        # Process task to create training data
        print("Creating training data...")
        train_data = process_task(
            task=task,
            augmenters=augmenters,
            formatter=representer,
            tokenizer=tokenizer,
            leave_n=[1, 2],
            permute_n=1,
            Nmax=250,
            seed=0
        )
        
        if len(train_data) == 0:
            print("No training data generated, skipping...")
            continue
        
        print(f"Generated {len(train_data)} training examples")
        
        # Extract text
        task_text_list = [data["full_text"] for data in train_data]
        
        # Train LoRA adapter
        task_name_clean = task.name.split("-")[0] if "-" in task.name else task.name
        output_path = os.path.join(args.output_dir, task_name_clean)
        
        print(f"Training LoRA adapter...")
        print(f"Output: {output_path}")
        
        adapter_path = ttt.update_model(
            task_text_list=task_text_list,
            output_dir=output_path,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            loss_on_all_tokens=False
        )
        
        results.append({
            "task_name": task.name,
            "adapter_path": adapter_path,
            "num_training_examples": len(train_data)
        })
        
        print(f"✓ LoRA adapter saved: {adapter_path}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Save results summary
    results_file = os.path.join(args.output_dir, "ttt_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"TTT Complete!")
    print(f"Processed {len(results)} tasks")
    print(f"Results saved to: {results_file}")
    print("=" * 80)
    
    # Cleanup
    del ttt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
