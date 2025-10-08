"""
T4-Optimized Self-Edit RL Loop for SEAL
Implements the full self-editing reinforcement learning pipeline on T4 GPU

Features:
- Config generation using small LM
- LoRA fine-tuning per task
- Memory-efficient batch processing
- No OpenAI API dependency
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
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
from inference.engine_t4 import T4Engine


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


# Self-edit prompts
SYSTEM_MESSAGE = """You are an AI assistant that analyzes ARC (Abstraction and Reasoning Corpus) tasks and generates training configurations."""

SELF_EDIT_PROMPT = """Analyze the examples above and generate a JSON configuration for training:

{
  "data_generation": {
    "basic_augmenters": true/false,
    "size_augmenters": true/false,
    "chain_augmenters": true/false,
    "repeat_augmenters": true/false
  },
  "training": {
    "strategy": "train_using_all_tokens" or "train_using_output_tokens",
    "num_train_epochs": 1-5,
    "learning_rate": 1e-5 to 1e-3
  }
}

Return ONLY valid JSON, no other text."""


def get_task_prompt(task, system_message: str, self_edit_prompt: str) -> str:
    """Format task as prompt for config generation."""
    train_examples = task.serialize()['train']
    formatted_examples = ""
    
    for example in train_examples:
        # Format input grid
        input_grid = example['input']
        input_str = "Input:\n"
        for row in input_grid:
            input_str += " ".join(map(str, row)) + "\n"
        
        # Format output grid
        output_grid = example['output']
        output_str = "\nOutput:\n"
        for row in output_grid:
            output_str += " ".join(map(str, row)) + "\n"
        
        formatted_examples += input_str + output_str + "\n"
    
    user_message = formatted_examples + "------\n\n" + self_edit_prompt
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def parse_config(response: str) -> Dict:
    """Parse JSON config from model response."""
    # Find JSON in response
    try:
        # Try direct parse
        config = json.loads(response)
        return config
    except json.JSONDecodeError:
        # Try to extract JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                config = json.loads(json_match.group(0))
                return config
            except:
                pass
    
    # Return default config
    return {
        "data_generation": {
            "basic_augmenters": True,
            "size_augmenters": True,
            "chain_augmenters": True,
            "repeat_augmenters": True
        },
        "training": {
            "strategy": "train_using_output_tokens",
            "num_train_epochs": 3,
            "learning_rate": 1e-4
        }
    }


def get_augmenters_from_config(config: Dict) -> List:
    """Get augmenters based on config."""
    from ttt_t4 import get_augmenters
    
    data_gen = config.get("data_generation", {})
    return get_augmenters(
        include_basic=data_gen.get("basic_augmenters", True),
        include_size=data_gen.get("size_augmenters", True),
        include_chain=data_gen.get("chain_augmenters", True),
        include_repeat=data_gen.get("repeat_augmenters", True)
    )


def get_test_time_train_data(task, augmenters, n: int = 1, permute_n: int = 1, seed: int = 0):
    """Generate test-time training data with augmentation and permutations."""
    from arclib.augmenters import IdentityAugmenter
    import numpy as np
    
    rng = np.random.RandomState(seed)
    train_data = []
    augmenters_with_identity = [IdentityAugmenter()] + augmenters
    
    for augmenter in augmenters_with_identity:
        # CRITICAL: Pass rng to augmenter (required for RandomTranslateXY and others)
        augmented_task = augmenter.apply_to_task(task, rng=rng, to_input=True, to_output=True)
        
        if len(augmented_task.train_examples) > n:
            examples_to_use = augmented_task.train_examples[:-n]
            
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
    formatted = formatter.encode(task)
    data = {"input": formatted[0], "output": formatted[1]}
    
    task_text = tokenizer.apply_chat_template(
        formatted[0] + [formatted[1]],
        tokenize=False,
        add_generation_prompt=True
    )
    
    tokens = tokenizer(task_text, return_tensors="pt")
    total_tokens = tokens['input_ids'].shape[1]
    
    data["total_tokens"] = total_tokens
    data["full_text"] = task_text
    
    return data if total_tokens < max_tokens else None


def get_formatted_data(task, augmenters, formatter, tokenizer, leave_n: int = 1,
                       permute_n: int = 1, seed: int = 0, max_tokens: int = 8192):
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


def process_task(task, augmenters, formatter, tokenizer, leave_n=[1, 2], permute_n=1, Nmax=250, seed=0):
    """Process task to create training data (matches original ttt.py logic)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    
    train = []
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer,
            leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)
    
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]
    
    return train


def main():
    parser = argparse.ArgumentParser(description='T4-Optimized Self-Edit RL Loop')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--skip_repeated_configs', action='store_true')
    
    # Data arguments
    parser.add_argument('--challenge_file', type=str, required=True)
    parser.add_argument('--solution_file', type=str, required=True)
    parser.add_argument('--n_tasks', type=int, default=12)
    parser.add_argument('--n_self_edits_per_task', type=int, default=5)
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-3.2-1B-Instruct'
    )
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='loras_t4/self_edit'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T4-Optimized Self-Edit RL Loop for SEAL")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Tasks: {args.n_tasks}")
    print(f"Self-edits per task: {args.n_self_edits_per_task}")
    print("=" * 80)
    
    # Create output directories
    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Download ARC data if needed
    print("\nChecking ARC dataset...")
    data_dir = os.path.dirname(args.challenge_file) or "data"
    download_arc_data(data_dir)
    
    # Load tasks with fallback
    print("\nLoading tasks...")
    try:
        tasks = read_tasks_from_single_file(
            challenge_file=args.challenge_file,
            solution_file=args.solution_file
        )
        tasks = tasks[:args.n_tasks]
        print(f"Loaded {len(tasks)} tasks from single JSON file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Single JSON file not found or invalid: {e}")
        print("Falling back to public ARC directory format...")
        from arclib.arc import read_tasks_from_arc_directory
        challenge_dir = os.path.join(data_dir, "training")
        solution_dir = os.path.join(data_dir, "training")  # Same dir in public ARC
        tasks = read_tasks_from_arc_directory(challenge_dir, solution_dir, max_tasks=args.n_tasks)
        print(f"Loaded {len(tasks)} tasks from ARC directory.")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup formatters
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
    
    # Phase 1: Generate configs using self-edit model
    print("\n" + "=" * 80)
    print("PHASE 1: Generating Configurations")
    print("=" * 80)
    
    config_engine = T4Engine(model_name=args.model_name, use_8bit=True)
    
    explored_configs = {}
    task_configs = {}
    
    for i, task in enumerate(tqdm(tasks, desc="Generating configs")):
        base_task_name = task.name.split("-")[0] if "-" in task.name else task.name
        
        if base_task_name.endswith("-1"):
            continue
        
        if base_task_name not in explored_configs:
            explored_configs[base_task_name] = set()
            task_configs[base_task_name] = []
        
        # Generate configs
        prompt = get_task_prompt(task, SYSTEM_MESSAGE, SELF_EDIT_PROMPT)
        
        attempts = 0
        while len(task_configs[base_task_name]) < args.n_self_edits_per_task and attempts < args.n_self_edits_per_task * 3:
            attempts += 1
            
            response = config_engine.generate(
                [prompt],
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True
            )[0]
            
            config = parse_config(response)
            
            # Check if valid
            if "data_generation" not in config or "training" not in config:
                continue
            
            # Check if unique (if required)
            config_key = json.dumps(config, sort_keys=True)
            if args.skip_repeated_configs and config_key in explored_configs[base_task_name]:
                continue
            
            explored_configs[base_task_name].add(config_key)
            task_configs[base_task_name].append({
                "config": config,
                "prompt": prompt,
                "response": response
            })
    
    config_engine.cleanup()
    torch.cuda.empty_cache()
    
    print(f"\nGenerated configs for {len(task_configs)} tasks")
    
    # Phase 2: Train LoRA adapters
    print("\n" + "=" * 80)
    print("PHASE 2: Training LoRA Adapters")
    print("=" * 80)
    
    # Initialize TTT
    ttt = TTT(
        model_name=args.model_name,
        state_dict_path=None,
        lora_config=lora_config
    )
    
    final_configs_and_indices = {}
    
    for task in tqdm(tasks, desc="Training tasks"):
        base_task_name = task.name.split("-")[0] if "-" in task.name else task.name
        
        if base_task_name.endswith("-1") or base_task_name not in task_configs:
            continue
        
        curr_task_configs = {}
        
        for task_idx, config_data in enumerate(task_configs[base_task_name]):
            config = config_data["config"]
            
            print(f"\n--- Task: {base_task_name}, Config {task_idx + 1} ---")
            
            # Get augmenters
            augmenters = get_augmenters_from_config(config)
            
            # Create training data
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
                print("No training data, skipping...")
                continue
            
            print(f"Training examples: {len(train_data)}")
            
            # Validate config
            training_config = config.get("training", {})
            num_epochs = training_config.get("num_train_epochs", 3)
            learning_rate = training_config.get("learning_rate", 1e-4)
            loss_strategy = training_config.get("strategy", "train_using_output_tokens")
            
            # Sanity checks
            if num_epochs <= 0 or num_epochs > 10:
                print(f"Invalid epochs ({num_epochs}), skipping...")
                continue
            
            if learning_rate <= 0 or learning_rate > 1e-2:
                print(f"Invalid learning rate ({learning_rate}), skipping...")
                continue
            
            # Limit total steps
            total_steps = num_epochs * len(train_data) // args.batch_size
            if total_steps > 500:
                print(f"Too many steps ({total_steps}), reducing epochs...")
                num_epochs = max(1, 500 * args.batch_size // len(train_data))
            
            # Train LoRA
            task_text_list = [data["full_text"] for data in train_data]
            output_path = os.path.join(exp_dir, base_task_name, str(task_idx))
            
            adapter_path = ttt.update_model(
                task_text_list=task_text_list,
                output_dir=output_path,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                lr_scheduler_type="cosine",
                loss_on_all_tokens=(loss_strategy == "train_using_all_tokens")
            )
            
            curr_task_configs[task_idx] = {
                "config": config,
                "adapter_path": adapter_path,
                "num_examples": len(train_data),
                "prompt": config_data["prompt"],
                "response": config_data["response"]
            }
            
            print(f"✓ Adapter saved: {adapter_path}")
            torch.cuda.empty_cache()
        
        final_configs_and_indices[base_task_name] = curr_task_configs
    
    # Save configs
    configs_file = os.path.join(exp_dir, "final_configs_and_indices.json")
    with open(configs_file, 'w') as f:
        json.dump(final_configs_and_indices, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Self-Edit RL Loop Complete!")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output: {exp_dir}")
    print(f"Configs: {configs_file}")
    print("=" * 80)
    
    # Cleanup
    del ttt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
