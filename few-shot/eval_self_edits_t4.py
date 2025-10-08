"""
T4-Optimized Evaluation for SEAL
Evaluates LoRA adapters on ARC tasks using T4 GPU

Features:
- Batch processing for efficiency
- Memory-optimized inference
- Works with self-edit generated adapters
- No vLLM or OpenAI dependency
"""

import os
import json
import torch
import argparse
import glob
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import urllib.request

from transformers import AutoTokenizer
from arclib.arc import read_tasks_from_single_file, to_list
from arclib.representers import (
    TextTaskRepresenter,
    TextExampleRepresenter,
    PythonListGridRepresenter,
)
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.eval import evaluate
from arclib.voting import vote
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


def extract_grid_from_response(response: str, height: int, width: int):
    """Extract grid from model response."""
    import re
    
    # Remove markdown code blocks
    response = re.sub(r'```(?:python|json)?\s*', '', response)
    response = re.sub(r'```', '', response)
    
    # Try to find list patterns
    patterns = [
        r'\[\s*\[[\d\s,]+\]\s*(?:,\s*\[[\d\s,]+\]\s*)*\]',  # [[1,2,3],[4,5,6]]
        r'\[(?:\s*\d+\s*,?\s*)+\]',  # [1,2,3,4,5,6]
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                # Try to parse as Python list
                grid = eval(matches[0])
                
                # Handle flat list
                if isinstance(grid, list) and len(grid) > 0:
                    if not isinstance(grid[0], list):
                        # Reshape flat list
                        if len(grid) == height * width:
                            grid = [grid[i*width:(i+1)*width] for i in range(height)]
                    
                    # Validate dimensions
                    if len(grid) == height and all(len(row) == width for row in grid):
                        return grid
            except:
                pass
    
    # Return zero grid as fallback
    return [[0] * width for _ in range(height)]


def format_task_prompt(task, representer):
    """Format task as prompt for model."""
    messages = representer.process(task)
    if not messages:
        return None
    return representer.format_messages(messages)


def main():
    parser = argparse.ArgumentParser(description='T4-Optimized SEAL Evaluation')
    
    # Experiment arguments
    parser.add_argument('--experiment_folder', type=str, required=True)
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--lora_checkpoints_folder', type=str, default=None)
    
    # Data arguments
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--solution_file', type=str, required=True)
    parser.add_argument('--num_examples', type=int, default=None)
    
    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--n_self_edits', type=int, default=15)
    
    # Model arguments
    parser.add_argument('--use_8bit', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Format arguments
    parser.add_argument('--new_format', action='store_true', default=True)
    parser.add_argument('--include_n', type=int, default=1)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("T4-Optimized SEAL Evaluation")
    print("=" * 80)
    print(f"Model: {args.pretrained_checkpoint}")
    print(f"LoRA folder: {args.lora_checkpoints_folder}")
    print(f"Data: {args.data_file}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples: {args.n_sample}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.experiment_folder, exist_ok=True)
    
    # Download ARC data if needed
    print("\nChecking ARC dataset...")
    data_dir = os.path.dirname(args.data_file) or "data"
    download_arc_data(data_dir)
    
    # Load tasks with fallback
    print("\nLoading tasks...")
    try:
        tasks = read_tasks_from_single_file(
            args.data_file,
            solution_file=args.solution_file,
            test=True
        )
        print(f"Loaded {len(tasks)} tasks from single JSON file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Single JSON file not found or invalid: {e}")
        print("Falling back to public ARC directory format...")
        from arclib.arc import read_tasks_from_arc_directory
        challenge_dir = os.path.join(data_dir, "evaluation", "challenge")
        solution_dir = os.path.join(data_dir, "evaluation", "solution")
        tasks = read_tasks_from_arc_directory(challenge_dir, solution_dir)
        print(f"Loaded {len(tasks)} tasks from ARC directory.")
    
    # Get LoRA paths
    id_to_lora_path = {}
    if args.lora_checkpoints_folder:
        # Look for adapter folders
        for lora_path in glob.glob(f"{args.lora_checkpoints_folder}/*"):
            if os.path.isdir(lora_path):
                # Check if it's a valid adapter
                if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
                    lora_id = os.path.basename(lora_path)
                    id_to_lora_path[lora_id] = lora_path
                # Or check subdirectories (self-edit structure)
                else:
                    for sub_path in glob.glob(f"{lora_path}/*"):
                        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "adapter_config.json")):
                            lora_id = os.path.basename(lora_path)
                            id_to_lora_path[f"{lora_id}-{os.path.basename(sub_path)}"] = sub_path
    
    print(f"Found {len(id_to_lora_path)} LoRA adapters")
    
    # Filter tasks if needed
    if args.num_examples:
        np.random.seed(42)
        np.random.shuffle(tasks)
        tasks = tasks[:args.num_examples]
    
    print(f"Evaluating on {len(tasks)} tasks")
    
    # Setup formatter
    if args.new_format:
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
    
    # Initialize engine
    print("\nInitializing engine...")
    engine = T4Engine(
        model_name=args.pretrained_checkpoint,
        use_8bit=args.use_8bit,
        max_length=4096
    )
    
    # Evaluation loop
    results = {}
    all_correct = []
    
    print("\nEvaluating tasks...")
    for task in tqdm(tasks, desc="Tasks"):
        task_name = task.name
        task_id = task_name.split("-")[0] if "-" in task_name else task_name
        
        # Format prompt
        prompt = format_task_prompt(task, representer)
        if not prompt:
            continue
        
        # Find matching LoRA adapters
        matching_adapters = []
        for lora_id, lora_path in id_to_lora_path.items():
            if lora_id.startswith(task_id):
                matching_adapters.append((lora_id, lora_path))
        
        # Limit to n_self_edits
        matching_adapters = matching_adapters[:args.n_self_edits]
        
        if len(matching_adapters) == 0:
            print(f"No adapters found for task {task_id}")
            continue
        
        # Generate predictions with each adapter
        all_predictions = []
        
        for lora_id, lora_path in matching_adapters:
            # Load adapter
            engine.load_adapter(lora_path, adapter_name=lora_id)
            
            # Generate
            for _ in range(args.n_sample):
                response = engine.generate(
                    [prompt],
                    max_new_tokens=512,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0),
                    adapter_name=lora_id
                )[0]
                
                # Extract grid
                test_example = task.test[0]
                height = len(test_example.output.grid)
                width = len(test_example.output.grid[0])
                
                pred_grid = extract_grid_from_response(response, height, width)
                all_predictions.append(pred_grid)
            
            # Unload adapter
            engine.unload_adapter(lora_id)
        
        # Vote on predictions
        if len(all_predictions) > 0:
            final_prediction = vote(all_predictions)
        else:
            # Fallback
            final_prediction = [[0] * width for _ in range(height)]
        
        # Evaluate
        ground_truth = to_list(task.test[0].output.grid)
        correct = (final_prediction == ground_truth)
        
        all_correct.append(correct)
        
        results[task_name] = {
            "correct": correct,
            "prediction": final_prediction,
            "ground_truth": ground_truth,
            "num_adapters": len(matching_adapters),
            "num_predictions": len(all_predictions)
        }
    
    # Calculate accuracy
    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    
    # Save results
    results_file = os.path.join(args.experiment_folder, "eval_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "total_tasks": len(all_correct),
            "correct_tasks": sum(all_correct),
            "results": results
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.2%} ({sum(all_correct)}/{len(all_correct)})")
    print(f"Results saved to: {results_file}")
    print("=" * 80)
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
