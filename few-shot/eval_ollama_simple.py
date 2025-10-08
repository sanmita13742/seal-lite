"""
Simplified evaluation script for ARC using local Ollama.
No LoRA, no vLLM - just pure inference for laptop usage.
"""
import argparse
import json
import os
import numpy as np
from transformers import AutoTokenizer

from arclib.arc import make_submission, read_tasks_from_single_file, to_tuple
from arclib.eval import evaluate
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from arclib.voting import vote
from inference.engine_ollama import (
    get_sampling_params,
    initialize_engine,
    process_requests
)
from inference.preprocess import get_preprocessed_tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC with Ollama")
    
    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.2:latest",
        help="Ollama model name"
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL"
    )
    
    # Data settings
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to ARC challenge JSON file"
    )
    parser.add_argument(
        "--solution_file",
        type=str,
        default=None,
        help="Path to ARC solution JSON file"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of tasks to evaluate (use small number for laptop)"
    )
    
    # Generation settings
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--n_sample", type=int, default=1, help="Samples per input")
    
    # Output settings
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiments/ollama/",
        help="Output folder"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SEAL Evaluation with Local Ollama")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Tasks: {args.num_examples}")
    print(f"Temperature: {args.temperature}")
    print(f"Samples per task: {args.n_sample}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.experiment_folder, exist_ok=True)
    
    # Load tasks
    print("Loading tasks...")
    tasks = read_tasks_from_single_file(
        args.data_file,
        solution_file=args.solution_file,
        test=True
    )
    
    # Limit number of tasks for laptop
    if args.num_examples is not None:
        np.random.seed(args.seed)
        np.random.shuffle(tasks)
        tasks = tasks[:args.num_examples]
    
    print(f"Loaded {len(tasks)} tasks\n")
    
    # Initialize formatter
    messager = GPTTextMessageRepresenterV2(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                io_sep=" -> ",
                input_header="",
                output_header="",
                output_footer="#",
                grid_representer=PythonListGridRepresenter(),
            )
        )
    )
    
    # Initialize Ollama engine
    print("Initializing Ollama engine...")
    engine = initialize_engine(
        model=args.model_name,
        enable_lora=False,
        ollama_base_url=args.ollama_url
    )
    
    # Dummy tokenizer for compatibility
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare tasks
    print("Preparing prompts...")
    formatted_tasks = get_preprocessed_tasks(
        tasks=tasks,
        formatters=[messager],
        include_n=[0],  # No leave-out for baseline
        permute_n=1
    )
    
    # Create requests
    test_prompts = []
    for task_data in formatted_tasks:
        task_id = task_data["id"]
        prompt = task_data["prompt"]
        
        num_tokens = len(tokenizer.encode(prompt))
        sampling_params = get_sampling_params(
            tokenizer=tokenizer,
            num_tokens=num_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n=args.n_sample
        )
        
        test_prompts.append((prompt, sampling_params, None, task_id))
    
    # Process requests
    print(f"\nProcessing {len(test_prompts)} prompts...")
    print("Note: This may take a while on CPU!\n")
    
    all_outputs = process_requests(engine, test_prompts)
    
    # Parse outputs and create submission
    print("\nParsing outputs...")
    submission_data = {}
    
    for task in tasks:
        task_id = task.name
        if task_id in all_outputs:
            outputs = all_outputs[task_id]
            
            # Try to parse predictions
            parsed_preds = []
            for output in outputs:
                try:
                    # Extract grid from output
                    # Look for Python list format: [[...], [...]]
                    import re
                    pattern = r'\[\[.*?\]\]'
                    matches = re.findall(pattern, output, re.DOTALL)
                    if matches:
                        pred_grid = eval(matches[0])
                        parsed_preds.append(pred_grid)
                except:
                    pass
            
            if parsed_preds:
                # Vote if multiple samples
                if len(parsed_preds) > 1:
                    final_pred = vote(parsed_preds)
                else:
                    final_pred = parsed_preds[0]
                
                submission_data[task_id] = [final_pred]
            else:
                # Default empty prediction
                submission_data[task_id] = [[[0]]]
    
    # Evaluate
    print("\nEvaluating...")
    accuracy = evaluate(tasks, submission_data)
    
    print("\n" + "="*80)
    print(f"Accuracy: {accuracy:.2%} ({sum(1 for t in tasks if t.name in submission_data and evaluate([t], {t.name: submission_data[t.name]}) > 0)}/{len(tasks)} correct)")
    print("="*80 + "\n")
    
    # Save submission
    submission_file = os.path.join(args.experiment_folder, "submission.json")
    with open(submission_file, "w") as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Submission saved to: {submission_file}")
    
    # Save results
    results = {
        "accuracy": accuracy,
        "num_tasks": len(tasks),
        "model": args.model_name,
        "temperature": args.temperature,
        "n_sample": args.n_sample,
    }
    
    results_file = os.path.join(args.experiment_folder, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
