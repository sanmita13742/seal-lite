"""
Simple script to test SEAL with local Ollama.
No SLURM, no multi-GPU, no vLLM - just Ollama on your laptop.
"""
import os
import sys
import argparse


def check_ollama():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✓ Ollama is running")
            print(f"  Available models: {[m['name'] for m in models]}")
            return True
        else:
            print("✗ Ollama is not responding correctly")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def run_arc_eval(model: str, num_examples: int = 5):
    """Run ARC evaluation with Ollama."""
    print(f"\n{'='*80}")
    print(f"Running ARC Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Tasks: {num_examples}")
    print(f"{'='*80}\n")
    
    cmd = f"""python few-shot/eval_ollama_simple.py \
        --model_name={model} \
        --data_file=few-shot/data/arc-agi_evaluation_challenges.json \
        --solution_file=few-shot/data/arc-agi_evaluation_solutions.json \
        --num_examples={num_examples} \
        --temperature=0.0 \
        --n_sample=1 \
        --experiment_folder=experiments/ollama_test"""
    
    print(f"Running: {cmd}\n")
    os.system(cmd)


def run_knowledge_test(model: str):
    """Run simple knowledge incorporation test."""
    print(f"\n{'='*80}")
    print(f"Testing Knowledge Incorporation")
    print(f"{'='*80}\n")
    
    from general_knowledge.src.utils_ollama import init_ollama_client, format_answer_prompts
    
    # Initialize client
    client = init_ollama_client(model=model)
    
    # Test question
    test_questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare"
        }
    ]
    
    prompts = format_answer_prompts(test_questions, instruct_model=False)
    
    print("Testing basic Q&A:")
    for q, p in zip(test_questions, prompts):
        print(f"\nQuestion: {q['question']}")
        print(f"Expected: {q['answer']}")
        response = client.generate(p, temperature=0.0, max_tokens=50)
        print(f"Got: {response.strip()}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run SEAL experiments with local Ollama"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Ollama model name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["arc", "knowledge", "both", "check"],
        default="check",
        help="What to run"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of ARC examples (keep small for laptop)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SEAL with Local Ollama - Laptop Edition")
    print("="*80 + "\n")
    
    # Check Ollama
    if not check_ollama():
        print("\nPlease start Ollama first:")
        print("  ollama serve")
        print("\nAnd pull the model:")
        print(f"  ollama pull {args.model}")
        sys.exit(1)
    
    if args.mode == "check":
        print("\n✓ Everything looks good!")
        print("\nNext steps:")
        print("  1. Run ARC evaluation:")
        print(f"     python run_ollama_simple.py --mode=arc --num_examples=3")
        print("  2. Run knowledge test:")
        print(f"     python run_ollama_simple.py --mode=knowledge")
        print("  3. Run both:")
        print(f"     python run_ollama_simple.py --mode=both")
        return
    
    if args.mode in ["arc", "both"]:
        run_arc_eval(args.model, args.num_examples)
    
    if args.mode in ["knowledge", "both"]:
        run_knowledge_test(args.model)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
