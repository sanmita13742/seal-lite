"""
Setup script to download and organize ARC dataset for SEAL-Lite
Run this once before using SEAL-Lite scripts.

Usage:
    python few-shot/setup_arc_data.py
"""
import os
import shutil
from pathlib import Path

def setup_arc_dataset():
    """Download and setup ARC dataset in the correct location"""
    
    print("=" * 80)
    print("ARC Dataset Setup for SEAL-Lite")
    print("=" * 80)
    print()
    
    # Get the script's directory (few-shot/)
    script_dir = Path(__file__).parent
    # Go up one level to project root
    project_root = script_dir.parent
    
    # Paths
    arc_clone_dir = project_root / "ARC"
    data_dir = project_root / "data"
    training_dir = data_dir / "training"
    evaluation_dir = data_dir / "evaluation"
    
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print()
    
    # Step 1: Clone ARC repository if not exists
    if arc_clone_dir.exists():
        print("✓ ARC repository already exists")
    else:
        print("Cloning ARC repository...")
        try:
            ret = os.system(f"git clone https://github.com/fchollet/ARC.git {arc_clone_dir}")
            if ret == 0:
                print("✓ ARC repository cloned")
            else:
                print("✗ Failed to clone ARC repository")
                return False
        except Exception as e:
            print(f"✗ Failed to clone ARC: {e}")
            return False
    
    # Step 2: Create data directories
    print("\nCreating data directories...")
    training_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Data directories created")
    
    # Step 3: Copy training files
    print("\nCopying training files...")
    arc_training = arc_clone_dir / "data" / "training"
    if arc_training.exists():
        training_files = list(arc_training.glob("*.json"))
        if len(training_files) == 0:
            print(f"✗ No training files found in {arc_training}")
            return False
        for file in training_files:
            shutil.copy2(file, training_dir / file.name)
        print(f"✓ Copied {len(training_files)} training files")
    else:
        print(f"✗ ARC training directory not found: {arc_training}")
        return False
    
    # Step 4: Copy evaluation files
    print("\nCopying evaluation files...")
    arc_evaluation = arc_clone_dir / "data" / "evaluation"
    if arc_evaluation.exists():
        evaluation_files = list(arc_evaluation.glob("*.json"))
        if len(evaluation_files) == 0:
            print(f"✗ No evaluation files found in {arc_evaluation}")
            return False
        for file in evaluation_files:
            shutil.copy2(file, evaluation_dir / file.name)
        print(f"✓ Copied {len(evaluation_files)} evaluation files")
    else:
        print(f"✗ ARC evaluation directory not found: {arc_evaluation}")
        return False
    
    # Step 5: Verify setup
    print("\nVerifying setup...")
    training_count = len(list(training_dir.glob("*.json")))
    evaluation_count = len(list(evaluation_dir.glob("*.json")))
    
    print(f"✓ Training tasks: {training_count}")
    print(f"✓ Evaluation tasks: {evaluation_count}")
    
    if training_count == 0 or evaluation_count == 0:
        print("\n✗ Setup failed: No tasks found")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ARC Dataset Setup Complete!")
    print("=" * 80)
    print(f"\nData location: {data_dir}")
    print(f"Training: {training_dir} ({training_count} tasks)")
    print(f"Evaluation: {evaluation_dir} ({evaluation_count} tasks)")
    print("\nYou can now run SEAL-Lite scripts:")
    print("  python few-shot/ttt_t4.py --num_tasks=2")
    print("  python few-shot/self_edit_t4.py --experiment_name=demo --n_tasks=2")
    print()
    
    return True

if __name__ == "__main__":
    success = setup_arc_dataset()
    exit(0 if success else 1)
