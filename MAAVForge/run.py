"""
Entry script for KLA Hackathon Benchmarking.
Usage: python run.py <input-dir> <output-dir>

This script acts as the required entry point and delegates to the fully 
optimized evaluate.py engine which handles TTA, AMP, and torch.compile.
"""
import sys
import os
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Resolve absolute paths to ensure safety
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_script = os.path.join(script_dir, "evaluate.py")
    
    # The benchmark requires the model to be in the 'models/' directory
    model_weights = os.path.join(script_dir, "models", "best_model.pt")
    
    cmd = [
        sys.executable, evaluate_script,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--model_weights", model_weights
    ]
    
    print("==================================================")
    print(" MAAVForge Restoration Pipeline (run.py entry)    ")
    print("==================================================")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Weights:    {model_weights}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
