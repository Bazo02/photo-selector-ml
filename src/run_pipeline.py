import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n▶️ Running: {script_path}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        raise RuntimeError(f"❌ Script failed: {script_path}")
    print(f"✅ Finished: {script_path}\n")

if __name__ == "__main__":
    # Get absolute paths to scripts
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, "src")

    steps = [
                                                             # Step 1: Create synthetic image data
        os.path.join(src_path, "train.py"),                  # Step 2: Train the pairwise model
        os.path.join(src_path, "scorer.py"),                 # Step 3: Score image pairs
        os.path.join(src_path, "enhancer.py")                # Step 4: Enhance selected images
    ]

    print("🚀 Starting the full pipeline...\n")

    for step in steps:
        run_script(step)

    print("🎉 All steps completed successfully!")
