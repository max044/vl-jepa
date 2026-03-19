#!/usr/bin/env python3
"""
AutoResearch Loop - Runs experiments autonomously on cloud instance
"""

import subprocess
import time
import re
import os
from datetime import datetime

# Experiment configurations to try
# Format: (name, param_changes)
EXPERIMENTS = [
    # Baseline first
    ("baseline", {}),
    
    # Learning rate experiments
    ("lr_3e-4", {"LEARNING_RATE": "3e-4"}),
    ("lr_3e-5", {"LEARNING_RATE": "3e-5"}),
    ("lr_1e-3", {"LEARNING_RATE": "1e-3"}),
    
    # Temperature experiments
    ("temp_0.1", {"TEMPERATURE": "0.1"}),
    ("temp_0.05", {"TEMPERATURE": "0.05"}),
    ("temp_0.03", {"TEMPERATURE": "0.03"}),
    
    # SIGReg weight experiments
    ("sigreg_0.05", {"SIGREG_WEIGHT": "0.05"}),
    ("sigreg_0.2", {"SIGREG_WEIGHT": "0.2"}),
    ("sigreg_0.0", {"SIGREG_WEIGHT": "0.0"}),
    
    # Combined experiments (best params so far)
    ("combo_1", {"LEARNING_RATE": "3e-4", "TEMPERATURE": "0.1"}),
    ("combo_2", {"LEARNING_RATE": "1e-4", "TEMPERATURE": "0.05", "SIGREG_WEIGHT": "0.05"}),
]

def read_train_py():
    with open("autoresearch/train.py", "r") as f:
        return f.read()

def write_train_py(content):
    with open("autoresearch/train.py", "w") as f:
        f.write(content)

def apply_params(content, params):
    """Apply parameter changes to train.py content"""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        for param_name, param_value in params.items():
            if line.startswith(f"{param_name} = "):
                line = f"{param_name} = {param_value}"
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def run_experiment(name, params, baseline_loss=None):
    """Run a single experiment and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Params: {params}")
    print(f"{'='*60}\n")
    
    # Read current train.py
    content = read_train_py()
    
    # Apply parameter changes
    modified_content = apply_params(content, params)
    
    # Write modified train.py
    write_train_py(modified_content)
    
    # Git commit (only train.py, ignore other changes)
    subprocess.run(["git", "stash", "-u"], check=False)  # Stash any other changes
    subprocess.run(["git", "add", "autoresearch/train.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"exp: {name}"], check=True)
    
    # Run training
    start_time = time.time()
    result = subprocess.run(
        ["bash", "autoresearch/run.sh"],
        capture_output=True,
        text=True,
        cwd="/root/vl-jepa"
    )
    elapsed = time.time() - start_time
    
    # Parse results from output
    output = result.stdout + result.stderr
    
    val_loss = None
    peak_vram = None
    
    for line in output.split('\n'):
        if 'val_loss:' in line and 'best_val_loss' not in line:
            match = re.search(r'val_loss:\s+([\d.]+)', line)
            if match:
                val_loss = float(match.group(1))
        if 'peak_vram_mb:' in line:
            match = re.search(r'peak_vram_mb:\s+([\d.]+)', line)
            if match:
                peak_vram = float(match.group(1))
    
    # Also check run.log file
    try:
        with open("autoresearch/run.log", "r") as f:
            log_content = f.read()
            for line in log_content.split('\n'):
                if 'val_loss:' in line and 'best_val_loss' not in line:
                    match = re.search(r'val_loss:\s+([\d.]+)', line)
                    if match:
                        val_loss = float(match.group(1))
                if 'peak_vram_mb:' in line:
                    match = re.search(r'peak_vram_mb:\s+([\d.]+)', line)
                    if match:
                        peak_vram = float(match.group(1))
    except:
        pass
    
    result_data = {
        "name": name,
        "params": params,
        "val_loss": val_loss,
        "peak_vram": peak_vram,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat()
    }
    
    return result_data

def main():
    os.chdir("/root/vl-jepa")
    
    results = []
    baseline_loss = None
    
    print("Starting AutoResearch Loop")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Estimated time: ~{len(EXPERIMENTS) * 5} minutes\n")
    
    for i, (name, params) in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Experiment: {name}")
        
        # Run experiment
        result = run_experiment(name, params, baseline_loss)
        results.append(result)
        
        # Update baseline if this is the baseline
        if name == "baseline":
            baseline_loss = result["val_loss"]
            print(f"\n✓ Baseline established: {baseline_loss:.6f}")
        
        # Record result
        status = "keep" if (baseline_loss is None or result["val_loss"] < baseline_loss) else "discard"
        if name != "baseline" and result["val_loss"] < baseline_loss:
            print(f"\n✓ NEW BEST! {result['val_loss']:.6f} < {baseline_loss:.6f}")
            baseline_loss = result["val_loss"]
        elif name != "baseline":
            print(f"\n✗ Worse than baseline: {result['val_loss']:.6f} >= {baseline_loss:.6f}")
            # Reset
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        
        # Save to results file
        with open("autoresearch/results.tsv", "a") as f:
            param_str = ",".join([f"{k}={v}" for k, v in params.items()]) if params else "baseline"
            f.write(f"{name}\t{result.get('val_loss', 'N/A')}\t{result.get('peak_vram', 'N/A')}\t{status}\t{param_str}\n")
        
        # Wait before next experiment (except for last one)
        if i < len(EXPERIMENTS) - 1:
            print(f"\nWaiting 5 minutes before next experiment...")
            time.sleep(300)  # 5 minutes
    
    # Summary
    print("\n" + "="*60)
    print("AUTO-RESEARCH COMPLETE")
    print("="*60)
    print("\nResults summary:")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:20s} loss={r.get('val_loss', 'N/A'):10s} VRAM={r.get('peak_vram', 'N/A'):8s}")
    print("-" * 60)
    print(f"\nBest val_loss: {baseline_loss:.6f}")
    print(f"Results saved to: autoresearch/results.tsv")

if __name__ == "__main__":
    main()
