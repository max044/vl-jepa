#!/usr/bin/env python3
"""
AutoResearch Loop - Runs experiments autonomously on cloud instance
Accumulates best parameters from previous experiments.
"""

import subprocess
import time
import re
import os
from datetime import datetime

# Experiment configurations to try
# Final comparison: 4 combinations of (Regression × SIGReg)
EXPERIMENT_CONFIGS = [
    # 1. No regression + With SIGReg (baseline optimized)
    ("baseline_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.05",
        "USE_REGRESSION": "False"
    }),
    
    # 2. No regression + No SIGReg
    ("baseline_no_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.0",
        "USE_REGRESSION": "False"
    }),
    
    # 3. With regression + With SIGReg
    ("regression_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.05",
        "USE_REGRESSION": "True",
        "REGRESSION_WEIGHT": "1.0"
    }),
    
    # 4. With regression + No SIGReg
    ("regression_no_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.0",
        "USE_REGRESSION": "True",
        "REGRESSION_WEIGHT": "1.0"
    }),
]

# Default parameters (best found so far)
DEFAULT_PARAMS = {
    "LEARNING_RATE": "3e-4",
    "TEMPERATURE": "0.03",
    "SIGREG_WEIGHT": "0.05",
}

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

def run_experiment(name, params, best_loss_so_far):
    """Run a single experiment and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Params: {params}")
    print(f"{'='*60}\n")
    
    # Discard unwanted changes first
    subprocess.run(["git", "checkout", "--", "uv.lock", "autoresearch/runner.py"], check=False)
    
    # Read current train.py
    content = read_train_py()
    
    # Apply parameter changes
    modified_content = apply_params(content, params)
    
    # Write modified train.py
    write_train_py(modified_content)
    
    # Commit the changes
    subprocess.run(["git", "add", "autoresearch/train.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"exp: {name}", "--allow-empty"], check=True)
    
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
    best_loss = float('inf')
    best_params = DEFAULT_PARAMS.copy()  # Start with defaults
    
    print("Starting AutoResearch Loop")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Default params: {DEFAULT_PARAMS}")
    print(f"Estimated time: ~{len(EXPERIMENT_CONFIGS) * 5} minutes\n")
    
    for i, (name, experiment_params) in enumerate(EXPERIMENT_CONFIGS):
        print(f"\n[{i+1}/{len(EXPERIMENT_CONFIGS)}] Experiment: {name}")
        
        print(f"Using params: {experiment_params}")
        
        # Run experiment
        result = run_experiment(name, experiment_params, best_loss)
        results.append(result)
        
        # Record result
        is_improvement = result["val_loss"] < best_loss
        status = "keep" if is_improvement else "discard"
        
        if is_improvement:
            print(f"\n✓ NEW BEST! {result['val_loss']:.6f} < {best_loss:.6f}")
            best_loss = result["val_loss"]
            # Update best_params with the params that worked
            best_params = experiment_params.copy()
            print(f"✓ Updated best params: {best_params}")
        else:
            print(f"\n✗ Not best: {result['val_loss']:.6f} >= {best_loss:.6f}")
        
        # Save to results file
        with open("autoresearch/results.tsv", "a") as f:
            param_str = ",".join([f"{k}={v}" for k, v in experiment_params.items()])
            f.write(f"{name}\t{result.get('val_loss', 'N/A')}\t{result.get('peak_vram', 'N/A')}\t{status}\t{param_str}\n")
        
        # Wait before next experiment (except for last one)
        if i < len(EXPERIMENT_CONFIGS) - 1:
            print(f"\nWaiting 5 minutes before next experiment...")
            time.sleep(300)  # 5 minutes
    
    # Summary
    print("\n" + "="*60)
    print("AUTO-RESEARCH COMPLETE")
    print("="*60)
    print(f"\nBest configuration found:")
    print(f"  val_loss: {best_loss:.6f}")
    print(f"  params: {best_params}")
    print("\nAll results:")
    print("-" * 60)
    print(f"{'Experiment':<20} {'Loss':<12} {'Status':<10} {'Params':<30}")
    print("-" * 60)
    for r in results:
        params_str = ",".join([f"{k}={v}" for k, v in r['params'].items()])[:30]
        status = "✓ BEST" if r['val_loss'] == best_loss else ("keep" if r['val_loss'] < best_loss * 1.1 else "discard")
        print(f"{r['name']:<20} {r.get('val_loss', 'N/A'):<12.6f} {status:<10} {params_str}")
    print("-" * 60)
    print(f"\nResults saved to: autoresearch/results.tsv")

if __name__ == "__main__":
    main()
