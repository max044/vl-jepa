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
# Remaining 3 experiments (baseline_sigreg already done: 0.868623)
EXPERIMENT_CONFIGS = [
    # 1. No regression + No SIGReg
    ("baseline_no_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.0",
        "USE_REGRESSION": "False"
    }),
    
    # 2. With regression + With SIGReg
    ("regression_sigreg", {
        "LEARNING_RATE": "3e-4",
        "TEMPERATURE": "0.03",
        "SIGREG_WEIGHT": "0.05",
        "USE_REGRESSION": "True",
        "REGRESSION_WEIGHT": "1.0"
    }),
    
    # 3. With regression + No SIGReg
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
    
    best_mIoU = None
    best_R1 = None
    peak_vram = None
    
    for line in output.split('\n'):
        if 'best_mIoU:' in line:
            match = re.search(r'best_mIoU:\s+([\d.]+)', line)
            if match:
                best_mIoU = float(match.group(1))
        if 'best_R@1:' in line:
            match = re.search(r'best_R@1:\s+([\d.]+)', line)
            if match:
                best_R1 = float(match.group(1))
        if 'peak_vram_mb:' in line:
            match = re.search(r'peak_vram_mb:\s+([\d.]+)', line)
            if match:
                peak_vram = float(match.group(1))
    
    # Also check run.log file
    try:
        with open("autoresearch/run.log", "r") as f:
            log_content = f.read()
            for line in log_content.split('\n'):
                if 'best_mIoU:' in line:
                    match = re.search(r'best_mIoU:\s+([\d.]+)', line)
                    if match:
                        best_mIoU = float(match.group(1))
                if 'best_R@1:' in line:
                    match = re.search(r'best_R@1:\s+([\d.]+)', line)
                    if match:
                        best_R1 = float(match.group(1))
                if 'peak_vram_mb:' in line:
                    match = re.search(r'peak_vram_mb:\s+([\d.]+)', line)
                    if match:
                        peak_vram = float(match.group(1))
    except:
        pass
    
    result_data = {
        "name": name,
        "params": params,
        "best_mIoU": best_mIoU,
        "best_R1": best_R1,
        "peak_vram": peak_vram,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat()
    }
    
    return result_data

def main():
    os.chdir("/root/vl-jepa")
    
    results = []
    best_mIoU = 0.0
    best_params = DEFAULT_PARAMS.copy()  # Start with defaults
    
    print("Starting AutoResearch Loop")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Default params: {DEFAULT_PARAMS}")
    print(f"Estimated time: ~{len(EXPERIMENT_CONFIGS) * 5} minutes\n")
    
    for i, (name, experiment_params) in enumerate(EXPERIMENT_CONFIGS):
        print(f"\n[{i+1}/{len(EXPERIMENT_CONFIGS)}] Experiment: {name}")
        
        print(f"Using params: {experiment_params}")
        
        # Run experiment
        result = run_experiment(name, experiment_params, best_mIoU)
        results.append(result)
        
        # Record result
        result_mIoU = result.get("best_mIoU", 0.0) or 0.0
        is_improvement = result_mIoU > best_mIoU
        status = "keep" if is_improvement else "discard"
        
        if is_improvement:
            print(f"\n✓ NEW BEST! mIoU={result_mIoU:.6f} > {best_mIoU:.6f}")
            best_mIoU = result_mIoU
            # Update best_params with the params that worked
            best_params = experiment_params.copy()
            print(f"✓ Updated best params: {best_params}")
        else:
            print(f"\n✗ Not best: mIoU={result_mIoU:.6f} <= {best_mIoU:.6f}")
        
        # Save to results file
        with open("autoresearch/results.tsv", "a") as f:
            param_str = ",".join([f"{k}={v}" for k, v in experiment_params.items()])
            f.write(f"{name}\t{result.get('best_mIoU', 'N/A')}\t{result.get('best_R1', 'N/A')}\t{result.get('peak_vram', 'N/A')}\t{status}\t{param_str}\n")
        
        # Next experiment starts immediately (no wait needed, each run has its own 5min budget)
    
    # Summary
    print("\n" + "="*60)
    print("AUTO-RESEARCH COMPLETE")
    print("="*60)
    print(f"\nBest configuration found:")
    print(f"  best_mIoU: {best_mIoU:.6f}")
    print(f"  params: {best_params}")
    print("\nAll results:")
    print("-" * 70)
    print(f"{'Experiment':<20} {'mIoU':<12} {'R@1':<10} {'Status':<10} {'Params':<20}")
    print("-" * 70)
    for r in results:
        params_str = ",".join([f"{k}={v}" for k, v in r['params'].items()])[:20]
        result_mIoU = r.get('best_mIoU', 0.0) or 0.0
        result_R1 = r.get('best_R1', 0.0) or 0.0
        status = "✓ BEST" if result_mIoU == best_mIoU else ("keep" if result_mIoU > best_mIoU * 0.9 else "discard")
        print(f"{r['name']:<20} {result_mIoU:<12.6f} {result_R1:<10.4f} {status:<10} {params_str}")
    print("-" * 70)
    print(f"\nResults saved to: autoresearch/results.tsv")

if __name__ == "__main__":
    main()
