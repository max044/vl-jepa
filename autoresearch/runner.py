#!/usr/bin/env python3
"""
Auto Research System for VL-JEPA

Runs experiments with a time budget, tracks best val_loss, and iterates on winners.

Usage:
    python autoresearch/runner.py --experiments 20 --time-limit 300
    
    # Continue from previous results
    python autoresearch/runner.py --continue-from results.json --time-limit 300
"""

import json
import subprocess
import time
import signal
import os
import sys
from datetime import datetime
from pathlib import Path

# Default hyperparameter space
HYPERPARAMETER_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size": [2, 4],
    "temperature": [0.05, 0.07, 0.1],
    "sigreg_weight": [0.0, 0.05, 0.1, 0.2],
    "warmup_steps": [100, 200],
    "y_encoder_lr_multiplier": [0.01, 0.05, 0.1],
    "predictor_layers": [0, 4, 8],  # 0 = all layers
}

# Fixed configuration
BASE_CONFIG = {
    "epochs": 100,  # Will be limited by time
    "grad_accumulation": 2,
    "dtype": "fp32",
}


def generate_random_config(seed=None):
    """Generate a random configuration from the hyperparameter space."""
    import random
    if seed is not None:
        random.seed(seed)
    
    config = BASE_CONFIG.copy()
    config.update({
        "lr": random.choice(HYPERPARAMETER_SPACE["learning_rate"]),
        "batch_size": random.choice(HYPERPARAMETER_SPACE["batch_size"]),
        "temperature": random.choice(HYPERPARAMETER_SPACE["temperature"]),
        "sigreg_weight": random.choice(HYPERPARAMETER_SPACE["sigreg_weight"]),
        "warmup_steps": random.choice(HYPERPARAMETER_SPACE["warmup_steps"]),
        "y_encoder_lr_multiplier": random.choice(HYPERPARAMETER_SPACE["y_encoder_lr_multiplier"]),
        "predictor_layers": random.choice(HYPERPARAMETER_SPACE["predictor_layers"]),
    })
    return config


def mutate_config(base_config, mutation_rate=0.3):
    """Mutate a base configuration for exploration around winners."""
    import random
    config = base_config.copy()
    
    if random.random() < mutation_rate:
        # Mutate learning rate (±50%)
        config["lr"] = base_config["lr"] * random.choice([0.5, 1.5, 2.0])
        config["lr"] = max(1e-5, min(3e-3, config["lr"]))
    
    if random.random() < mutation_rate:
        # Mutate temperature
        config["temperature"] = random.choice(HYPERPARAMETER_SPACE["temperature"])
    
    if random.random() < mutation_rate:
        # Mutate sigreg_weight
        config["sigreg_weight"] = random.choice(HYPERPARAMETER_SPACE["sigreg_weight"])
    
    if random.random() < mutation_rate:
        # Mutate batch size
        config["batch_size"] = random.choice(HYPERPARAMETER_SPACE["batch_size"])
    
    return config


def run_experiment(config, exp_id, time_limit=300):
    """
    Run a single experiment with time limit.
    
    Returns:
        dict: Results with val_loss, best_val_loss, steps_completed, etc.
    """
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}")
    print(f"Config: lr={config['lr']:.0e}, bs={config['batch_size']}, temp={config['temperature']}, sigreg={config['sigreg_weight']}")
    print(f"Time limit: {time_limit}s")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python3", "train.py",
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--wandb-run-name", f"autoresearch/exp_{exp_id}",
    ]
    
    # Add optional args
    if config.get("temperature"):
        cmd.extend(["--temperature", str(config["temperature"])])
    if config.get("sigreg_weight") is not None:
        cmd.extend(["--sigreg-weight", str(config["sigreg_weight"])])
    if config.get("warmup_steps"):
        cmd.extend(["--warmup-steps", str(config["warmup_steps"])])
    if config.get("y_encoder_lr_multiplier"):
        cmd.extend(["--y-encoder-lr-multiplier", str(config["y_encoder_lr_multiplier"])])
    if config.get("predictor_layers") is not None:
        cmd.extend(["--predictor-layers", str(config["predictor_layers"])])
    
    # Setup monitoring
    start_time = time.time()
    best_val_loss = float('inf')
    steps_completed = 0
    
    # Create temp log file
    log_file = Path(f"autoresearch/logs/exp_{exp_id}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run with timeout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Monitor output
    with open(log_file, 'w') as f:
        for line in process.stdout:
            f.write(line)
            f.flush()
            
            # Parse val_loss
            if "Val loss:" in line:
                try:
                    val_loss = float(line.split("Val loss:")[1].split("|")[0].strip())
                    best_val_loss = min(best_val_loss, val_loss)
                    print(f"  → Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                except:
                    pass
            
            # Parse steps
            if "batches/epoch" in line:
                try:
                    steps_completed = int(line.split("(")[1].split(" batches")[0])
                except:
                    pass
            
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print(f"\n⏰ Time limit reached ({elapsed:.0f}s). Stopping...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except:
                    process.kill()
                break
    
    elapsed = time.time() - start_time
    
    result = {
        "exp_id": exp_id,
        "config": config,
        "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
        "steps_completed": steps_completed,
        "time_elapsed": elapsed,
        "timestamp": datetime.now().isoformat(),
        "log_file": str(log_file),
        "status": "completed" if elapsed < time_limit else "timeout"
    }
    
    print(f"\n✅ Experiment {exp_id} finished")
    print(f"   Best val_loss: {result['best_val_loss']:.4f}" if result['best_val_loss'] else "   No validation completed")
    print(f"   Time: {elapsed:.0f}s | Steps: {steps_completed}")
    
    return result


def select_top_configs(results, top_k=3):
    """Select top k configurations based on best_val_loss."""
    valid_results = [r for r in results if r.get('best_val_loss') is not None]
    if not valid_results:
        return []
    
    sorted_results = sorted(valid_results, key=lambda x: x['best_val_loss'])
    return sorted_results[:top_k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto Research for VL-JEPA")
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--time-limit", type=int, default=300, help="Time limit per experiment (seconds)")
    parser.add_argument("--continue-from", type=str, default=None, help="Continue from existing results.json")
    parser.add_argument("--explore-winners", action="store_true", help="Generate variations of top configs")
    parser.add_argument("--output", type=str, default="autoresearch/results.json", help="Output results file")
    args = parser.parse_args()
    
    # Load existing results if continuing
    all_results = []
    completed_exp_ids = set()
    
    if args.continue_from and os.path.exists(args.continue_from):
        with open(args.continue_from) as f:
            all_results = json.load(f)
        completed_exp_ids = {r['exp_id'] for r in all_results}
        print(f"Loaded {len(all_results)} previous experiments")
        
        # Show current best
        top = select_top_configs(all_results, top_k=5)
        if top:
            print("\nCurrent top configs:")
            for i, r in enumerate(top, 1):
                print(f"  {i}. {r['exp_id']}: val_loss={r['best_val_loss']:.4f}, lr={r['config']['lr']:.0e}")
    
    # Generate configs to run
    configs_to_run = []
    
    if args.explore_winners and all_results:
        # Generate mutations of top configs
        top_configs = select_top_configs(all_results, top_k=3)
        for i in range(args.experiments):
            base = top_configs[i % len(top_configs)]['config'] if top_configs else None
            if base:
                config = mutate_config(base)
            else:
                config = generate_random_config(seed=i)
            exp_id = f"mut_{i}_{datetime.now().strftime('%H%M%S')}"
            configs_to_run.append((exp_id, config))
    else:
        # Random exploration
        for i in range(args.experiments):
            exp_id = f"exp_{len(all_results) + i}"
            if exp_id not in completed_exp_ids:
                config = generate_random_config(seed=len(all_results) + i)
                configs_to_run.append((exp_id, config))
    
    print(f"\nRunning {len(configs_to_run)} experiments...")
    print(f"Time budget: ~{len(configs_to_run) * args.time_limit / 60:.0f} minutes")
    print(f"Output: {args.output}\n")
    
    # Run experiments
    for exp_id, config in configs_to_run:
        try:
            result = run_experiment(config, exp_id, time_limit=args.time_limit)
            all_results.append(result)
            
            # Save after each experiment
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Show running best
            top = select_top_configs(all_results, top_k=3)
            if top:
                print(f"\n🏆 Current best: {top[0]['exp_id']} with val_loss={top[0]['best_val_loss']:.4f}")
                
        except Exception as e:
            print(f"❌ Error in experiment {exp_id}: {e}")
            all_results.append({
                "exp_id": exp_id,
                "config": config,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            })
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    top = select_top_configs(all_results, top_k=5)
    if top:
        print("\nTop 5 configurations:")
        for i, r in enumerate(top, 1):
            c = r['config']
            print(f"\n{i}. {r['exp_id']}")
            print(f"   Val Loss: {r['best_val_loss']:.4f}")
            print(f"   LR: {c['lr']:.0e}, BS: {c['batch_size']}, Temp: {c['temperature']}")
            print(f"   SigReg: {c['sigreg_weight']}, Warmup: {c['warmup_steps']}")
    
    print(f"\nResults saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review results.json")
    print("  2. Run: python autoresearch/runner.py --explore-winners --continue-from results.json")
    print("  3. Or: python train.py with the best config for full training")


if __name__ == "__main__":
    main()
