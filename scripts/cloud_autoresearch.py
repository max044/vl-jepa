"""
Cloud Auto-Research Launcher for VL-JEPA on Vast.ai

Launches GPU instances for rapid hyperparameter experimentation.
Optimized for RTX 4090 (~$0.30-0.50/hour) or RTX 3090 (~$0.20-0.40/hour).

Usage:
    # Launch new instance with auto-research
    python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 5 --experiments 20
    
    # Run on existing instance
    python scripts/cloud_autoresearch.py --instance-id 12345 --config scripts/sweep_config.json

Features:
- Automatic instance provisioning with cheapest available GPU
- Fast data download from HF Storage (XET) using hf sync
- Parallel experiment execution with result tracking
- Automatic result upload to HF Storage
- Instance termination after completion
"""

import os
import sys
import time
import json
import argparse
import csv
import re
import subprocess
from pathlib import Path
def load_dotenv():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)

try:
    from vastai_sdk import VastAI
    HAS_VAST_SDK = True
except ImportError:
    HAS_VAST_SDK = False
    print("⚠️  vastai_sdk not installed. Install with: pip install vastai")


# Configuration
HF_STORAGE_BUCKET = "max044/charades-sta-storage"
PROJECT_NAME = "vl-jepa-autoresearch"

# GPU pricing and specs
GPU_OPTIONS = {
    "rtx4090": {"min_gpu_ram": 24, "max_price": 0.60, "cuda": "12.1", "vram": 24},
    "rtx3090": {"min_gpu_ram": 24, "max_price": 0.40, "cuda": "12.1", "vram": 24},
    "a6000": {"min_gpu_ram": 48, "max_price": 0.90, "cuda": "12.1", "vram": 48},
    "a5000": {"min_gpu_ram": 24, "max_price": 0.70, "cuda": "12.1", "vram": 24},
}


def extract_metrics(log_text):
    """Parse the log text to find validation metrics."""
    val_loss = None
    info_nce = None
    recall_at_1 = None
    
    # Look for various metric formats
    for line in log_text.splitlines():
        if "Val loss:" in line and "Val InfoNCE:" in line:
            loss_match = re.search(r"Val loss:\s*([0-9.]+)", line)
            nce_match = re.search(r"Val InfoNCE:\s*([0-9.]+)", line)
            if loss_match:
                val_loss = float(loss_match.group(1))
            if nce_match:
                info_nce = float(nce_match.group(1))
        
        # Also look for recall metrics
        if "R@1:" in line or "recall@1" in line.lower():
            recall_match = re.search(r"R@1[=:]\s*([0-9.]+)", line)
            if recall_match:
                recall_at_1 = float(recall_match.group(1))
                
    return val_loss, info_nce, recall_at_1


def generate_cloud_init_script(args):
    """Generate the cloud-init script for instance setup."""
    gpu_config = GPU_OPTIONS[args.gpu]
    
    script = f'''#!/bin/bash
set -e

echo "=== VL-JEPA Auto-Research Setup ==="

# Install system dependencies
apt-get update -qq
apt-get install -y -qq git curl

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate huggingface-hub datasets sentence-transformers
pip install -q opencv-python-headless timm wandb

# Install HF CLI with XET support
pip install -q "huggingface-hub>=0.24.0"

# Login to HF if token available
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# Setup data directory
echo "Setting up data directory..."
mkdir -p ~/data/Charades_v1_480

# Download dataset from HF Storage (fast XET sync)
echo "Downloading dataset from HF Storage (this may take 10-20 minutes)..."
cd ~/data
hf sync hf://buckets/{HF_STORAGE_BUCKET}/Charades_v1_480 Charades_v1_480 --progress || {{
    echo "Warning: hf sync failed, falling back to git-lfs style download..."
    # Fallback: download annotations at minimum
    curl -sL https://raw.githubusercontent.com/lntzm/MESM/main/data/charades/annotations/charades_sta_train.txt -o Charades_v1_480/charades_sta_train.txt
    curl -sL https://raw.githubusercontent.com/lntzm/MESM/main/data/charades/annotations/charades_sta_test.txt -o Charades_v1_480/charades_sta_test.txt
}}

# Clone repository
echo "Cloning VL-JEPA repository..."
cd ~
if [ -d "vl-jepa" ]; then
    cd vl-jepa && git pull
else
    git clone https://github.com/max044/vl-jepa.git
    cd vl-jepa
fi

# Create results directory
mkdir -p autoresearch/results

echo "=== Setup Complete ==="
echo "Ready to run experiments!"
'''
    return script


def launch_new_instance(args):
    """Launch a new Vast.ai instance for experiments."""
    if not HAS_VAST_SDK:
        print("❌ vastai_sdk required for launching instances")
        sys.exit(1)
    
    load_dotenv()
    api_key = os.getenv("VASTAI_API_KEY")
    if not api_key:
        print("❌ VASTAI_API_KEY not found in environment")
        print("Get your API key from: https://cloud.vast.ai/account/")
        sys.exit(1)
    
    sdk = VastAI(api_key=api_key)
    gpu_config = GPU_OPTIONS[args.gpu]
    
    print(f"\n🔍 Searching for {args.gpu} instances under ${gpu_config['max_price']}/hour...")
    
    # Search for instances
    try:
        # Use CLI for search (more reliable)
        search_cmd = [
            "uv", "run", "vastai", "search", "offers",
            f"gpu_name={args.gpu}",
            f"dph <= {gpu_config['max_price']}",
            "cuda_vers >= 12",
            "inet_up > 50",
            "-o", "dph",
        ]
        
        result = subprocess.run(search_cmd, capture_output=True, text=True, env={**os.environ, "VASTAI_API_KEY": api_key})
        
        if result.returncode != 0:
            print(f"❌ Error searching for instances: {result.stderr}")
            sys.exit(1)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            print("❌ No suitable instances found. Try different GPU or higher budget.")
            sys.exit(1)
        
        # Parse first available instance
        headers = lines[0].split()
        first_offer = lines[1].split()
        
        instance_id = first_offer[0]
        price_per_hour = float(first_offer[headers.index('dph')])
        
        print(f"✓ Found instance: ID={instance_id}, Price=${price_per_hour:.2f}/hour")
        print(f"  Estimated cost: ${price_per_hour * args.budget:.2f} for {args.budget}h")
        
        # Confirm launch
        confirm = input("\n🚀 Launch instance? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        
        # Create instance
        print("\n📦 Creating instance...")
        create_cmd = [
            "vastai", "create", "instance", instance_id,
            "--disk", "50",
            "--image", "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
            "--env", f"HF_TOKEN={os.getenv('HF_TOKEN', '')}",
            "--onstart-cmd", generate_cloud_init_script(args),
        ]
        
        result = subprocess.run(create_cmd, capture_output=True, text=True, env={**os.environ, "VASTAI_API_KEY": api_key})
        
        if result.returncode != 0:
            print(f"❌ Error creating instance: {result.stderr}")
            sys.exit(1)
        
        print(f"✓ Instance created successfully!")
        print(f"\n⏳ Waiting for instance to be ready (this may take 2-3 minutes)...")
        
        # Poll for instance readiness
        new_instance_id = None
        for line in result.stdout.split('\n'):
            if 'Created instance' in line:
                new_instance_id = line.split()[-1]
                break
        
        if new_instance_id:
            print(f"  Instance ID: {new_instance_id}")
            print(f"\nTo monitor: vastai show instances")
            print(f"To connect: vastai ssh {new_instance_id}")
            
            # Wait for setup to complete
            time.sleep(30)
            print(f"\n✓ Instance should be ready for experiments!")
            return new_instance_id
        else:
            print("⚠️  Could not parse instance ID. Check vastai show instances")
            return None
            
    except FileNotFoundError:
        print("❌ vastai CLI not found. Install with: pip install vastai")
        sys.exit(1)


def run_experiments_on_instance(instance_id, args):
    """Run experiments on an existing Vast.ai instance."""
    load_dotenv()
    api_key = os.getenv("VASTAI_API_KEY")
    
    if not HAS_VAST_SDK or not api_key:
        print("❌ VastAI SDK and API key required")
        return
    
    sdk = VastAI(api_key=api_key)
    
    # Load config
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        # Create default config
        configs = generate_default_configs(args.experiments)
        print(f"✓ Generated {len(configs)} default experiment configs")
    else:
        with open(args.config, 'r') as f:
            configs = json.load(f)
    
    # Prepare results file
    results_file = args.output
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(['timestamp', 'experiment_id', 'val_loss', 'info_nce', 'recall_at_1', 'status', 'params'])
        
        print(f"\n🌟 Starting Auto-Research on Instance {instance_id}")
        print(f"   Experiments: {len(configs)}")
        print(f"   Results: {results_file}")
        
        for i, config_dict in enumerate(configs):
            exp_id = config_dict.pop("exp_id", f"exp_{i+1:03d}")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n🚀 [{i+1}/{len(configs)}] Running {exp_id}")
            print(f"   Config: {json.dumps(config_dict, indent=2)}")
            
            # Build CLI arguments
            cli_args = ""
            for k, v in config_dict.items():
                k_cli = k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        cli_args += f"--{k_cli} "
                else:
                    cli_args += f"--{k_cli} {v} "
            
            # Add short run flags
            cli_args += "--epochs 1 --max-steps 100 --val-every 50"
            
            # Execute experiment
            log_file = f"autoresearch/run_{exp_id}.log"
            run_cmd = f"mkdir -p ~/vl-jepa/autoresearch && cd ~/vl-jepa && python train.py {cli_args} 2>&1 | tee {log_file}"
            
            print(f"   Executing...")
            start_time = time.time()
            
            try:
                sdk.execute(ID=instance_id, COMMAND=run_cmd)
                elapsed = time.time() - start_time
                
                print(f"   ✅ Completed in {elapsed:.1f}s")
                
                # Retrieve and parse logs
                cat_cmd = f"cat ~/vl-jepa/{log_file}"
                log_output = sdk.execute(ID=instance_id, COMMAND=cat_cmd)
                
                log_text = log_output.get('output', str(log_output)) if isinstance(log_output, dict) else str(log_output)
                val_loss, info_nce, recall_at_1 = extract_metrics(log_text)
                
                # Determine status
                if val_loss is not None:
                    status = "keep"
                    print(f"   📊 val_loss: {val_loss:.4f}, info_nce: {info_nce:.4f if info_nce else 'N/A'}, R@1: {recall_at_1:.3f if recall_at_1 else 'N/A'}")
                else:
                    status = "crash"
                    print(f"   ⚠️  Could not parse metrics")
                
                # Log results
                writer.writerow([
                    timestamp, exp_id,
                    val_loss if val_loss else "0.0",
                    info_nce if info_nce else "0.0",
                    recall_at_1 if recall_at_1 else "0.0",
                    status,
                    json.dumps(config_dict)
                ])
                f.flush()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                writer.writerow([timestamp, exp_id, "0.0", "0.0", "0.0", "error", json.dumps(config_dict)])
                f.flush()
    
    print(f"\n✅ Auto-Research completed! Results saved to {results_file}")
    
    # Upload results to HF Storage
    print(f"\n📤 Uploading results to HF Storage...")
    try:
        subprocess.run([
            "hf", "upload", results_file,
            f"{HF_STORAGE_BUCKET}/autoresearch/results/{os.path.basename(results_file)}"
        ], check=True)
        print(f"✓ Results uploaded!")
    except Exception as e:
        print(f"⚠️  Could not upload results: {e}")


def generate_default_configs(n_experiments):
    """Generate default experiment configurations for hyperparameter search."""
    import random
    
    base_config = {
        "batch_size": 4,
        "lr": 3e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "temperature": 0.07,
        "sigreg_weight": 0.1,
        "warmup_steps": 200,
    }
    
    configs = []
    
    # Grid search for key hyperparameters
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [2, 4, 8]
    lora_ranks = [32, 64, 128]
    temperatures = [0.05, 0.07, 0.1]
    
    # Generate configs
    for i in range(min(n_experiments, 20)):
        config = base_config.copy()
        config["exp_id"] = f"exp_{i+1:03d}"
        
        # Vary parameters systematically
        if i < len(learning_rates):
            config["lr"] = learning_rates[i]
            config["description"] = f"lr_{learning_rates[i]}"
        elif i < len(learning_rates) + len(batch_sizes):
            config["batch_size"] = batch_sizes[i - len(learning_rates)]
            config["description"] = f"bs_{batch_sizes[i - len(learning_rates)]}"
        elif i < len(learning_rates) + len(batch_sizes) + len(lora_ranks):
            config["lora_r"] = lora_ranks[i - len(learning_rates) - len(batch_sizes)]
            config["description"] = f"lora_r_{lora_ranks[i - len(learning_rates) - len(batch_sizes)]}"
        else:
            # Random combination
            config["lr"] = random.choice(learning_rates)
            config["batch_size"] = random.choice(batch_sizes)
            config["temperature"] = random.choice(temperatures)
            config["description"] = "random"
        
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="VL-JEPA Cloud Auto-Research Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch new RTX 4090 instance for 5 hours
  python scripts/cloud_autoresearch.py --gpu rtx4090 --budget 5 --experiments 20
  
  # Use existing instance
  python scripts/cloud_autoresearch.py --instance-id 12345 --config scripts/sweep_config.json
  
  # Dry run to see what would be launched
  python scripts/cloud_autoresearch.py --gpu rtx3090 --budget 3 --dry-run
        """
    )
    
    # Instance options
    parser.add_argument("--gpu", choices=list(GPU_OPTIONS.keys()), default="rtx4090",
                       help="GPU type for new instance (default: rtx4090)")
    parser.add_argument("--instance-id", type=int,
                       help="Existing Vast.ai instance ID to use")
    
    # Experiment options
    parser.add_argument("--budget", type=float, default=5.0,
                       help="Budget in hours for new instance (default: 5)")
    parser.add_argument("--experiments", type=int, default=20,
                       help="Number of experiments to run (default: 20)")
    parser.add_argument("--config", type=str, default="scripts/sweep_config.json",
                       help="JSON config file with experiment parameters")
    parser.add_argument("--output", type=str, default="autoresearch/results.tsv",
                       help="Output file for results")
    
    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    parser.add_argument("--terminate", action="store_true",
                       help="Terminate instance after experiments complete")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VL-JEPA Cloud Auto-Research")
    print("=" * 70)
    
    if args.instance_id:
        # Run on existing instance
        print(f"\nUsing existing instance: {args.instance_id}")
        run_experiments_on_instance(args.instance_id, args)
    else:
        # Launch new instance
        gpu_config = GPU_OPTIONS[args.gpu]
        print(f"\nConfiguration:")
        print(f"  GPU: {args.gpu} ({gpu_config['vram']}GB VRAM)")
        print(f"  Max price: ${gpu_config['max_price']}/hour")
        print(f"  Budget: {args.budget} hours")
        print(f"  Experiments: {args.experiments}")
        print(f"  Est. cost: ${gpu_config['max_price'] * args.budget:.2f}")
        print(f"  Data source: HF Storage (XET)")
        
        if args.dry_run:
            print("\n[DRY RUN] Setup script:")
            print(generate_cloud_init_script(args))
            return
        
        # Launch instance
        instance_id = launch_new_instance(args)
        
        if instance_id:
            # Run experiments
            print(f"\n⏳ Waiting for setup to complete...")
            time.sleep(60)  # Give time for setup
            
            args.instance_id = int(instance_id)
            run_experiments_on_instance(args.instance_id, args)
            
            # Optionally terminate
            if args.terminate:
                print(f"\n🗑️  Terminating instance {instance_id}...")
                subprocess.run([
                    "vastai", "destroy", "instance", str(instance_id)
                ], env={**os.environ, "VASTAI_API_KEY": os.getenv("VASTAI_API_KEY", "")})
                print("✓ Instance terminated")


if __name__ == "__main__":
    main()
