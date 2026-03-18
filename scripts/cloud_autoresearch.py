import os
import time
import json
import argparse
import csv
import re
from dotenv import load_dotenv
from vastai_sdk import VastAI

def extract_metrics(log_text):
    """Parse the log text to find the best/last validation metrics."""
    val_loss = None
    info_nce = None
    
    # Look for lines like: "  → Val loss: 1.2345 | Val InfoNCE: 0.9876"
    for line in log_text.splitlines():
        if "Val loss:" in line and "Val InfoNCE:" in line:
            loss_match = re.search(r"Val loss:\s*([0-9.]+)", line)
            nce_match = re.search(r"Val InfoNCE:\s*([0-9.]+)", line)
            if loss_match:
                val_loss = float(loss_match.group(1))
            if nce_match:
                info_nce = float(nce_match.group(1))
                
    return val_loss, info_nce

def main():
    parser = argparse.ArgumentParser(description="Cloud Autoresearch Orchestrator (Vast.ai)")
    parser.add_argument("--instance-id", type=int, required=True, help="Vast.ai Instance ID to run on")
    parser.add_argument("--config", type=str, default="scripts/sweep_config.json", help="JSON file containing the list of configs to sweep")
    parser.add_argument("--output", type=str, default="results.tsv", help="Output file for results")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("VASTAI_API_KEY")
    if not api_key:
        print("❌ VASTAI_API_KEY not found in .env")
        return

    sdk = VastAI(api_key=api_key)

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        configs = json.load(f)

    # Prepare results file
    file_exists = os.path.isfile(args.output)
    
    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(['experiment_id', 'val_loss', 'info_nce', 'status', 'params'])

        print(f"🌟 Starting Cloud Autoresearch on Vast Instance {args.instance_id}")
        
        for i, config_dict in enumerate(configs):
            exp_id = config_dict.pop("exp_id", f"exp_{i+1}")
            print(f"\n🚀 Running {exp_id} ({i+1}/{len(configs)})")
            print(f"   Parameters: {config_dict}")
            
            # Build CLI arguments from the dict
            # We skip 'epochs' if the user provided it, but we enforce short runs in the config
            cli_args = ""
            for k, v in config_dict.items():
                k_cli = k.replace("_", "-")
                if isinstance(v, bool):
                    if v: cli_args += f"--{k_cli} "
                else:
                    cli_args += f"--{k_cli} {v} "
            
            # Command to execute (we pipe to a file to be safe, then read it back)
            # The execution might block. If it doesn't, we'd need to poll.
            # vast_sdk.execute essentially wraps raw execution.
            log_file = f"run_{exp_id}.log"
            run_cmd = f"cd ~/vl-jepa && bash scripts/train_cloud.sh {cli_args} > {log_file} 2>&1"
            
            print(f"   Executing: {run_cmd}")
            print("   ⏳ Waiting for completion...")
            
            start_time = time.time()
            try:
                # Execute the training command
                # Note: this might block depending on the SDK implementation.
                sdk.execute(ID=args.instance_id, COMMAND=run_cmd)
                
                elapsed = time.time() - start_time
                print(f"   ✅ Execution finished in {elapsed:.1f}s")
                
                # Retrieve the log to parse metrics
                cat_cmd = f"cat ~/vl-jepa/{log_file}"
                log_output = sdk.execute(ID=args.instance_id, COMMAND=cat_cmd)
                
                # If SDK returns a dict with stdout, extract it
                if isinstance(log_output, dict) and 'output' in log_output:
                    log_text = log_output['output']
                elif isinstance(log_output, str):
                    log_text = log_output
                else:
                    log_text = str(log_output)
                    
                val_loss, info_nce = extract_metrics(log_text)
                
                if val_loss is not None and info_nce is not None:
                    print(f"   📊 Results -> val_loss: {val_loss:.4f}, info_nce: {info_nce:.4f}")
                    writer.writerow([exp_id, val_loss, info_nce, "keep", json.dumps(config_dict)])
                else:
                    print(f"   ⚠️ Could not parse metrics from log. Did it crash?")
                    writer.writerow([exp_id, "0.0", "0.0", "crash", json.dumps(config_dict)])
                    
                f.flush()
                
            except Exception as e:
                print(f"❌ Error during execution: {e}")
                writer.writerow([exp_id, "0.0", "0.0", "error", json.dumps(config_dict)])
                f.flush()

    print("\n✅ Autoresearch Sweep completed! Results logged in", args.output)

if __name__ == "__main__":
    main()
