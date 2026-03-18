import os
import time
import argparse
import subprocess
from dotenv import load_dotenv
from vastai_sdk import VastAI

def main():
    parser = argparse.ArgumentParser(description="VL-JEPA Vast.ai Instance Launcher")
    parser.add_argument("--gpu", type=str, default="RTX 4090", help="GPU name to search (e.g., 'RTX 4090', 'A100', 'H100')")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs required")
    parser.add_argument("--disk", type=int, default=50, help="Disk space in GB")
    parser.add_argument("--image", type=str, default="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime", help="Docker image to use")
    parser.add_argument("--script", type=str, default="scripts/train_cloud.sh", help="Script to run after bootstrap (default: scripts/train_cloud.sh)")
    parser.add_argument("--no-run", action="store_true", help="Only run bootstrap, do not start any script")
    parser.add_argument("--no-dataset", action="store_true", help="Skip downloading the 180GB video dataset during bootstrap")
    parser.add_argument("--dry-run", action="store_true", help="Search for instances without launching")
    parser.add_argument("--order", type=str, default="dph", help="Search order (e.g., 'dph' for dollars per hour)")
    
    args, unknown = parser.parse_known_args()
    
    extra_args_str = " ".join(unknown)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("VASTAI_API_KEY")
    if not api_key:
        print("❌ Error: VASTAI_API_KEY not found in .env file.")
        return

    sdk = VastAI(api_key=api_key)
    
    print(f"🔍 Searching for available {args.gpu} instances...")
    
    # Construct search query
    query = f"gpu_name={args.gpu.replace(' ', '_')} num_gpus={args.num_gpus} disk_space>={args.disk} rentable=True rented=False"
    
    offers = sdk.search_offers(query=query, order=args.order)
    
    if not offers:
        print(f"❌ No offers found matching: {query}")
        return

    # Select the best offer (first one since it's ordered by price)
    best_offer = offers[0]
    offer_id = best_offer['id']
    price = best_offer['dph_total']
    machine_id = best_offer['machine_id']
    
    print(f"✅ Found best offer: ID {offer_id} on Machine {machine_id} at ${price:.3f}/hr")
    
    if args.dry_run:
        print("ℹ️ Dry run complete. No instance launched.")
        return
    
    print(f"🚀 Launching instance {offer_id} with {args.disk}GB disk...")
    result = sdk.create_instance(id=offer_id, image=args.image, label="vl-jepa-training", disk=args.disk)
    
    if 'new_contract' not in result:
        print(f"❌ Error launching instance: {result}")
        return
    
    new_instance_id = result['new_contract']
    print(f"⌛ Instance {new_instance_id} created. Waiting for it to be ready...")
    
    # Wait for the instance to be reachable via SSH
    max_retries = 30
    retry_delay = 10
    instance_info = None
    
    for i in range(max_retries):
        instances = sdk.show_instances()
        instance_info = next((inst for inst in instances if inst['id'] == new_instance_id), None)
        
        if instance_info and instance_info.get('ssh_host'):
            print(f"✨ Instance is ready! Host: {instance_info['ssh_host']}, Port: {instance_info['ssh_port']}")
            break
        
        print(f"  ({i+1}/{max_retries}) Still waiting...")
        time.sleep(retry_delay)
    else:
        print("❌ Timeout waiting for instance to be ready.")
        return

    ssh_host = instance_info['ssh_host']
    ssh_port = instance_info['ssh_port']
    ssh_user = "root" # Vast instances usually use root
    ssh_prefix = f"ssh -p {ssh_port} {ssh_user}@{ssh_host} -o StrictHostKeyChecking=no"
    
    # ── Step 2: Upload .env and bootstrap ──────────────────
    print("📤 Uploading .env and project configuration...")
    
    # Create a command to clone and bootstrap
    # Note: We use -o StrictHostKeyChecking=no to avoid manual confirmation
    # We also start the specified script automatically if requested
    env_content = open('.env', encoding='utf-8').read().replace('$', '\\$').replace('"', '\\"')
    
    if args.no_dataset:
        env_content += "\nDOWNLOAD_DATASET=false\n"
    
    if args.no_run:
        remote_cmd = (
            f"git clone https://github.com/max044/vl-jepa.git ~/vl-jepa || (cd ~/vl-jepa && git pull) && "
            f"cd ~/vl-jepa && "
            f"printf \"{env_content}\" > .env && "
            f"bash scripts/bootstrap.sh"
        )
        print("🛠️  Running bootstrap on remote instance...")
        print("ℹ️  This will take a few minutes to install dependencies.")
    else:
        remote_cmd = (
            f"git clone https://github.com/max044/vl-jepa.git ~/vl-jepa || (cd ~/vl-jepa && git pull) && "
            f"cd ~/vl-jepa && "
            f"printf \"{env_content}\" > .env && "
            f"bash scripts/bootstrap.sh && "
            f"bash {args.script} {extra_args_str}"
        )
        print(f"🛠️  Running bootstrap and starting {args.script} {extra_args_str} on remote instance...")
        print("ℹ️  This will take a few minutes to install dependencies.")
        print("ℹ️  Once the script starts, you can safely Ctrl+C; it will continue in the background.")
    
    try:
        # Use subprocess.run without check=True to allow the user to Ctrl+C without a stack trace
        subprocess.run(f"{ssh_prefix} \"{remote_cmd}\"", shell=True)
    except KeyboardInterrupt:
        print("\n👋 Disconnected from remote. Process continues in the background.")
    except Exception as e:
        print(f"❌ Error during remote execution: {e}")
        return

    print("\n✅ Setup complete!")
    
    if not args.no_run:
        script_log = "train_cloud.log" if "train" in args.script else "run.log"
        print(f"🔗 To monitor progress, run:")
        print(f"   {ssh_prefix} \"tail -f ~/vl-jepa/{script_log}\"")
        
    print(f"🔗 SSH Login: {ssh_prefix}")

if __name__ == "__main__":
    main()
