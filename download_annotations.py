"""Download Charades-STA annotations."""

import os

DATA_DIR = "./data"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    _extract_hf("test", os.path.join(DATA_DIR, "charades_sta_test.txt"))

def _extract_hf(split: str, dest: str):
    """Download from HuggingFace and convert to .txt format.

    The lmms-lab/charades_sta dataset has only a 'test' split with columns:
    video (str, e.g. '3MSZA.mp4'), caption (str), timestamp (list[float, float])
    """
    try:
        from datasets import load_dataset
        
        # We extract what we can from HF
        ds = load_dataset("lmms-lab/charades_sta", split="test")
        lines = []
        for item in ds:
            vid = item.get("video", "")
            if vid.endswith(".mp4"):
                vid = vid[:-4]  # Remove .mp4 extension
            caption = item.get("caption", "")
            ts = item.get("timestamp", [0, 10])
            start, end = float(ts[0]), float(ts[1])
            if vid and caption:
                lines.append(f"{vid} {start} {end}##{caption}")

        with open(dest, "w") as f:
            f.write("\n".join(lines))

        print(f"✓ Created {dest} from HuggingFace ({len(lines)} entries)")
    except Exception as e:
        print(f"HuggingFace dataset extraction failed: {e}")
        print("Please download annotations manually.")

if __name__ == "__main__":
    main()
