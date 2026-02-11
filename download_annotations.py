"""Download Charades-STA annotations."""

import os
import requests


ANNO_URLS = {
    "train": "https://raw.githubusercontent.com/mayu-ot/hidden-challenges-MR/main/data/raw/charades/charades_sta_train.txt",
    "test": "https://raw.githubusercontent.com/mayu-ot/hidden-challenges-MR/main/data/raw/charades/charades_sta_test.txt",
}

DATA_DIR = "./data"


def download_file(url: str, dest: str):
    """Download a file from url to dest."""
    print(f"Downloading {url}\n  -> {dest}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "w") as f:
        f.write(resp.text)
    lines = resp.text.strip().split("\n")
    print(f"  Saved ({len(lines)} lines)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for split, url in ANNO_URLS.items():
        dest = os.path.join(DATA_DIR, f"charades_sta_{split}.txt")
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Failed to download {split} from GitHub: {e}")
            print("Falling back to HuggingFace datasets...")
            _fallback_hf(split, dest)


def _fallback_hf(split: str, dest: str):
    """Download from HuggingFace and convert to .txt format.

    The lmms-lab/charades_sta dataset has only a 'test' split with columns:
    video (str, e.g. '3MSZA.mp4'), caption (str), timestamp (list[float, float])
    """
    try:
        from datasets import load_dataset

        # HF dataset only has 'test' split
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

        print(f"  Created {dest} from HuggingFace ({len(lines)} entries)")
    except Exception as e:
        print(f"  HuggingFace fallback also failed: {e}")
        print("  Please download annotations manually.")


if __name__ == "__main__":
    main()
