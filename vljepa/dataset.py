"""Charades-STA dataset for VL-JEPA training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from vljepa.config import Config
from vljepa.utils import load_video_frames


class CharadesSTADataset(Dataset):
    """Dataset for Charades-STA temporal grounding.

    Annotation format: video_id start end##sentence
    Example: 3MSZA 24.3 30.4##person turn a light on

    For training, the query is a neutral prompt ("What is happening in this video?")
    and the target is the ground-truth caption.
    """

    NEUTRAL_QUERIES = [
        "What is happening in this video?",
        "Describe this video clip.",
        "What action is being performed?",
    ]

    def __init__(
        self,
        anno_file: str,
        videos_dir: str,
        config: Config,
        split: str = "train",
    ):
        self.videos_dir = videos_dir
        self.config = config
        self.split = split
        self.samples = []

        self._load_annotations(anno_file)

        if config.debug:
            self.samples = self.samples[: config.debug_samples]

        print(f"[{split}] Loaded {len(self.samples)} samples")

    def _load_annotations(self, anno_file: str):
        """Parse Charades-STA annotation file."""
        if not os.path.exists(anno_file):
            # Try loading from HuggingFace datasets
            self._load_from_hf()
            return

        with open(anno_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: video_id start end##sentence
                parts = line.split("##")
                if len(parts) < 2:
                    continue

                meta = parts[0].strip().split()
                sentence = parts[1].strip()

                if len(meta) < 3:
                    continue

                video_id = meta[0]
                start = float(meta[1])
                end = float(meta[2])

                video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    self.samples.append({
                        "video_path": video_path,
                        "video_id": video_id,
                        "start": start,
                        "end": end,
                        "caption": sentence,
                    })

    def _load_from_hf(self):
        """Fallback: load annotations from HuggingFace datasets."""
        try:
            from datasets import load_dataset

            print("Loading annotations from HuggingFace (lmms-lab/charades_sta)...")
            ds = load_dataset("lmms-lab/charades_sta", split="test")

            for item in ds:
                video_id = item.get("video_id") or item.get("video", "")
                start = float(item.get("start", 0))
                end = float(item.get("end", 10))
                caption = item.get("query", "") or item.get("description", "")

                video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
                if os.path.exists(video_path) and caption:
                    self.samples.append({
                        "video_path": video_path,
                        "video_id": video_id,
                        "start": start,
                        "end": end,
                        "caption": caption,
                    })

        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Please download annotations manually. See download_annotations.py")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict | None:
        sample = self.samples[idx]

        # Load frames from the annotated temporal segment
        frames = load_video_frames(
            sample["video_path"],
            start_sec=sample["start"],
            end_sec=sample["end"],
            num_frames=self.config.num_frames,
        )

        if frames is None or len(frames) == 0:
            return None

        # Use a neutral query for training
        # (VL-JEPA learns to predict the target caption embedding from video + query)
        query_idx = idx % len(self.NEUTRAL_QUERIES)
        query = self.NEUTRAL_QUERIES[query_idx]

        return {
            "frames": frames,           # list of numpy arrays (H, W, 3)
            "query": query,             # neutral text query
            "caption": sample["caption"],  # target caption
            "video_id": sample["video_id"],
            "start": sample["start"],
            "end": sample["end"],
        }


def collate_fn(batch: list[dict | None]) -> dict | None:
    """Custom collate that filters out None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    return {
        "frames": [b["frames"] for b in batch],
        "queries": [b["query"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "video_ids": [b["video_id"] for b in batch],
        "starts": [b["start"] for b in batch],
        "ends": [b["end"] for b in batch],
    }
