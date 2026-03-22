"""Charades-STA dataset for VL-JEPA training."""

import os
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from vljepa.config import Config
from vljepa.utils import load_video_frames

QUERY = "What is happening in this video?"  # Fixed neutral query (paper Section 3.1)


class CharadesSTADataset(Dataset):
    """Charades-STA temporal grounding dataset.

    Annotation format: video_id start end##sentence
    Example: 3MSZA 24.3 30.4##person turn a light on

    Each sample provides:
      - frames   : video frames from the annotated temporal segment
      - query    : fixed neutral prompt ("What is happening in this video?")
      - caption  : ground-truth sentence (target for Y-Encoder)
    """

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
        if not os.path.exists(anno_file):
            raise FileNotFoundError(
                f"Annotation file not found: {anno_file}\n"
                "Download Charades-STA annotations from https://github.com/jiyanggao/TALL"
            )

        with open(anno_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("##")
                if len(parts) < 2:
                    continue

                meta = parts[0].strip().split()
                sentence = parts[1].strip()

                if len(meta) < 3:
                    continue

                video_id = meta[0]
                try:
                    start = float(meta[1])
                    end   = float(meta[2])
                except ValueError:
                    continue

                if end <= start:
                    continue

                video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "video_id":   video_id,
                    "start":      start,
                    "end":        end,
                    "caption":    sentence,
                })

    def video_ids(self) -> list[str]:
        """Return the list of unique video IDs in this dataset."""
        return list({s["video_id"] for s in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict | None:
        sample = self.samples[idx]

        frames = load_video_frames(
            sample["video_path"],
            start_sec=sample["start"],
            end_sec=sample["end"],
            num_frames=self.config.num_frames,
        )

        if frames is None or len(frames) == 0:
            return None

        return {
            "frames":   frames,             # list of np.ndarray (H, W, 3)
            "query":    QUERY,
            "caption":  sample["caption"],
            "video_id": sample["video_id"],
            "start":    sample["start"],
            "end":      sample["end"],
        }


def make_video_split(
    dataset: CharadesSTADataset,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split dataset indices by video_id to avoid leakage between train and val.

    Returns (train_indices, val_indices).
    """
    video_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        video_to_indices[s["video_id"]].append(i)

    video_ids = list(video_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_val = max(1, int(val_split * len(video_ids)))
    val_videos  = set(video_ids[:n_val])
    train_videos = set(video_ids[n_val:])

    train_indices = [i for vid in train_videos for i in video_to_indices[vid]]
    val_indices   = [i for vid in val_videos   for i in video_to_indices[vid]]

    return train_indices, val_indices


def collate_fn(batch: list[dict | None]) -> dict | None:
    """Filter None samples and collate into a batch dict."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    return {
        "frames":    [b["frames"]   for b in batch],
        "queries":   [b["query"]    for b in batch],
        "captions":  [b["caption"]  for b in batch],
        "video_ids": [b["video_id"] for b in batch],
        "starts":    [b["start"]    for b in batch],
        "ends":      [b["end"]      for b in batch],
    }