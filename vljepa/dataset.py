"""Charades-STA dataset for VL-JEPA training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from vljepa.config import Config
from vljepa.utils import load_video_frames

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# HF Storage bucket configuration - read from env or use default
HF_STORAGE_BUCKET = os.getenv("HF_BUCKET_ID", "max044/charades-sta-storage")


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
                
                # If streaming/lazy loading is enabled, we add even if not local
                if os.path.exists(video_path) or self.config.use_hf_storage:
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
                if (os.path.exists(video_path) or self.config.use_hf_storage) and caption:
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
        video_path = sample["video_path"]

        # ── Lazy Loading from HF Storage Bucket ────────────────────────────
        # Check if we should use HF Storage (XET) for lazy loading
        use_hf_storage = (
            not os.path.exists(video_path) and self.config.use_hf_storage
        )

        if use_hf_storage and HAS_HF_HUB:
            try:
                # HF Storage bucket with XET (fast)
                video_path = hf_hub_download(
                    repo_id=HF_STORAGE_BUCKET,
                    filename=f"Charades_v1_480/{sample['video_id']}.mp4",
                    repo_type="dataset",
                    local_dir=self.videos_dir,
                    local_dir_use_symlinks=False,
                    token=os.getenv('HF_TOKEN'),
                )
            except Exception as e:
                print(f"Error downloading {sample['video_id']} from HF Storage: {e}")
                # Fallback to HF Dataset
                try:
                    video_path = hf_hub_download(
                        repo_id=getattr(self.config, 'hf_dataset_id', 'max044/Charades_v1_480'),
                        filename=f"{sample['video_id']}.mp4",
                        repo_type="dataset",
                        local_dir=self.videos_dir,
                        local_dir_use_symlinks=False,
                        token=os.getenv('HF_TOKEN'),
                    )
                except Exception as e2:
                    print(f"Error downloading {sample['video_id']}: {e2}")
                    return None

        # Load frames from the annotated temporal segment
        # If use_regression is enabled, we occasionally sample a larger window
        # to train the regression head.
        start_sec = sample["start"]
        end_sec = sample["end"]
        
        if self.split == "train" and getattr(self.config, "use_regression", False):
            # Jitter window: +/- 20% of duration, or fixed 2s
            dur = end_sec - start_sec
            jitter = min(2.0, dur * 0.2)
            
            # Randomly shift start/end
            win_start = max(0, start_sec - np.random.uniform(0, jitter))
            win_end = end_sec + np.random.uniform(0, jitter)
            
            # These are the boundaries of the frames we load
            load_start, load_end = win_start, win_end
        else:
            load_start, load_end = start_sec, end_sec

        frames = load_video_frames(
            video_path,
            start_sec=load_start,
            end_sec=load_end,
            num_frames=self.config.num_frames,
        )

        if frames is None or len(frames) == 0:
            return None

        # Calculate regression targets relative to the loaded window
        # o_start = (gt_start - win_start) / win_duration
        # o_end = (gt_end - win_start) / win_duration
        win_dur = load_end - load_start
        offset_start = (start_sec - load_start) / win_dur
        offset_end = (end_sec - load_start) / win_dur

        # Use a neutral query for training
        # (VL-JEPA learns to predict the target caption embedding from video + query)
        query_idx = idx % len(self.NEUTRAL_QUERIES)
        query = self.NEUTRAL_QUERIES[query_idx]

        return {
            "frames": frames,           # list of numpy arrays (H, W, 3)
            "query": query,             # neutral text query
            "caption": sample["caption"],  # target caption
            "video_id": sample["video_id"],
            "start": start_sec,
            "end": end_sec,
            "offset_targets": [offset_start, offset_end]
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
        "offset_targets": [b["offset_targets"] for b in batch],
    }