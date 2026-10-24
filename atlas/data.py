"""Benchmark metadata, keyframe sampling and ground-truth loading.

Every benchmark is a folder of JPEG frames per video plus a ``meta_expressions``
JSON listing the referring / reasoning expressions of each video. Ground truth
comes either as per-frame PNG annotations (Ref-YouTube-VOS train, used for
training) or as RLE mask dictionaries (evaluation sets).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np
from PIL import Image

from atlas.config import DATA_ROOT

# meta: expression file, frames: JPEG root, annotations: PNG root (training only),
# masks: RLE ground truth used by tools/evaluate.py (absent for server-evaluated splits).
BENCHMARKS: Dict[str, Dict[str, str]] = {
    "ytvos_train": {
        "meta": "ref-youtube-vos/meta_expressions/train/meta_expressions.json",
        "frames": "ref-youtube-vos/train/JPEGImages",
        "annotations": "ref-youtube-vos/train/Annotations",
    },
    "ytvos_valid": {
        "meta": "ref-youtube-vos/meta_expressions/valid/meta_expressions.json",
        "frames": "ref-youtube-vos/valid/JPEGImages",
    },
    "davis": {
        "meta": "davis17/meta_expressions/valid/meta_expressions.json",
        "frames": "davis17/valid/JPEGImages",
        "masks": "davis17/valid/mask_dict.pkl",
    },
    "reasonvos": {
        "meta": "ReasonVOS/meta_expressions_v2.json",
        "frames": "ReasonVOS/JPEGImages",
        "masks": "ReasonVOS/mask_dict_v2.pkl",
    },
    "revos": {
        "meta": "ReVOS/meta_expressions_valid_.json",
        "frames": "ReVOS/JPEGImages",
        "masks": "ReVOS/mask_dict.json",
    },
    "mevis": {  # MeViS valid_u (public ground truth)
        "meta": "mevis/valid_u/meta_expressions.json",
        "frames": "mevis/valid_u/JPEGImages",
        "masks": "mevis/valid_u/mask_dict.json",
    },
    "mevis_valid": {  # MeViS valid (evaluation server)
        "meta": "mevis/valid/meta_expressions.json",
        "frames": "mevis/valid/JPEGImages",
    },
}


def benchmark_path(dataset: str, key: str) -> Optional[str]:
    rel = BENCHMARKS[dataset].get(key)
    return os.path.join(DATA_ROOT, rel) if rel else None


@dataclass
class Sample:
    video: str          # video id (key of the meta file)
    exp_id: str         # expression id within the video
    expression: str
    frame_paths: List[str]  # all frames of the video in temporal order
    annotation: Dict = field(default_factory=dict)  # raw meta entry (obj_id / anno_id)

    @property
    def name(self) -> str:
        """File-system friendly identifier, used to name attention caches."""
        return f"{self.video}__{self.exp_id}".replace("/", "--")


def load_meta(dataset: str) -> Dict:
    """Return ``{video: {"expressions": {exp_id: {...}}, ...}}``."""
    with open(benchmark_path(dataset, "meta")) as f:
        videos = json.load(f)["videos"]
    if dataset == "ytvos_valid":
        # The valid meta file also lists the test videos; remove them (202 videos remain).
        with open(os.path.join(DATA_ROOT, "ref-youtube-vos/meta_expressions/test/meta_expressions.json")) as f:
            for video in json.load(f)["videos"]:
                videos.pop(video, None)
    if dataset == "davis":
        # DAVIS masks are indexed by a running annotation id over the sorted
        # videos / expression ids (same convention as the released mask_dict.pkl).
        anno_id = 0
        for video in sorted(videos):
            for exp_id in sorted(videos[video]["expressions"]):
                videos[video]["expressions"][exp_id]["anno_id"] = [anno_id]
                anno_id += 1
    return videos


def list_frames(dataset: str, video: str) -> List[str]:
    folder = os.path.join(benchmark_path(dataset, "frames"), video)
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(folder, n) for n in names]


def keyframe_indices(num_frames: int, num_keyframes: int) -> List[int]:
    """Uniformly sample ``num_keyframes`` frames, including the first and last (Sec. 3.2)."""
    k = min(num_keyframes, num_frames)
    return np.linspace(0, num_frames - 1, k).astype(int).tolist()


def in_shard(sample_name: str, shard_id: int, num_shards: int) -> bool:
    """Stable split of the samples over ``num_shards`` GPU workers."""
    digest = hashlib.md5(sample_name.encode()).hexdigest()
    return int(digest, 16) % num_shards == shard_id


def iter_samples(dataset: str, shard_id: int = 0, num_shards: int = 1) -> Iterator[Sample]:
    meta = load_meta(dataset)
    for video in sorted(meta):
        frames = None
        for exp_id, exp in meta[video]["expressions"].items():
            sample = Sample(video, str(exp_id), exp["exp"], [], exp)
            if not in_shard(sample.name, shard_id, num_shards):
                continue
            if frames is None:
                frames = list_frames(dataset, video)
            sample.frame_paths = frames
            yield sample


def load_training_masks(sample: Sample, frame_ids: List[int]) -> Optional[np.ndarray]:
    """Binary Ref-YouTube-VOS masks ``(T, H, W)`` of the referred object at ``frame_ids``.

    Returns ``None`` when the object is absent from every selected frame, so
    that such samples are skipped for training.
    """
    obj_id = int(sample.annotation["obj_id"])
    ann_dir = os.path.join(benchmark_path("ytvos_train", "annotations"), sample.video)
    masks = []
    for i in frame_ids:
        stem = os.path.splitext(os.path.basename(sample.frame_paths[i]))[0]
        png = os.path.join(ann_dir, stem + ".png")
        if os.path.isfile(png):
            masks.append((np.array(Image.open(png)) == obj_id).astype(np.uint8))
        else:
            w, h = Image.open(sample.frame_paths[i]).size
            masks.append(np.zeros((h, w), dtype=np.uint8))
    masks = np.stack(masks)
    return masks if masks.any() else None


def load_mask_dict(dataset: str):
    path = benchmark_path(dataset, "masks")
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path) as f:
        return json.load(f)
