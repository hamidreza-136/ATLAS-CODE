"""Paths and paper defaults shared by every ATLAS entry point."""

import os

# Root folders (relative to the ATLAS directory unless overridden).
DATA_ROOT = os.environ.get("ATLAS_DATA_ROOT", "datasets")
CKPT_ROOT = os.environ.get("ATLAS_CKPT_ROOT", "ckpts")

# Frozen LVLMs evaluated in the paper: short name -> Hugging Face repository.
LVLMS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl3-8b": "OpenGVLab/InternVL3-8B-hf",
}

# Frozen SAM2 model (Sec. 4, "Models").
SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.join(CKPT_ROOT, "sam2_hiera_large.pt")

# Sec. 3.2 / 4: sampled keyframes and visual-token budgets.
NUM_KEYFRAMES = 16
FRAME_TOKENS = 1024   # frame branch (one forward pass per keyframe)
VIDEO_TOKENS = 64     # video branch (one forward pass over all keyframes)


def lvlm_path(name: str) -> str:
    """Local checkpoint folder if it was downloaded into CKPT_ROOT, else the HF repo id."""
    repo = LVLMS[name]
    local = os.path.join(CKPT_ROOT, repo.split("/")[-1])
    return local if os.path.isdir(local) else repo
