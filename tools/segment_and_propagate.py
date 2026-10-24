"""Stages 2 + 3 at test time: attention adaptation, then segmentation and propagation.

For every sample of a benchmark, the trained adapter turns the cached
attention into localization maps ``P_k`` (Sec. 3.3), which seed and select a
SAM2 tracklet that is propagated over the whole video (Sec. 3.4). Masks are
written as ``<out>/<video>/<exp_id>/<frame>.png`` (0 / 255), the layout
expected by ``tools/evaluate.py`` and by the evaluation servers.

Run one process per GPU with ``--num-shards`` / ``--shard-id``; finished
samples are skipped.
"""

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from atlas.adapter import adapter_input, load_adapter
from atlas.config import SAM2_CHECKPOINT, SAM2_CONFIG
from atlas.data import BENCHMARKS, iter_samples, keyframe_indices
from atlas.segmentation import build_sam2, segment_video


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(BENCHMARKS))
    p.add_argument("--cache", required=True, help="Attention caches of this dataset.")
    p.add_argument("--adapter", required=True, help="Checkpoint written by tools/train_adapter.py.")
    p.add_argument("--out", required=True, help="Output folder for the predicted masks.")
    p.add_argument("--sam2-config", default=SAM2_CONFIG)
    p.add_argument("--sam2-checkpoint", default=SAM2_CHECKPOINT)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    adapter, grid = load_adapter(args.adapter)
    predictor = build_sam2(args.sam2_config, args.sam2_checkpoint)

    samples = iter_samples(args.dataset, args.shard_id, args.num_shards)
    for sample in tqdm(samples, desc=f"{args.dataset} [shard {args.shard_id}/{args.num_shards}]"):
        out_dir = os.path.join(args.out, sample.video, sample.exp_id)
        names = [os.path.splitext(os.path.basename(p))[0] for p in sample.frame_paths]
        if all(os.path.exists(os.path.join(out_dir, n + ".png")) for n in names):
            continue

        rec = torch.load(os.path.join(args.cache, sample.name + ".pt"), map_location="cpu")
        num_kf = rec["frame_attn"].shape[0]
        ids = rec.get("keyframe_indices") or keyframe_indices(len(sample.frame_paths), num_kf)
        with torch.no_grad():
            x = adapter_input(rec["frame_attn"], rec["video_attn"], grid)[None].cuda()
            prob = adapter(x).sigmoid()                    # P_k: (T_s, H, W)

        masks = segment_video(predictor, sample.frame_paths, ids, prob)
        if masks is None:  # no reliable tracklet: empty prediction
            h, w = cv2.imread(sample.frame_paths[0], cv2.IMREAD_GRAYSCALE).shape
            masks = np.zeros((len(names), h, w), dtype=bool)

        os.makedirs(out_dir, exist_ok=True)
        for name, mask in zip(names, masks):
            cv2.imwrite(os.path.join(out_dir, name + ".png"), mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
