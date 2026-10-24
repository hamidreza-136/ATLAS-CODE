"""Stage 1 - Attention Extraction (Sec. 3.2).

For every (video, expression) sample, uniformly sample ``T_s`` keyframes, wrap
the expression in the object extraction prompt, and save one ``.pt`` file with

  frame_attn  (T_s, C, h_f, w_f)  one LVLM pass per keyframe, high token budget
  video_attn  (T_s, C, h_v, w_v)  one LVLM pass over all keyframes, low budget
  gt_frame    (T_s, h_f, w_f)     ground truth at the frame-attention grid
                                  (Ref-YouTube-VOS train only, for training)

Run one process per GPU with ``--num-shards`` / ``--shard-id``; finished
samples are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from atlas.config import LVLMS, NUM_KEYFRAMES
from atlas.data import BENCHMARKS, iter_samples, keyframe_indices, load_training_masks
from atlas.lvlm import load_extractor
from atlas.prompts import build_prompt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lvlm", required=True, choices=sorted(LVLMS))
    p.add_argument("--dataset", required=True, choices=sorted(BENCHMARKS))
    p.add_argument("--out", required=True, help="Output folder for the attention caches.")
    p.add_argument("--num-keyframes", type=int, default=NUM_KEYFRAMES)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--attn-implementation", default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    extractor = load_extractor(args.lvlm, attn_implementation=args.attn_implementation)
    print(f"{args.lvlm}: layers {extractor.first_layer}-{extractor.num_layers - 1}, "
          f"{extractor.num_heads} heads -> C={extractor.channels}")

    samples = iter_samples(args.dataset, args.shard_id, args.num_shards)
    for sample in tqdm(samples, desc=f"{args.dataset} [shard {args.shard_id}/{args.num_shards}]"):
        out_path = os.path.join(args.out, sample.name + ".pt")
        if os.path.exists(out_path):
            continue
        ids = keyframe_indices(len(sample.frame_paths), args.num_keyframes)
        paths = [sample.frame_paths[i] for i in ids]
        gt = None
        if args.dataset == "ytvos_train":
            gt = load_training_masks(sample, ids)
            if gt is None:  # referred object never visible in the keyframes
                continue

        prompt = build_prompt(sample.expression)
        frame_attn = torch.stack([extractor.frame_attention(p, prompt) for p in paths])
        video_attn = extractor.video_attention(paths, prompt)

        record = {
            "frame_attn": frame_attn.half().cpu(),
            "video_attn": video_attn.half().cpu(),
            "keyframe_indices": ids,
            "frame_paths": paths,
            "video": sample.video,
            "exp_id": sample.exp_id,
            "expression": sample.expression,
            "lvlm": args.lvlm,
        }
        if gt is not None:
            gt = torch.from_numpy(gt).float()[:, None]
            record["gt_frame"] = F.interpolate(gt, size=frame_attn.shape[-2:], mode="nearest")[:, 0]
        torch.save(record, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)  # never leave half-written caches


if __name__ == "__main__":
    main()
