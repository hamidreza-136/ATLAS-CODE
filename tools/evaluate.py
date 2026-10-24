"""Compute J, F and J&F of predicted masks against the benchmark ground truth.

Per expression, J and F are averaged over all frames of the video; the
reported numbers are averages over expressions. For ReVOS, scores are also
given for referring and reasoning expressions, and "overall" is their mean.
Writes ``<pred>.json`` (summary) and ``<pred>.csv`` (per expression).
Ref-YouTube-VOS valid and MeViS valid are evaluated on the official servers.
"""

import argparse
import csv
import json
import multiprocessing
import os

import cv2
import numpy as np
from pycocotools import mask as cocomask

from atlas.data import BENCHMARKS, load_mask_dict, load_meta
from atlas.metrics import contour_accuracy, region_similarity

MASKS = None  # ground-truth RLE dictionary, shared with the worker processes
cv2.setNumThreads(1)  # one thread per worker process


def score(job):
    pred_dir, video, exp_id, frames, anno_ids = job
    gt, pred = [], []
    for i, frame in enumerate(frames):
        p = cv2.imread(os.path.join(pred_dir, video, exp_id, frame + ".png"), cv2.IMREAD_GRAYSCALE)
        if p is None:
            return video, exp_id, 0.0, 0.0  # missing prediction counts as a failure
        g = np.zeros(p.shape, dtype=np.uint8)
        for a in anno_ids:
            rle = MASKS[str(a)][i]
            if rle:
                g += cocomask.decode(rle)
        gt.append(g > 0)
        pred.append(p > 0)
    gt, pred = np.stack(gt), np.stack(pred)
    return video, exp_id, region_similarity(gt, pred).mean(), contour_accuracy(gt, pred).mean()


def summarize(rows):
    j = 100 * float(np.mean([r["J"] for r in rows]))
    f = 100 * float(np.mean([r["F"] for r in rows]))
    return {"J": j, "F": f, "J&F": (j + f) / 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=[d for d in BENCHMARKS if "masks" in BENCHMARKS[d]])
    p.add_argument("--pred", required=True, help="Folder written by tools/segment_and_propagate.py.")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    global MASKS
    args = parse_args()
    meta = load_meta(args.dataset)
    MASKS = load_mask_dict(args.dataset)
    jobs = [
        (args.pred, video, exp_id, meta[video]["frames"], exp["anno_id"])
        for video in meta for exp_id, exp in meta[video]["expressions"].items()
    ]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:  # workers share MASKS
        results = pool.map(score, jobs, chunksize=4)

    rows = []
    for video, exp_id, j, f in results:
        exp = meta[video]["expressions"][exp_id]
        rows.append({"video": video, "exp_id": exp_id, "expression": exp["exp"], "J": j, "F": f,
                     "type_id": exp.get("type_id")})

    if args.dataset == "revos":  # type_id 0: referring, 1: reasoning
        groups = {"referring": [r for r in rows if r["type_id"] == 0],
                  "reasoning": [r for r in rows if r["type_id"] == 1]}
        summary = {name: summarize(g) for name, g in groups.items() if g}
        summary["overall"] = {k: float(np.mean([s[k] for s in summary.values()])) for k in ("J", "F", "J&F")}
    else:
        summary = {"overall": summarize(rows)}

    out = args.pred.rstrip("/")
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split, s in summary.items():
        print(f"{args.dataset} {split:9s}  J={s['J']:.1f}  F={s['F']:.1f}  J&F={s['J&F']:.1f}")


if __name__ == "__main__":
    main()
