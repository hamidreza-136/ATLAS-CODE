"""Stage 3 - Segmentation and Propagation (Sec. 3.4).

Given the localization maps ``P_k`` of the ``T_s`` keyframes:

1. Seed each keyframe with one positive point at the argmax of ``P_k`` (Eq. 11)
   and let SAM2 turn it into a seed mask.
2. Remove duplicate seeds: starting from the earliest keyframe, propagate its
   seed mask over the keyframes; a later seed whose mask overlaps the
   propagated mask at its keyframe with IoU > tau is already tracked and is
   dropped. Repeat from the earliest remaining seed. Each kept seed yields one
   candidate tracklet.
3. Select the tracklet whose (attention-resolution) masks correlate best with
   ``P_k``, averaging the per-keyframe Pearson correlation (Eq. 12). If the
   best score is below 0.1 the referred object is considered absent.
4. Propagate the selected seed over the full video.

Candidate tracklets are propagated over the keyframes only; only the selected
one is propagated over every frame of the video.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np
import sam2.sam2_video_predictor
import sam2.utils.misc
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor

DUPLICATE_IOU = 0.8   # tau
MIN_PEARSON = 0.1     # below this no reliable tracklet exists

# Silence SAM2's per-call progress bars (one per propagation / frame loading).
sam2.sam2_video_predictor.tqdm = sam2.utils.misc.tqdm = lambda it, **kwargs: it


def build_sam2(config: str, checkpoint: str, device: str = "cuda"):
    return build_sam2_video_predictor(config, checkpoint, device=device)


def init_state(predictor, frame_paths: List[str], offload_video_to_cpu: bool = False):
    """SAM2 inference state over an arbitrary, ordered list of frames."""
    with tempfile.TemporaryDirectory() as tmp:
        # SAM2 reads a folder of frames named by their index.
        for i, path in enumerate(frame_paths):
            os.symlink(os.path.abspath(path), os.path.join(tmp, f"{i:05d}.jpg"))
        return predictor.init_state(video_path=tmp, offload_video_to_cpu=offload_video_to_cpu)


def seed_points(prob: torch.Tensor) -> torch.Tensor:
    """Eq. (11): argmax location of each map ``(T, h, w)`` as normalized ``(x, y)``."""
    _, h, w = prob.shape
    idx = prob.flatten(1).argmax(dim=1)
    rows, cols = idx // w, idx % w
    return torch.stack([(cols + 0.5) / w, (rows + 0.5) / h], dim=1).cpu()


def point_to_mask(predictor, state, frame_idx: int, point: torch.Tensor) -> torch.Tensor:
    """SAM2 mask ``(H, W)`` for a single positive point given in normalized coordinates."""
    predictor.reset_state(state)
    _, _, logits = predictor.add_new_points_or_box(
        state, frame_idx=frame_idx, obj_id=1, points=point[None], labels=np.array([1], np.int32),
        normalize_coords=False,
    )
    return logits[0, 0] > 0


def propagate(predictor, state, frame_idx: int, mask: torch.Tensor, binarize: bool = False) -> torch.Tensor:
    """Propagate a mask prompt forward and backward over all frames of ``state``.

    Returns logits ``(T, H, W)``, or boolean masks on the CPU if ``binarize``.
    """
    predictor.reset_state(state)
    predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=1, mask=mask)
    out = {}
    for reverse in (False, True):
        for f, _, logits in predictor.propagate_in_video(state, start_frame_idx=frame_idx, reverse=reverse):
            out[f] = (logits[0, 0] > 0).cpu() if binarize else logits[0, 0]
    return torch.stack([out[f] for f in range(state["num_frames"])])


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 0.0


def build_tracklets(predictor, keyframe_paths: List[str], prob: torch.Tensor) -> List[dict]:
    """Seed every keyframe and keep one tracklet per distinct object hypothesis."""
    state = init_state(predictor, keyframe_paths)
    points = seed_points(prob)
    tracklets = []
    for k in range(len(keyframe_paths)):
        seed = point_to_mask(predictor, state, k, points[k])
        if not seed.any():
            continue  # SAM2 found no object at this point
        if any(mask_iou(seed, t["logits"][k] > 0) > DUPLICATE_IOU for t in tracklets):
            continue  # the object of this seed is already tracked
        tracklets.append({"keyframe": k, "seed": seed, "logits": propagate(predictor, state, k, seed)})
    return tracklets


def tracklet_scores(tracklets: List[dict], prob: torch.Tensor) -> torch.Tensor:
    """Eq. (12): mean over keyframes of the spatial Pearson correlation with ``P_k``.

    The correlation of a keyframe is taken as 0 when either map is constant
    there (e.g. the tracklet is empty on that keyframe).
    """
    t, h, w = prob.shape
    p = prob.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    scores = []
    for tr in tracklets:
        # V_m: tracklet probabilities downsampled to the attention resolution.
        v = F.adaptive_avg_pool2d(tr["logits"].sigmoid()[None], (h, w))[0].flatten(1)
        v = v - v.mean(dim=1, keepdim=True)
        denom = v.norm(dim=1) * p.norm(dim=1)
        rho = torch.where(denom > 0, (v * p).sum(dim=1) / denom.clamp_min(1e-12), torch.zeros_like(denom))
        scores.append(rho.mean())
    return torch.stack(scores)


@torch.inference_mode()
def segment_video(predictor, frame_paths: List[str], keyframe_ids: List[int], prob: torch.Tensor) -> Optional[np.ndarray]:
    """Binary masks ``(num_frames, H, W)`` of the referred object, or ``None`` if not found."""
    tracklets = build_tracklets(predictor, [frame_paths[i] for i in keyframe_ids], prob)
    if not tracklets:
        return None
    scores = tracklet_scores(tracklets, prob.to(tracklets[0]["logits"].device))
    best = int(scores.argmax())
    if scores[best] < MIN_PEARSON:
        return None
    selected = tracklets[best]
    state = init_state(predictor, frame_paths, offload_video_to_cpu=True)
    return propagate(predictor, state, keyframe_ids[selected["keyframe"]], selected["seed"], binarize=True).numpy()
