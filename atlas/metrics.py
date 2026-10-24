"""Region similarity J and contour accuracy F (DAVIS benchmark definitions)."""

import cv2
import numpy as np
from skimage.morphology import disk


def region_similarity(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """J: per-frame IoU of binary masks ``(T, H, W)``; 1 when both masks are empty."""
    gt, pred = gt.astype(bool), pred.astype(bool)
    inter = np.sum(gt & pred, axis=(-2, -1))
    union = np.sum(gt | pred, axis=(-2, -1))
    j = inter / np.maximum(union, 1)
    j[union == 0] = 1
    return j


def _boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide boundary map of a binary mask."""
    seg = mask.astype(bool)
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """Boundary F-measure of one frame, matching within ``bound_th`` of the image diagonal."""
    radius = np.ceil(bound_th * np.linalg.norm(pred.shape))
    pred_b, gt_b = _boundary(pred), _boundary(gt)
    kernel = disk(radius).astype(np.uint8)
    pred_dil = cv2.dilate(pred_b.astype(np.uint8), kernel)
    gt_dil = cv2.dilate(gt_b.astype(np.uint8), kernel)

    n_pred, n_gt = pred_b.sum(), gt_b.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    precision = np.sum(pred_b * gt_dil) / n_pred
    recall = np.sum(gt_b * pred_dil) / n_gt
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def contour_accuracy(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """F: per-frame boundary F-measure of binary masks ``(T, H, W)``."""
    return np.array([_boundary_f(p, g) for g, p in zip(gt, pred)])
