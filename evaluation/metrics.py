"""
Metrics Module
--------------
Covers every standard metric for person Re-ID & tracking:

  Re-ID metrics
  ─────────────
  • CMC  (Cumulative Matching Characteristics) – Rank-1/5/10/20
  • mAP  (mean Average Precision)
  • mINP (mean Inverse Negative Penalty) – focuses on hardest positives

  Embedding quality
  ─────────────────
  • Intra-class distance  (should be small)
  • Inter-class distance  (should be large)
  • Silhouette score

  Tracking metrics (MOT)
  ──────────────────────
  • MOTA  (Multi-Object Tracking Accuracy)
  • MOTP  (Multi-Object Tracking Precision)
  • IDF1  (ID F1 Score)
  • ID switches
  • Fragmentation
  • Precision / Recall / F1

  Detection metrics
  ─────────────────
  • Precision, Recall, F1 at IoU thresholds
  • mAP@50, mAP@50-95
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    average_precision_score,
    silhouette_score,
    confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — Re-ID Metrics
# ══════════════════════════════════════════════════════════════════

def compute_distance_matrix(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    metric: str = "cosine",          # "cosine" or "euclidean"
) -> np.ndarray:
    """
    Returns (Q, G) pairwise distance matrix.
    For cosine distance, smaller value = more similar.
    """
    return cdist(query_feats, gallery_feats, metric=metric)


def compute_ap(
    good_matches: np.ndarray,        # boolean array, sorted by rank
) -> float:
    """Compute Average Precision for a single query."""
    num_pos = good_matches.sum()
    if num_pos == 0:
        return 0.0

    cumsum = np.cumsum(good_matches)
    precision_at_k = cumsum / (np.arange(len(good_matches)) + 1)
    ap = (precision_at_k * good_matches).sum() / num_pos
    return float(ap)


def compute_cmc_map(
    dist_matrix: np.ndarray,
    query_pids:   np.ndarray,
    gallery_pids: np.ndarray,
    query_cids:   np.ndarray,
    gallery_cids: np.ndarray,
    max_rank: int = 20,
    remove_same_camera: bool = True,       # standard DukeMTMC protocol
) -> Dict[str, float]:
    """
    Core Re-ID evaluation following the standard Market-1501 / DukeMTMC protocol.

    Returns
    -------
    dict with keys: rank-1, rank-5, rank-10, rank-20, mAP, mINP
    """
    num_query   = dist_matrix.shape[0]
    num_gallery = dist_matrix.shape[1]
    max_rank    = min(max_rank, num_gallery)

    all_cmc  = np.zeros(max_rank, dtype=np.float32)
    all_ap   = []
    all_inp  = []

    for q_idx in range(num_query):
        q_pid = query_pids[q_idx]
        q_cid = query_cids[q_idx]

        # ── Build masks ──────────────────────────────────────────
        order    = np.argsort(dist_matrix[q_idx])
        g_pids_s = gallery_pids[order]
        g_cids_s = gallery_cids[order]

        if remove_same_camera:
            # Remove gallery samples from the same camera as query
            keep = ~((g_pids_s == q_pid) & (g_cids_s == q_cid))
        else:
            keep = np.ones(num_gallery, dtype=bool)

        orig_cmc = (g_pids_s == q_pid)[keep]

        if not orig_cmc.any():
            continue

        cmc = orig_cmc[:max_rank].cumsum()
        cmc[cmc > 1] = 1
        all_cmc += cmc

        # AP
        ap = compute_ap(orig_cmc.astype(float))
        all_ap.append(ap)

        # mINP
        pos_indices = np.where(orig_cmc)[0]
        if len(pos_indices):
            hardest = pos_indices[-1] + 1             # 1-indexed rank of last positive
            inp     = orig_cmc[:hardest].sum() / hardest
            all_inp.append(inp)

    num_valid = len(all_ap)
    if num_valid == 0:
        raise ValueError("No valid queries found. Check PID/CID arrays.")

    cmc_scores = all_cmc / num_valid

    results = {
        "rank-1":  round(cmc_scores[0] * 100, 2),
        "rank-5":  round(cmc_scores[4] * 100, 2) if max_rank >= 5  else None,
        "rank-10": round(cmc_scores[9] * 100, 2) if max_rank >= 10 else None,
        "rank-20": round(cmc_scores[19] * 100, 2) if max_rank >= 20 else None,
        "mAP":     round(np.mean(all_ap) * 100, 2),
        "mINP":    round(np.mean(all_inp) * 100, 2),
    }
    return results


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — Embedding Quality Metrics
# ══════════════════════════════════════════════════════════════════

def embedding_quality(
    features: np.ndarray,
    labels:   np.ndarray,
    sample_size: int = 2000,
) -> Dict[str, float]:
    """
    Computes intra/inter-class distances and silhouette score
    to measure how well embeddings are separated.

    Higher inter-class + lower intra-class = better Re-ID.
    """
    if len(features) > sample_size:
        idx      = np.random.choice(len(features), sample_size, replace=False)
        features = features[idx]
        labels   = labels[idx]

    unique_labels = np.unique(labels)
    intra_dists, inter_dists = [], []

    for lbl in unique_labels:
        mask = labels == lbl
        same = features[mask]
        diff = features[~mask]

        if len(same) > 1:
            d = cdist(same, same, "cosine")
            intra_dists.append(d[np.triu_indices_from(d, k=1)].mean())

        if len(same) and len(diff):
            d = cdist(same, diff, "cosine")
            inter_dists.append(d.mean())

    silhouette = (
        silhouette_score(features, labels, metric="cosine")
        if len(unique_labels) > 1 and len(features) > 1 else 0.0
    )

    return {
        "intra_class_dist": round(np.mean(intra_dists), 4),
        "inter_class_dist": round(np.mean(inter_dists), 4),
        "dist_ratio":       round(np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8), 4),
        "silhouette_score": round(float(silhouette), 4),
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — Tracking Metrics (MOT)
# ══════════════════════════════════════════════════════════════════

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0]);  ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]);  yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter)


def compute_mot_metrics(
    gt_frames:   Dict[int, List[Tuple]],   # {frame_id: [(id, x1,y1,x2,y2), ...]}
    pred_frames: Dict[int, List[Tuple]],   # same format
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes MOTA, MOTP, IDF1, ID Switches, Precision, Recall, F1.

    This is a simplified single-class MOT evaluator.
    For production, use the `motmetrics` library.
    """
    TP = FP = FN = 0
    total_iou = 0.0
    num_matches = 0
    id_switches = 0

    prev_gt_to_pred: Dict[int, int] = {}          # {gt_id -> pred_id} from last frame

    all_frames = sorted(set(gt_frames) | set(pred_frames))

    for fid in all_frames:
        gts   = gt_frames.get(fid,   [])          # [(id,x1,y1,x2,y2), ...]
        preds = pred_frames.get(fid, [])

        matched_gt   = set()
        matched_pred = set()
        gt_to_pred   = {}

        # Greedy matching by IoU
        iou_matrix = np.zeros((len(gts), len(preds)))
        for gi, gt in enumerate(gts):
            for pi, pred in enumerate(preds):
                iou_matrix[gi, pi] = compute_iou(
                    np.array(gt[1:]), np.array(pred[1:])
                )

        for _ in range(min(len(gts), len(preds))):
            if iou_matrix.max() < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            total_iou   += iou_matrix[gi, pi]
            num_matches += 1
            matched_gt.add(gi);  matched_pred.add(pi)
            gt_to_pred[gts[gi][0]] = preds[pi][0]

            # ID switch check
            gt_id   = gts[gi][0]
            pred_id = preds[pi][0]
            if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                id_switches += 1

            iou_matrix[gi, :] = -1
            iou_matrix[:, pi] = -1

        TP += len(matched_gt)
        FP += len(preds)     - len(matched_pred)
        FN += len(gts)       - len(matched_gt)
        prev_gt_to_pred = gt_to_pred

    total_gt = TP + FN
    motp     = total_iou / num_matches if num_matches else 0.0
    mota     = 1 - (FP + FN + id_switches) / max(total_gt, 1)
    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    # IDF1 approximation
    idf1 = 2 * TP / max(2 * TP + FP + FN, 1)

    return {
        "MOTA":        round(mota * 100, 2),
        "MOTP":        round(motp * 100, 2),
        "IDF1":        round(idf1 * 100, 2),
        "Precision":   round(precision * 100, 2),
        "Recall":      round(recall * 100, 2),
        "F1":          round(f1 * 100, 2),
        "TP":          TP,
        "FP":          FP,
        "FN":          FN,
        "ID_Switches": id_switches,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — Detection Metrics
# ══════════════════════════════════════════════════════════════════

def compute_detection_metrics(
    gt_boxes:   List[np.ndarray],       # list of (N,4) arrays per image
    pred_boxes: List[np.ndarray],       # list of (M,4) arrays per image
    pred_scores: List[np.ndarray],      # list of (M,) confidence arrays
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """
    Computes mAP@50 and mAP@50:95 for detection (person class only).
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    aps_per_iou = {}

    for thr in iou_thresholds:
        tp_list, score_list, total_gt = [], [], 0

        for gts, preds, scores in zip(gt_boxes, pred_boxes, pred_scores):
            total_gt += len(gts)
            matched   = np.zeros(len(gts), dtype=bool)

            order   = np.argsort(-scores)
            preds   = preds[order]
            scores  = scores[order]

            for pred in preds:
                best_iou, best_gi = -1, -1
                for gi, gt in enumerate(gts):
                    if matched[gi]:
                        continue
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi

                if best_iou >= thr:
                    tp_list.append(1)
                    matched[best_gi] = True
                else:
                    tp_list.append(0)
                score_list.append(scores[0] if len(scores) else 0.0)

        if not tp_list:
            aps_per_iou[thr] = 0.0
            continue

        tp_arr  = np.array(tp_list)
        cum_tp  = np.cumsum(tp_arr)
        cum_fp  = np.cumsum(1 - tp_arr)
        prec    = cum_tp / (cum_tp + cum_fp + 1e-8)
        rec     = cum_tp / (total_gt + 1e-8)
        aps_per_iou[thr] = float(np.trapz(prec, rec))

    map50    = aps_per_iou.get(0.5, 0.0)
    map5095  = np.mean(list(aps_per_iou.values()))

    return {
        "mAP@50":    round(map50   * 100, 2),
        "mAP@50:95": round(map5095 * 100, 2),
        "per_iou":   {round(k,2): round(v*100,2) for k,v in aps_per_iou.items()},
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — Pretty Printer
# ══════════════════════════════════════════════════════════════════

def print_metrics(results: Dict, title: str = "Evaluation Results"):
    bar = "═" * 50
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

    section_labels = {
        # Re-ID
        "rank-1":  ("Re-ID / CMC",  "%"),
        "rank-5":  ("Re-ID / CMC",  "%"),
        "rank-10": ("Re-ID / CMC",  "%"),
        "rank-20": ("Re-ID / CMC",  "%"),
        "mAP":     ("Re-ID",        "%"),
        "mINP":    ("Re-ID",        "%"),
        # Embedding
        "intra_class_dist": ("Embedding Quality", ""),
        "inter_class_dist": ("Embedding Quality", ""),
        "dist_ratio":       ("Embedding Quality", "x"),
        "silhouette_score": ("Embedding Quality", ""),
        # MOT
        "MOTA":        ("MOT Tracking", "%"),
        "MOTP":        ("MOT Tracking", "%"),
        "IDF1":        ("MOT Tracking", "%"),
        "ID_Switches": ("MOT Tracking", ""),
        # Detection
        "mAP@50":    ("Detection",  "%"),
        "mAP@50:95": ("Detection",  "%"),
    }

    current_section = None
    for key, value in results.items():
        if key in section_labels:
            section, unit = section_labels[key]
            if section != current_section:
                print(f"\n  ── {section} ──")
                current_section = section
            if value is not None:
                print(f"    {key:<22}: {value}{unit}")
        else:
            print(f"    {key:<22}: {value}")

    print(f"{bar}\n")
