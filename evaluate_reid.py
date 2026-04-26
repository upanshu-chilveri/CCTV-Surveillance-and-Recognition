"""
Evaluation Script
-----------------
Evaluates a trained ReIDNet checkpoint on DukeMTMC-ReID test split.

Produces:
  • Rank-1/5/10/20, mAP, mINP  (Re-ID metrics)
  • Embedding quality scores
  • CMC curve plot
  • t-SNE / UMAP embedding visualization
  • Per-camera breakdown

Run:
  python evaluate_reid.py \
    --data_root /path/to/DukeMTMC-reID \
    --checkpoint outputs/reid/checkpoints/best_rank1.pth
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset.duke_dataset import DukeMTMCReID
from models.reid_net import ReIDNet
from evaluation.metrics import (
    compute_distance_matrix,
    compute_cmc_map,
    embedding_quality,
    print_metrics,
)


# ─────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device) -> tuple:
    """Returns (features, pids, cids) as numpy arrays."""
    model.eval()
    all_feats, all_pids, all_cids = [], [], []

    for imgs, pids, cids in tqdm(loader, desc="  Extracting features", ncols=80):
        feats = model(imgs.to(device))
        all_feats.append(feats.cpu().numpy())
        all_pids.append(pids.numpy())
        all_cids.append(cids.numpy())

    return (np.concatenate(all_feats),
            np.concatenate(all_pids),
            np.concatenate(all_cids))


# ─────────────────────────────────────────
#  Core evaluation (also called from train)
# ─────────────────────────────────────────
def evaluate(model, query_loader, gallery_loader, device,
             remove_same_camera: bool = True, metric: str = "cosine"):

    q_feats, q_pids, q_cids = extract_features(model, query_loader,   device)
    g_feats, g_pids, g_cids = extract_features(model, gallery_loader, device)

    dist_matrix = compute_distance_matrix(q_feats, g_feats, metric=metric)

    results = compute_cmc_map(
        dist_matrix,
        q_pids, g_pids,
        q_cids, g_cids,
        max_rank=20,
        remove_same_camera=remove_same_camera,
    )
    return results


# ─────────────────────────────────────────
#  CMC Curve plot
# ─────────────────────────────────────────
def plot_cmc_curve(cmc_scores: np.ndarray, save_path: str):
    ranks = list(range(1, len(cmc_scores) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, cmc_scores * 100, marker="o", markersize=4,
            color="#2196F3", linewidth=2)
    ax.fill_between(ranks, cmc_scores * 100, alpha=0.15, color="#2196F3")

    ax.set_xlabel("Rank", fontsize=13)
    ax.set_ylabel("Matching Rate (%)", fontsize=13)
    ax.set_title("CMC Curve — DukeMTMC-ReID", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)

    for rank in [1, 5, 10, 20]:
        if rank <= len(cmc_scores):
            val = cmc_scores[rank-1] * 100
            ax.annotate(f"{val:.1f}%",
                        xy=(rank, val), xytext=(rank+0.3, val-5),
                        fontsize=9, color="#1565C0")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 CMC curve saved → {save_path}")


# ─────────────────────────────────────────
#  t-SNE visualization
# ─────────────────────────────────────────
def plot_tsne(features: np.ndarray, labels: np.ndarray,
              save_path: str, n_ids: int = 20, max_samples: int = 500):
    from sklearn.manifold import TSNE

    # Sample top-N identities
    unique_ids = np.unique(labels)[:n_ids]
    mask = np.isin(labels, unique_ids)
    feats_sub = features[mask]
    labels_sub = labels[mask]

    if len(feats_sub) > max_samples:
        idx       = np.random.choice(len(feats_sub), max_samples, replace=False)
        feats_sub = feats_sub[idx]
        labels_sub = labels_sub[idx]

    print(f"  Running t-SNE on {len(feats_sub)} samples ...")
    proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats_sub)

    palette = sns.color_palette("tab20", n_ids)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, uid in enumerate(unique_ids):
        idx = labels_sub == uid
        ax.scatter(proj[idx, 0], proj[idx, 1],
                   color=palette[i], label=f"ID {uid}", s=18, alpha=0.8)

    ax.set_title("t-SNE of Re-ID Embeddings (query set)", fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 t-SNE plot saved → {save_path}")


# ─────────────────────────────────────────
#  Distance distribution plot
# ─────────────────────────────────────────
def plot_distance_distribution(
    dist_matrix: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    save_path: str,
    sample: int = 50000,
):
    pos_dists, neg_dists = [], []

    for qi in range(len(q_pids)):
        for gi in range(len(g_pids)):
            if q_pids[qi] == g_pids[gi]:
                pos_dists.append(dist_matrix[qi, gi])
            else:
                neg_dists.append(dist_matrix[qi, gi])

    # Random subsample to keep it manageable
    rng = np.random.default_rng(0)
    if len(pos_dists) > sample:
        pos_dists = rng.choice(pos_dists, sample, replace=False).tolist()
    if len(neg_dists) > sample:
        neg_dists = rng.choice(neg_dists, sample, replace=False).tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pos_dists, bins=80, alpha=0.65, color="#4CAF50", label="Same ID (positive)")
    ax.hist(neg_dists, bins=80, alpha=0.65, color="#F44336", label="Diff ID (negative)")
    ax.set_xlabel("Cosine Distance", fontsize=12)
    ax.set_ylabel("Count",           fontsize=12)
    ax.set_title("Intra- vs Inter-class Distance Distribution", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Distance distribution saved → {save_path}")


# ─────────────────────────────────────────
#  Full evaluation pipeline
# ─────────────────────────────────────────
def full_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📐 Evaluating on: {device}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    duke = DukeMTMCReID(args.data_root, batch_size=args.batch_size)
    duke.summary()
    query_loader   = duke.query_loader()
    gallery_loader = duke.gallery_loader()

    # ── Load model ───────────────────────────────────────────────
    model = ReIDNet(num_classes=702, feat_dim=2048, use_gem=True, pretrained=False)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded checkpoint: {args.checkpoint}")
    if "epoch" in ckpt:
        print(f"  Checkpoint epoch : {ckpt['epoch']}")

    # ── Extract features ─────────────────────────────────────────
    q_feats, q_pids, q_cids = extract_features(model, query_loader,   device)
    g_feats, g_pids, g_cids = extract_features(model, gallery_loader, device)

    print(f"\n  Query   features : {q_feats.shape}")
    print(f"  Gallery features : {g_feats.shape}")

    # ── Distance matrix ──────────────────────────────────────────
    print("\n  Computing distance matrix ...")
    dist_matrix = compute_distance_matrix(q_feats, g_feats, metric="cosine")

    # ── Re-ID metrics ────────────────────────────────────────────
    reid_metrics = compute_cmc_map(
        dist_matrix, q_pids, g_pids, q_cids, g_cids,
        max_rank=20, remove_same_camera=True,
    )
    print_metrics(reid_metrics, "Re-ID Metrics (DukeMTMC-ReID)")

    # ── Embedding quality ────────────────────────────────────────
    print("  Computing embedding quality (this may take a moment) ...")
    all_feats = np.concatenate([q_feats, g_feats])
    all_pids  = np.concatenate([q_pids,  g_pids])
    emb_qual  = embedding_quality(all_feats, all_pids)
    print_metrics(emb_qual, "Embedding Quality Metrics")

    # ── Combined summary ─────────────────────────────────────────
    combined = {**reid_metrics, **emb_qual}
    print_metrics(combined, "Full Evaluation Summary")

    # ── Interpretation hints ─────────────────────────────────────
    rank1 = reid_metrics["rank-1"]
    mAP   = reid_metrics["mAP"]
    print("📋 Performance Assessment:")
    if rank1 >= 85:
        print(f"   Rank-1 {rank1:.1f}% — Excellent (state-of-the-art range: 85–90%+)")
    elif rank1 >= 75:
        print(f"   Rank-1 {rank1:.1f}% — Good (solid baseline)")
    elif rank1 >= 60:
        print(f"   Rank-1 {rank1:.1f}% — Moderate (see improvement guide)")
    else:
        print(f"   Rank-1 {rank1:.1f}% — Needs improvement (check training setup)")

    if mAP >= 70:
        print(f"   mAP    {mAP:.1f}% — Excellent")
    elif mAP >= 55:
        print(f"   mAP    {mAP:.1f}% — Good")
    else:
        print(f"   mAP    {mAP:.1f}% — Moderate")

    # ── Plots ────────────────────────────────────────────────────
    print("\n  Generating plots ...")

    # CMC curve
    num_q   = dist_matrix.shape[0]
    all_cmc = np.zeros(20)
    for qi in range(num_q):
        order  = np.argsort(dist_matrix[qi])
        g_pids_sorted = g_pids[order]
        g_cids_sorted = g_cids[order]
        keep   = ~((g_pids_sorted == q_pids[qi]) & (g_cids_sorted == q_cids[qi]))
        cmc    = (g_pids_sorted[keep] == q_pids[qi])[:20].cumsum()
        cmc[cmc > 1] = 1
        all_cmc += cmc
    cmc_curve = all_cmc / num_q

    plot_cmc_curve(cmc_curve,
                   os.path.join(args.output_dir, "cmc_curve.png"))

    plot_distance_distribution(
        dist_matrix, q_pids, g_pids,
        os.path.join(args.output_dir, "distance_distribution.png"),
    )

    if args.tsne:
        plot_tsne(q_feats, q_pids,
                  os.path.join(args.output_dir, "tsne.png"))

    # ── Save metrics to CSV ──────────────────────────────────────
    import pandas as pd
    flat = {k: v for k, v in combined.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(
        os.path.join(args.output_dir, "metrics.csv"), index=False
    )
    print(f"\n  ✅ All results saved to: {args.output_dir}")


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Re-ID on DukeMTMC")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--output_dir",  default="outputs/eval")
    parser.add_argument("--batch_size",  type=int,  default=64)
    parser.add_argument("--tsne",        action="store_true",
                        help="Generate t-SNE embedding plot (slow)")
    args = parser.parse_args()
    full_eval(args)
