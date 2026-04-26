"""
Accuracy Improvement Guide
--------------------------
Actionable techniques to push Re-ID performance higher on DukeMTMC.

SOTA benchmarks on DukeMTMC-ReID (2024):
  Rank-1  ~92%   (TransReID-SSL, ViT backbone)
  Rank-1  ~89%   (OSNet-AIN)
  Rank-1  ~86%   (ResNet-50 StrongBaseline — our architecture)
  Rank-1  ~80%   (Vanilla ResNet-50 baseline)

Run any section individually:
  python improvements.py --technique <name>
"""

# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 1 — Random Erasing Augmentation
# ══════════════════════════════════════════════════════════════════
"""
Random Erasing randomly blacks out rectangular patches during training.
This forces the network to use holistic features rather than relying
on a single region (e.g. a logo on a shirt).
Expected gain: +1–2% Rank-1
"""

import torchvision.transforms as T

IMPROVED_TRAIN_TRANSFORMS = T.Compose([
    T.Resize((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 128)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ← This is the key addition:
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
])


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 2 — Label Smoothing + Mixup
# ══════════════════════════════════════════════════════════════════
"""
Label smoothing prevents overconfident predictions.
Mixup blends two training images with a lambda factor.
Expected gain: +0.5–1% mAP
"""

import torch
import torch.nn.functional as F
import numpy as np


def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size  = x.size(0)
    index       = torch.randperm(batch_size, device=x.device)
    mixed_x     = lam * x + (1 - lam) * x[index]
    y_a, y_b    = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 3 — Cosine Annealing with Warm Restarts
# ══════════════════════════════════════════════════════════════════
"""
SGDR (Stochastic Gradient Descent with Warm Restarts) helps escape
local minima. Combine with a linear warm-up for the first 10 epochs.
Expected gain: +0.5–1% Rank-1  vs step LR
"""

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim


def get_optimizer_and_scheduler(model, base_lr=3.5e-4, weight_decay=5e-4,
                                 epochs=120, warmup_epochs=10):
    optimizer = optim.AdamW([
        {"params": model.base.parameters(),       "lr": base_lr * 0.1},  # lower LR for backbone
        {"params": model.pool.parameters(),       "lr": base_lr},
        {"params": model.neck.parameters(),       "lr": base_lr},
        {"params": model.classifier.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)

    # Cosine with restarts every 30 epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6)
    return optimizer, scheduler


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 4 — Part-Based Feature Pooling (PCB)
# ══════════════════════════════════════════════════════════════════
"""
PCB divides the person image into horizontal stripes (e.g. 6 parts)
and pools each part independently. This captures local features like
shoes, trousers, jacket, face.
Expected gain: +3–5% Rank-1 over global pooling
"""

import torch.nn as nn


class PCBHead(nn.Module):
    """
    Part-Based Convolutional Baseline.
    Replace the global pool in ReIDNet with this.
    """
    def __init__(self, in_channels: int = 2048, num_parts: int = 6,
                 num_classes: int = 702, feat_dim: int = 256):
        super().__init__()
        self.num_parts = num_parts
        self.pool      = nn.AdaptiveAvgPool2d((num_parts, 1))

        # One BN + classifier per part
        self.bns          = nn.ModuleList([nn.BatchNorm1d(in_channels) for _ in range(num_parts)])
        self.classifiers  = nn.ModuleList(
            [nn.Linear(in_channels, num_classes, bias=False) for _ in range(num_parts)]
        )
        self.feat_dim = in_channels * num_parts

    def forward(self, feat_map):
        # feat_map: (B, C, H, W)
        parts = self.pool(feat_map)   # (B, C, num_parts, 1)
        parts = parts.squeeze(-1)     # (B, C, num_parts)

        logits_list = []
        feat_list   = []
        for i in range(self.num_parts):
            f      = parts[:, :, i]              # (B, C)
            f_bn   = self.bns[i](f)
            logits = self.classifiers[i](f_bn)
            logits_list.append(logits)
            feat_list.append(f_bn)

        # Concatenate all part features for retrieval
        feat_concat = torch.cat(feat_list, dim=1)          # (B, C*num_parts)
        return logits_list, F.normalize(feat_concat, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 5 — OSNet (Omni-Scale Network)
# ══════════════════════════════════════════════════════════════════
"""
OSNet was specifically designed for Re-ID. It uses omni-scale feature
learning with unified aggregation gates. Consistently outperforms
ResNet-50 at the same parameter count.

Install: pip install torchreid
Expected gain: +3–6% Rank-1 vs ResNet-50

Usage example below.
"""

def build_osnet_model(num_classes: int = 702):
    try:
        import torchreid
        model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=num_classes,
            pretrained=True,          # ImageNet pretrained
        )
        return model
    except ImportError:
        print("Install torchreid: pip install torchreid")
        return None


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 6 — Test-Time Augmentation (TTA)
# ══════════════════════════════════════════════════════════════════
"""
At inference time, run each image through both original and
horizontally-flipped versions, then average the embeddings.
Expected gain: +0.5–1.5% Rank-1 for free
"""

import torchvision.transforms.functional as TF


@torch.no_grad()
def extract_with_tta(model, imgs: torch.Tensor, device: str) -> torch.Tensor:
    """Average original + horizontally flipped embeddings."""
    model.eval()
    imgs       = imgs.to(device)
    imgs_flip  = TF.hflip(imgs)

    feat       = model(imgs)
    feat_flip  = model(imgs_flip)

    return F.normalize((feat + feat_flip) / 2, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════
#  TECHNIQUE 7 — Re-Ranking (k-reciprocal encoding)
# ══════════════════════════════════════════════════════════════════
"""
Post-processing step that refines the distance matrix using
k-reciprocal neighbours. The most effective single improvement
for Re-ID evaluation (no retraining required).
Expected gain: +4–8% mAP, +1–3% Rank-1
"""


def re_ranking(q_feats: np.ndarray, g_feats: np.ndarray,
               k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
    """
    Re-ranking by k-reciprocal encoding.
    Reference: Zhong et al. (CVPR 2017) — "Re-ranking Person Re-identification"

    Returns: refined (Q, G) distance matrix
    """
    from scipy.spatial.distance import cdist

    q_num = q_feats.shape[0]
    g_num = g_feats.shape[0]
    all_feats = np.concatenate([q_feats, g_feats])          # (Q+G, D)

    # Initial cosine distance matrix
    dist = cdist(all_feats, all_feats, "cosine").astype(np.float32)

    # k-reciprocal neighbours
    def k_reciprocal_neigh(dist_mat, i, k1):
        forward  = np.argsort(dist_mat[i])[1:k1+1]
        backward = np.argsort(dist_mat[forward, i]).flatten()
        # keep indices that are "reciprocal"
        return forward[backward < k1]

    V = np.zeros_like(dist)
    for i in range(q_num + g_num):
        k_recip = k_reciprocal_neigh(dist, i, k1)
        # ½ k1 expansion
        k_recip_exp = list(k_recip)
        for j in k_recip:
            kj = k_reciprocal_neigh(dist, j, max(1, k1//2))
            if len(np.intersect1d(kj, k_recip)) > 2/3 * len(kj):
                k_recip_exp += list(kj)
        k_recip_exp = np.unique(k_recip_exp)

        # Jaccard distance kernel
        w = np.exp(-dist[i, k_recip_exp])
        V[i, k_recip_exp] = w / w.sum()

    # Query expansion (k2)
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(q_num + g_num):
            knn = np.argsort(dist[i])[:k2]
            V_qe[i] = V[knn].mean(axis=0)
        V = V_qe

    # Jaccard distance
    jac_dist = 1 - (V @ V.T)
    jac_dist = np.clip(jac_dist, 0, None)

    # Final distance
    final_dist = (1 - lambda_value) * jac_dist + lambda_value * dist
    return final_dist[:q_num, q_num:]


# ══════════════════════════════════════════════════════════════════
#  SUMMARY — Expected performance gains
# ══════════════════════════════════════════════════════════════════
IMPROVEMENT_SUMMARY = """
╔══════════════════════════════════════════════════════════════════╗
║            Improvement Roadmap — DukeMTMC-ReID                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Technique                   │ Rank-1 gain │  mAP gain  │ Cost ║
║─────────────────────────────────────────────────────────────── ║
║  Baseline (ResNet-50 + CE)   │    ~78%     │   ~60%     │  —   ║
║  + Triplet loss              │   +2–3%     │  +3–5%     │  Low ║
║  + Label smoothing           │   +0.5%     │  +1%       │  Low ║
║  + Random Erasing            │   +1–2%     │  +1–2%     │  Low ║
║  + BNNeck                    │   +1%       │  +1%       │  Low ║
║  + GeM pooling               │   +0.5%     │  +0.5%     │  Low ║
║  + Part-Based (PCB)          │   +3–5%     │  +3–4%     │  Med ║
║  → Swap to OSNet             │   +3–6%     │  +4–6%     │  Med ║
║  + Mixup augmentation        │   +0.5%     │  +1%       │  Low ║
║  + Test-Time Aug (TTA)       │   +0.5–1%   │  +0.5–1%   │  Low ║
║  + Re-ranking (post-proc)    │   +1–3%     │  +4–8%     │  Low ║
║  → Swap to TransReID (ViT)   │   +8–12%    │  +10–15%   │  High║
╠══════════════════════════════════════════════════════════════════╣
║  Combined strong baseline    │   ~86–88%   │  ~76–78%   │  Med ║
║  With OSNet + re-ranking     │   ~89–91%   │  ~82–84%   │  Med ║
║  With TransReID (SOTA)       │   ~92–93%   │  ~87–89%   │  High║
╚══════════════════════════════════════════════════════════════════╝

DeepSORT-specific improvements:
  ─────────────────────────────────────────────────────────────
  • Increase max_age: keeps IDs longer across occlusions
  • Decrease n_init:  confirms tracks faster
  • Tune max_iou_distance: looser = more robust to motion
  • Add Kalman filter velocity: already in DeepSORT
  • Use ByteTrack instead: keeps low-conf detections in buffer
"""

if __name__ == "__main__":
    print(IMPROVEMENT_SUMMARY)
