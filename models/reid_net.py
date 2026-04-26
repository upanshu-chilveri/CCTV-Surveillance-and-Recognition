"""
Re-ID Backbone
--------------
ResNet-50 with:
  • Global Average Pooling  → 2048-d feature vector
  • BatchNorm neck          → normalised embedding
  • ID classification head  → for training with CE + Triplet loss
  • Optional GeM pooling    → better than GAP for Re-ID

Architecture is compatible with DeepSORT's appearance extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ─────────────────────────────────────────
#  Generalised Mean Pooling (GeM)
# ─────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=(1, 1)
        ).pow(1.0 / self.p)


# ─────────────────────────────────────────
#  BatchNorm neck  (standard in Re-ID)
# ─────────────────────────────────────────
class BNNeck(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        return self.bn(x)


# ─────────────────────────────────────────
#  Main Re-ID Network
# ─────────────────────────────────────────
class ReIDNet(nn.Module):
    """
    Parameters
    ----------
    num_classes : int   – number of training identities
    feat_dim    : int   – embedding dimension (default 2048)
    use_gem     : bool  – use GeM pooling instead of GAP
    pretrained  : bool  – load ImageNet weights
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 2048,
        use_gem: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.base = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        # ── Pooling ───────────────────────────────
        self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d((1, 1))

        # ── Neck ──────────────────────────────────
        self.neck = BNNeck(feat_dim)

        # ── Classifier (only used during training) ─
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.feat_dim    = feat_dim
        self.num_classes = num_classes

    def forward(self, x):
        feat_map = self.base(x)                          # (B, 2048, H, W)
        feat     = self.pool(feat_map).flatten(1)        # (B, 2048)
        feat_bn  = self.neck(feat)                       # (B, 2048) – L2-normed neck

        if self.training:
            logits = self.classifier(feat_bn)
            return logits, feat                          # for CE + Triplet

        return F.normalize(feat_bn, p=2, dim=1)         # L2-normed for retrieval

    @torch.no_grad()
    def extract_features(self, x):
        """Inference-only, returns L2-normalised embeddings."""
        self.eval()
        return self(x)


# ─────────────────────────────────────────
#  Triplet Loss (hard mining)
# ─────────────────────────────────────────
class TripletLoss(nn.Module):
    """Batch-hard triplet loss with soft margin."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings : (B, D)
        labels     : (B,)
        """
        B = embeddings.size(0)

        # Pairwise L2 distances
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        mask_neg = ~mask_pos

        # Hard positive: largest distance among same-class pairs
        dist_ap = (dist * mask_pos.float()).max(dim=1)[0]        # (B,)

        # Hard negative: smallest distance among diff-class pairs
        dist_an = dist.clone()
        dist_an[mask_pos] = float("inf")
        dist_an = dist_an.min(dim=1)[0]                         # (B,)

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss


# ─────────────────────────────────────────
#  Combined Loss
# ─────────────────────────────────────────
class ReIDLoss(nn.Module):
    def __init__(self, margin: float = 0.3, ce_weight: float = 1.0,
                 triplet_weight: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        self.ce      = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = TripletLoss(margin=margin)
        self.ce_w    = ce_weight
        self.tri_w   = triplet_weight

    def forward(self, logits, features, labels):
        ce_loss  = self.ce(logits, labels)
        tri_loss = self.triplet(features, labels)
        total    = self.ce_w * ce_loss + self.tri_w * tri_loss
        return total, ce_loss, tri_loss


# ─────────────────────────────────────────
#  DeepSORT-compatible feature extractor
# ─────────────────────────────────────────
class DeepSORTExtractor:
    """
    Wraps ReIDNet for use as the appearance extractor inside DeepSORT.
    Accepts numpy/PIL crops, returns (N, 128) float32 feature arrays.
    """

    def __init__(self, checkpoint: str, device: str = "cuda",
                 feat_dim: int = 2048, num_classes: int = 702):
        import torchvision.transforms as T
        import numpy as np

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.np     = np

        self.model = ReIDNet(num_classes=num_classes, feat_dim=feat_dim)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, crops: list) -> "np.ndarray":
        if not crops:
            return self.np.empty((0, 2048), dtype=self.np.float32)

        tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
        feats   = self.model.extract_features(tensors)
        return feats.cpu().numpy()
