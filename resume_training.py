import os
import sys
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# local imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset.duke_dataset import DukeMTMCReID
from models.reid_net import ReIDNet, ReIDLoss
from evaluate_reid import evaluate


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✔ Checkpoint saved → {path}")


def warmup_cosine_lr(epoch: int, warmup_epochs: int, max_epochs: int,
                      base_lr: float, min_lr: float = 1e-6) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────
#  One Epoch
# ─────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = total_ce = total_tri = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, ncols=90)
    for imgs, pids, _ in pbar:
        imgs = imgs.to(device)
        pids = pids.to(device)

        logits, feats = model(imgs)
        loss, ce, tri = criterion(logits, feats, pids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce   += ce.item()
        total_tri  += tri.item()

        pred     = logits.argmax(dim=1)
        correct += (pred == pids).sum().item()
        total   += imgs.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.3f}",
                          "ce":   f"{ce.item():.3f}",
                          "tri":  f"{tri.item():.3f}"})

    n = len(loader)
    metrics = {
        "loss/total":   total_loss / n,
        "loss/ce":      total_ce   / n,
        "loss/triplet": total_tri  / n,
        "train/acc":    correct    / total * 100,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)

    return metrics


# ─────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")

    # ── Data ────────────────────────────────────────────────────
    duke = DukeMTMCReID(args.data_root, img_size=(256, 128),
                        batch_size=args.batch_size)
    duke.summary()

    train_loader   = duke.train_loader(num_workers=args.workers)
    query_loader   = duke.query_loader(num_workers=args.workers)
    gallery_loader = duke.gallery_loader(num_workers=args.workers)

    # ── Model ────────────────────────────────────────────────────
    model = ReIDNet(
        num_classes=duke.num_train_ids,
        feat_dim=2048,
        use_gem=True,
        pretrained=False,   # weights are loaded from checkpoint below
    ).to(device)
    print(f"  Model   : ReIDNet (ResNet-50 + GeM)")
    print(f"  Classes : {duke.num_train_ids} training identities\n")

    # ── Loss ─────────────────────────────────────────────────────
    criterion = ReIDLoss(
        margin=0.3,
        ce_weight=1.0,
        triplet_weight=1.0,
        label_smoothing=0.1,
    )

    # ── Optimiser ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Logging ──────────────────────────────────────────────────
    log_dir = os.path.join(args.output_dir, "logs")
    writer  = SummaryWriter(log_dir)

    best_rank1 = 0.0
    best_mAP   = 0.0

    # ── Resume from checkpoint ───────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, "checkpoints", "last.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'.\n"
            f"Run train_reid.py first to create a checkpoint."
        )

    print(f"\n🔄 Resuming from checkpoint: {ckpt_path}")
    checkpoint  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']          # epoch that was *completed*
    best_rank1  = checkpoint.get('rank-1', 0.0)
    best_mAP    = checkpoint.get('mAP',    0.0)
    print(f"   Resumed from epoch {start_epoch}  "
          f"| best Rank-1 so far: {best_rank1:.2f}%  "
          f"| best mAP so far: {best_mAP:.2f}%")

    # ── Guard: nothing left to train ────────────────────────────
    if args.epochs <= start_epoch:
        print(f"\n⚠️  --epochs ({args.epochs}) ≤ completed epoch ({start_epoch}). "
              f"Nothing to do. Increase --epochs and re-run.")
        writer.close()
        return

    print("─" * 55)
    print(f"  Remaining epochs : {start_epoch + 1} → {args.epochs}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  LR               : {args.lr} (warmup {args.warmup_epochs} epochs → cosine)")
    print("─" * 55)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # ── Update LR ─────────────────────────────────────────
        lr = warmup_cosine_lr(epoch-1, args.warmup_epochs, args.epochs,
                               args.lr, min_lr=1e-6)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        writer.add_scalar("train/lr", lr, epoch)

        # ── Train one epoch ───────────────────────────────────
        metrics = run_epoch(model, train_loader, criterion,
                             optimizer, device, epoch, writer)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss={metrics['loss/total']:.3f}  "
              f"ce={metrics['loss/ce']:.3f}  "
              f"tri={metrics['loss/triplet']:.3f}  "
              f"acc={metrics['train/acc']:.1f}%  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]")

        # ── Evaluate every eval_freq epochs ───────────────────
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            reid_metrics = evaluate(model, query_loader, gallery_loader,
                                     device, remove_same_camera=True)

            rank1 = reid_metrics["rank-1"]
            mAP   = reid_metrics["mAP"]

            for k, v in reid_metrics.items():
                if v is not None:
                    writer.add_scalar(f"val/{k}", v, epoch)

            print(f"  → Rank-1: {rank1:.2f}%  |  mAP: {mAP:.2f}%  |  "
                  f"mINP: {reid_metrics['mINP']:.2f}%")

            # ── Checkpointing ─────────────────────────────────
            state = {
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rank-1": rank1,
                "mAP":    mAP,
            }

            save_checkpoint(state,
                os.path.join(args.output_dir, "checkpoints", "last.pth"))

            if rank1 > best_rank1:
                best_rank1 = rank1
                best_mAP   = mAP
                save_checkpoint(state,
                    os.path.join(args.output_dir, "checkpoints", "best_rank1.pth"))
                print(f"  ★ New best Rank-1: {best_rank1:.2f}%")

    writer.close()
    print(f"\n✅ Training complete!")
    print(f"   Best Rank-1 : {best_rank1:.2f}%")
    print(f"   Best mAP    : {best_mAP:.2f}%")
    print(f"   Saved to    : {args.output_dir}\n")


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Re-ID on DukeMTMC")
    parser.add_argument("--data_root",    required=True,
                        help="Path to DukeMTMC-reID root folder")
    parser.add_argument("--output_dir",   default="outputs/reid",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--epochs",       type=int,   default=120)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs",type=int,   default=10)
    parser.add_argument("--eval_freq",    type=int,   default=10,
                        help="Evaluate every N epochs")
    parser.add_argument("--workers",      type=int,   default=4)

    args = parser.parse_args()
    train(args)
