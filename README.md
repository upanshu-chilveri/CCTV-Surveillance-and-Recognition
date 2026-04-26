# Person Re-ID with DeepSORT + DukeMTMC-ReID

A full pipeline for person detection, tracking and re-identification
with complete ML metrics evaluation.

## Project Structure

```
person_reid/
├── dataset/
│   └── duke_dataset.py       # DukeMTMC-ReID loader + transforms
├── models/
│   └── reid_net.py           # ResNet-50 + GeM + BNNeck + Triplet loss
├── evaluation/
│   └── metrics.py            # CMC, mAP, mINP, MOT, detection metrics
├── tracking/
│   └── deepsort_tracker.py   # YOLOv8 + DeepSORT + Re-ID extractor
├── train_reid.py             # Training with warm-up + cosine LR
├── evaluate_reid.py          # Full evaluation + plots
├── improvements.py           # 7 techniques to improve accuracy
└── requirements.txt
```

---

## 1. Setup

```bash
pip install -r requirements.txt
```

---

## 2. Download DukeMTMC-ReID

```bash
# Option A — via dedicated script
python download_data.py

# Option B — manual
# https://exposing.ai/duke_mtmc/  (fill in access form)
# Extract to:  ./data/DukeMTMC-reID/
#   bounding_box_train/
#   bounding_box_test/
#   query/
```

---

## 3. Train

```bash
python train_reid.py \
  --data_root ./data/DukeMTMC-reID \
  --output_dir outputs/reid \
  --epochs 120 \
  --batch_size 64 \
  --lr 3.5e-4
```

Training produces:
- `outputs/reid/checkpoints/best_rank1.pth`
- `outputs/reid/checkpoints/last.pth`
- `outputs/reid/logs/` (TensorBoard)

Monitor: `tensorboard --logdir outputs/reid/logs`

---

## 4. Evaluate Re-ID

```bash
python evaluate_reid.py \
  --data_root  ./data/DukeMTMC-reID \
  --checkpoint outputs/reid/checkpoints/best_rank1.pth \
  --output_dir outputs/eval \
  --tsne
```

Outputs:
- Console: Rank-1/5/10/20, mAP, mINP, embedding quality scores
- `outputs/eval/cmc_curve.png`
- `outputs/eval/distance_distribution.png`
- `outputs/eval/tsne.png`
- `outputs/eval/metrics.csv`

---

## 5. Run Tracker

```bash
# Webcam
python tracking/deepsort_tracker.py \
  --source 0 \
  --reid_ckpt outputs/reid/checkpoints/best_rank1.pth

# Video file + save + evaluate against GT
python tracking/deepsort_tracker.py \
  --source input.mp4 \
  --reid_ckpt outputs/reid/checkpoints/best_rank1.pth \
  --save_video output.mp4 \
  --gt_file gt_annotations.txt
```

---

## 6. Metrics Reference

### Re-ID Metrics
| Metric   | What it measures                          | Target       |
|----------|-------------------------------------------|--------------|
| Rank-1   | % correct top-1 retrievals               | > 85%        |
| Rank-5   | % correct in top-5                        | > 93%        |
| mAP      | Mean avg precision across all queries     | > 75%        |
| mINP     | How well hardest positives are retrieved  | > 50%        |

### Embedding Quality
| Metric           | Good value means...                  |
|------------------|--------------------------------------|
| Intra-class dist | Low → same person clusters tightly   |
| Inter-class dist | High → different people far apart    |
| Dist ratio       | > 2.0 → well-separated embeddings   |
| Silhouette score | Close to 1.0 → excellent separation |

### MOT Tracking Metrics
| Metric      | Formula                                 | Target |
|-------------|------------------------------------------|--------|
| MOTA        | 1 − (FP+FN+IDS)/GT                      | > 70%  |
| MOTP        | Average IoU of matched pairs             | > 75%  |
| IDF1        | 2TP / (2TP + FP + FN)                   | > 70%  |
| ID Switches | Times a track changes identity           | < 50   |

---

## 7. Improvements Cheatsheet

```
Baseline ResNet-50 + CE          ~78% Rank-1
+ Triplet loss                   +2–3%
+ Random Erasing                 +1–2%
+ BNNeck + GeM                   +1–2%
+ Re-ranking (post-processing)   +1–3% Rank-1, +4–8% mAP ← free!
+ OSNet backbone                 +3–6%
+ TransReID (ViT)                +8–12%
```

See `improvements.py` for drop-in code for each technique.
