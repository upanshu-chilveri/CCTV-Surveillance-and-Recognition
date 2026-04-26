"""
DeepSORT Tracker + YOLOv8 Detection
-------------------------------------
Integrates:
  • YOLOv8 for person bounding-box detection
  • Trained ReIDNet as DeepSORT's appearance extractor
  • Full MOT metric evaluation against a ground-truth annotation file

Output per frame:
  track_id | x1 | y1 | x2 | y2 | confidence

Usage:
  # Live webcam
  python tracking/deepsort_tracker.py --source 0 --reid_ckpt outputs/reid/checkpoints/best_rank1.pth

  # Video file + save output
  python tracking/deepsort_tracker.py \
      --source input.mp4 \
      --reid_ckpt best_rank1.pth \
      --save_video output.mp4 \
      --gt_file gt_annotations.txt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.reid_net import DeepSORTExtractor
from evaluation.metrics import compute_mot_metrics, print_metrics


# ─────────────────────────────────────────
#  Colour palette for track IDs
# ─────────────────────────────────────────
def _id_colour(track_id):
    np.random.seed(int(track_id) * 7 + 3)
    return tuple(int(x) for x in np.random.randint(80, 240, 3))


# ─────────────────────────────────────────
#  GT annotation loader (MOT format)
# ─────────────────────────────────────────
def load_gt_mot(path: str) -> Dict[int, List[Tuple]]:
    """
    Loads MOT17 style annotations:
      frame, id, x, y, w, h, conf, class, visibility
    Returns: {frame_id: [(id, x1,y1,x2,y2), ...]}
    """
    gt = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid, tid = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            gt[fid].append((tid, x, y, x+w, y+h))
    return dict(gt)


# ─────────────────────────────────────────
#  Tracker class
# ─────────────────────────────────────────
class PersonTracker:
    def __init__(
        self,
        yolo_model:  str   = "yolov8m.pt",
        reid_ckpt:   str   = None,
        conf_thresh: float = 0.45,
        max_age:     int   = 30,
        n_init:      int   = 3,
        device:      str   = "cuda",
    ):
        self.device      = device if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh

        # ── YOLO detector ─────────────────────────────────────
        self.yolo = YOLO(yolo_model)
        print(f"  [Detector]  YOLOv8 loaded: {yolo_model}")

        # ── Re-ID feature extractor ───────────────────────────
        self.reid = None
        if reid_ckpt and os.path.exists(reid_ckpt):
            self.reid = DeepSORTExtractor(reid_ckpt, device=self.device)
            print(f"  [Re-ID]     Custom extractor loaded: {reid_ckpt}")
        else:
            print("  [Re-ID]     No checkpoint — using DeepSORT's built-in extractor")

        # ── DeepSORT tracker ──────────────────────────────────
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=0.7,
            embedder=None if self.reid else "mobilenet",
            embedder_gpu=(self.device != "cpu"),
            half=(self.device != "cpu"),
        )
        print(f"  [Tracker]   DeepSORT (max_age={max_age}, n_init={n_init})")

    def detect(self, frame: np.ndarray):
        """Run YOLO detection, return list of [x1,y1,x2,y2, conf]."""
        results = self.yolo.predict(
            frame, classes=[0], conf=self.conf_thresh, verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        return detections

    def track(self, frame: np.ndarray):
        """
        Detect → (optionally) extract Re-ID features → update tracker.
        Returns list of (track_id, x1, y1, x2, y2).
        """
        dets = self.detect(frame)
        if not dets:
            return []

        # Convert to DeepSORT format: ([x1,y1,w,h], conf, class)
        ds_input = [
            ([d[0], d[1], d[2]-d[0], d[3]-d[1]], d[4], "person")
            for d in dets
        ]

        # Optional: pass Re-ID features
        if self.reid:
            crops = [
                frame[max(0,d[1]):d[3], max(0,d[0]):d[2]]
                for d in dets
            ]
            embeds = self.reid(crops)   # (N, 2048)
            tracks = self.tracker.update_tracks(ds_input, frame=frame,
                                                 embeds=embeds)
        else:
            tracks = self.tracker.update_tracks(ds_input, frame=frame)

        output = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            output.append((
                t.track_id,
                int(ltrb[0]), int(ltrb[1]),
                int(ltrb[2]), int(ltrb[3]),
            ))
        return output

    def draw(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        for tid, x1, y1, x2, y2 in tracks:
            colour = _id_colour(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label  = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame


# ─────────────────────────────────────────
#  Run on video / webcam
# ─────────────────────────────────────────
def run(args):
    tracker = PersonTracker(
        yolo_model  = args.yolo_model,
        reid_ckpt   = args.reid_ckpt,
        conf_thresh = args.conf,
        max_age     = args.max_age,
        n_init      = args.n_init,
        device      = args.device,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n  Source: {source}  ({width}×{height} @ {fps:.0f}fps)")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    # For MOT metric collection
    pred_frames: Dict[int, List[Tuple]] = {}
    gt_frames   = load_gt_mot(args.gt_file) if args.gt_file else {}

    frame_id  = 0
    times     = []

    print("  Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0     = time.perf_counter()
        tracks = tracker.track(frame)
        dt     = time.perf_counter() - t0
        times.append(dt)

        # Collect for MOT metrics
        if gt_frames:
            pred_frames[frame_id] = [
                (tid, x1, y1, x2, y2) for tid, x1, y1, x2, y2 in tracks
            ]

        frame = tracker.draw(frame, tracks)

        # FPS overlay
        avg_fps = 1.0 / (sum(times[-30:]) / min(len(times), 30))
        cv2.putText(frame, f"FPS: {avg_fps:.1f}  Tracks: {len(tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        if not args.headless:
            cv2.imshow("Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

        if args.max_frames and frame_id >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Performance summary ────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Processed   : {frame_id} frames")
    print(f"  Avg latency : {np.mean(times)*1000:.1f} ms/frame")
    print(f"  Avg FPS     : {1/np.mean(times):.1f}")
    print(f"  Min FPS     : {1/max(times):.1f}")
    print(f"  Max FPS     : {1/min(times):.1f}")

    if gt_frames and pred_frames:
        mot = compute_mot_metrics(gt_frames, pred_frames, iou_threshold=0.5)
        print_metrics(mot, "MOT Tracking Metrics")


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSORT Person Tracker")
    parser.add_argument("--source",     default="0",
                        help="Video path, image folder, or 0 for webcam")
    parser.add_argument("--reid_ckpt",  default=None,
                        help="Path to trained ReIDNet .pth checkpoint")
    parser.add_argument("--yolo_model", default="yolov8m.pt")
    parser.add_argument("--conf",       type=float, default=0.45)
    parser.add_argument("--max_age",    type=int,   default=30)
    parser.add_argument("--n_init",     type=int,   default=3)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--save_video", default=None)
    parser.add_argument("--gt_file",    default=None,
                        help="Ground-truth MOT annotation file for metrics")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--headless",   action="store_true",
                        help="Don't open display window (for servers)")
    args = parser.parse_args()
    run(args)
