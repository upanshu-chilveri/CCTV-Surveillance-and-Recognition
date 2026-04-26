"""
DukeMTMC-ReID Dataset Loader
------------------------------
Dataset structure expected:
  DukeMTMC-reID/
    bounding_box_train/   <- training images  (702 identities)
    bounding_box_test/    <- gallery images   (702 identities)
    query/                <- query images     (702 identities)

Filename format: <person_id>_c<camera_id>_<frame>_<detection>.jpg
  e.g. 0002_c1_f0044158_00.jpg
"""

import os
import re
from glob import glob
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─────────────────────────────────────────
#  Image transforms
# ─────────────────────────────────────────
def build_transforms(is_train: bool = True, img_size: tuple = (256, 128)):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if is_train:
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(img_size),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ─────────────────────────────────────────
#  Parse a single filename
# ─────────────────────────────────────────
def parse_filename(fname: str):
    """Return (person_id, camera_id) from DukeMTMC filename."""
    basename = os.path.splitext(os.path.basename(fname))[0]
    match = re.match(r"(\d+)_c(\d+)", basename)
    if match:
        pid = int(match.group(1))
        cid = int(match.group(2))
        return pid, cid
    return -1, -1


# ─────────────────────────────────────────
#  Load split (train / query / gallery)
# ─────────────────────────────────────────
def load_split(directory: str, relabel: bool = False):
    """
    Returns list of (img_path, person_id, camera_id).
    If relabel=True, remaps person_ids to 0-based contiguous integers.
    """
    img_paths = sorted(glob(os.path.join(directory, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg images found in: {directory}")

    data = []
    pid_set = set()
    for p in img_paths:
        pid, cid = parse_filename(p)
        if pid == -1:
            continue
        data.append((p, pid, cid))
        pid_set.add(pid)

    if relabel:
        pid_map = {pid: idx for idx, pid in enumerate(sorted(pid_set))}
        data = [(p, pid_map[pid], cid) for p, pid, cid in data]

    return data


# ─────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────
class DukeReIDDataset(Dataset):
    def __init__(self, data: list, transform=None):
        self.data      = data          # [(path, pid, cid), ...]
        self.transform = transform
        self.num_classes = len({pid for _, pid, _ in data})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, pid, cid = self.data[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pid, cid


# ─────────────────────────────────────────
#  High-level builder
# ─────────────────────────────────────────
class DukeMTMCReID:
    """
    Convenience wrapper that loads all three splits and exposes
    ready-to-use DataLoaders.

    Usage
    -----
    duke = DukeMTMCReID("/path/to/DukeMTMC-reID")
    train_loader  = duke.train_loader()
    query_loader  = duke.query_loader()
    gallery_loader = duke.gallery_loader()
    """

    def __init__(self, root: str, img_size: tuple = (256, 128), batch_size: int = 32):
        self.root       = root
        self.img_size   = img_size
        self.batch_size = batch_size

        train_dir   = os.path.join(root, "bounding_box_train")
        query_dir   = os.path.join(root, "query")
        gallery_dir = os.path.join(root, "bounding_box_test")

        self._train_data   = load_split(train_dir,   relabel=True)
        self._query_data   = load_split(query_dir,   relabel=False)
        self._gallery_data = load_split(gallery_dir, relabel=False)

        print(f"[DukeMTMC-ReID] Train   : {len(self._train_data):>6} images | "
              f"{len({p for _,p,_ in self._train_data})} IDs")
        print(f"[DukeMTMC-ReID] Query   : {len(self._query_data):>6} images | "
              f"{len({p for _,p,_ in self._query_data})} IDs")
        print(f"[DukeMTMC-ReID] Gallery : {len(self._gallery_data):>6} images | "
              f"{len({p for _,p,_ in self._gallery_data})} IDs")

    @property
    def num_train_ids(self):
        return len({pid for _, pid, _ in self._train_data})

    def train_loader(self, num_workers: int = 4):
        ds = DukeReIDDataset(self._train_data,
                             transform=build_transforms(True, self.img_size))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)

    def query_loader(self, num_workers: int = 4):
        ds = DukeReIDDataset(self._query_data,
                             transform=build_transforms(False, self.img_size))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    def gallery_loader(self, num_workers: int = 4):
        ds = DukeReIDDataset(self._gallery_data,
                             transform=build_transforms(False, self.img_size))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    def summary(self):
        """Print dataset statistics."""
        cam_counts = defaultdict(int)
        for _, _, cid in self._train_data:
            cam_counts[cid] += 1
        print("\n── Dataset Summary ──────────────────────────")
        print(f"  Root            : {self.root}")
        print(f"  Train images    : {len(self._train_data)}")
        print(f"  Train IDs       : {self.num_train_ids}")
        print(f"  Query images    : {len(self._query_data)}")
        print(f"  Gallery images  : {len(self._gallery_data)}")
        print(f"  Cameras (train) : {dict(cam_counts)}")
        print("─────────────────────────────────────────────\n")
