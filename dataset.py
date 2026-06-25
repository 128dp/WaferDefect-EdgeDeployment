"""
WM-811K Wafer Map Dataset loader.

Dataset source:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Setup:
    1. Download LSWMD.pkl from the link above
    2. Place it at:  data/LSWMD.pkl  (create the data/ folder if needed)

Wafer map pixel values
----------------------
    0  →  no die (outside wafer boundary)
    1  →  normal die
    2  →  defective die

9 failure classes (one-hot encoded, alphabetical order in the raw pkl)
----------------------------------------------------------------------
    0: Center    1: Donut     2: Edge-Loc   3: Edge-Ring
    4: Loc       5: Near-full 6: none       7: Random    8: Scratch

This module excludes "none" (index 6) by default, yielding a realistic
8-class imbalanced problem with ~25,519 labeled samples.
"""

import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import Config


# ── Dataset class ──────────────────────────────────────────────────────────────

class WaferMapDataset(Dataset):
    """PyTorch Dataset for WM-811K wafer map defect classification.

    Pixel normalisation: 0 → 0.0 (no die), 1 → 0.5 (good), 2 → 1.0 (defective)
    Output tensor: (3, image_size, image_size) with ImageNet normalisation
    so pretrained CNNs can be used without modification.
    """

    # ImageNet stats — applied after grayscale-to-RGB replication
    _NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # Spatial augmentations (training only; PIL-level, rotation-safe for wafer maps)
    _AUGMENT = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])

    def __init__(
        self,
        wafer_maps: List[np.ndarray],
        labels: List[int],
        image_size: int = 64,
        augment: bool = False,
    ):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wmap = self.wafer_maps[idx].astype(np.float32)

        # Normalise pixel values: {0, 1, 2} → {0.0, 0.5, 1.0}
        wmap = wmap / 2.0

        # Resize to fixed spatial resolution via PIL
        pil_img = Image.fromarray((wmap * 255).astype(np.uint8), mode="L")
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Spatial augmentation (training only)
        if self.augment:
            pil_img = self._AUGMENT(pil_img)

        # PIL → float tensor, replicate to 3 channels for pretrained CNNs
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        tensor = self._NORMALIZE(tensor)

        return tensor, self.labels[idx]


# ── Loader factory ─────────────────────────────────────────────────────────────

def load_wm811k(cfg: Config):
    """Load WM-811K, split, and return DataLoaders + class weights.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    class_weights : torch.FloatTensor  — for weighted CrossEntropyLoss
    """
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{data_path}'.\n\n"
            "  Steps to download:\n"
            "    1. Visit https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map\n"
            "    2. Download LSWMD.pkl\n"
            "    3. Create a 'data/' folder in this project root and place LSWMD.pkl inside it\n"
        )

    print(f"[Dataset] Loading {data_path} ...")
    # LSWMD.pkl compatibility fixes:
    #   1. Pickled with old pandas that used pandas.indexes (removed in pandas 1.0)
    #   2. Pickled under Python 2 — requires encoding='latin1'
    #   3. Suppress NumPy 2.x dtype deprecation warnings from the old file format
    if "pandas.indexes" not in sys.modules:
        import pandas.core.indexes
        sys.modules["pandas.indexes"] = pandas.core.indexes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(data_path, "rb") as f:
            df = pickle.load(f, encoding="latin1")

    # ── Filter labeled samples ─────────────────────────────────────────────────
    # Labeled: failureType = array([['Center']], dtype='<U6')  → shape (1,1), size 1
    # Unlabeled: failureType = array([], shape=(0,0), dtype=float64) → size 0
    def _is_labeled(val) -> bool:
        try:
            return val.size > 0 and str(val[0][0]).strip() != ""
        except (IndexError, AttributeError, TypeError):
            return False

    df_labeled = df[df["failureType"].apply(_is_labeled)].copy()
    print(f"[Dataset] Labeled: {len(df_labeled):,} / {len(df):,} total wafer maps")

    # ── Extract string label from nested array ────────────────────────────────
    df_labeled["label_str"] = df_labeled["failureType"].apply(
        lambda x: str(x[0][0])
    )

    # ── Print class distribution ───────────────────────────────────────────────
    print("\n[Dataset] Full class distribution (labeled set):")
    for name in sorted(df_labeled["label_str"].unique()):
        n = int((df_labeled["label_str"] == name).sum())
        pct = 100.0 * n / len(df_labeled)
        bar = "█" * max(int(pct / 2), 1)
        print(f"  {name:<12}  {n:6,}  ({pct:5.1f}%)  {bar}")

    # ── Optionally exclude "none" class ───────────────────────────────────────
    if cfg.exclude_none:
        df_labeled = df_labeled[df_labeled["label_str"] != "none"].copy()
        print(f"\n[Dataset] After excluding 'none': {len(df_labeled):,} samples (8 classes)")

    # ── Map string label → integer index ─────────────────────────────────────
    # cfg.class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
    label_to_idx = {name: i for i, name in enumerate(cfg.class_names)}
    df_labeled["label"] = df_labeled["label_str"].apply(lambda s: label_to_idx[s])

    wafer_maps: List[np.ndarray] = df_labeled["waferMap"].tolist()
    labels: List[int] = df_labeled["label"].tolist()

    # ── Stratified 60 / 15 / 25 split ─────────────────────────────────────────
    indices = list(range(len(labels)))
    tv_idx, test_idx = train_test_split(
        indices, test_size=cfg.test_ratio, stratify=labels, random_state=cfg.seed,
    )
    tv_labels = [labels[i] for i in tv_idx]
    val_ratio_adj = cfg.val_ratio / (1.0 - cfg.test_ratio)
    train_idx, val_idx = train_test_split(
        tv_idx, test_size=val_ratio_adj, stratify=tv_labels, random_state=cfg.seed,
    )
    print(
        f"[Dataset] Split → train: {len(train_idx):,} | "
        f"val: {len(val_idx):,} | test: {len(test_idx):,}"
    )

    # ── Class weights for imbalanced training ─────────────────────────────────
    train_labels = [labels[i] for i in train_idx]
    counts = np.bincount(train_labels, minlength=cfg.num_classes).astype(float)
    w = 1.0 / np.maximum(counts, 1.0)
    w = (w / w.sum()) * cfg.num_classes          # normalise so weights sum to num_classes
    class_weights = torch.FloatTensor(w)

    # WeightedRandomSampler — each batch will be roughly class-balanced
    sample_weights = [w[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True,
    )

    # ── Build datasets ─────────────────────────────────────────────────────────
    def _subset(idx_list):
        return [wafer_maps[i] for i in idx_list], [labels[i] for i in idx_list]

    tr_maps, tr_lbl = _subset(train_idx)
    vl_maps, vl_lbl = _subset(val_idx)
    ts_maps, ts_lbl = _subset(test_idx)

    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    train_loader = DataLoader(
        WaferMapDataset(tr_maps, tr_lbl, cfg.image_size, augment=True),
        sampler=sampler, **loader_kw,
    )
    val_loader = DataLoader(
        WaferMapDataset(vl_maps, vl_lbl, cfg.image_size, augment=False),
        shuffle=False, **loader_kw,
    )
    test_loader = DataLoader(
        WaferMapDataset(ts_maps, ts_lbl, cfg.image_size, augment=False),
        shuffle=False, **loader_kw,
    )

    return train_loader, val_loader, test_loader, class_weights
