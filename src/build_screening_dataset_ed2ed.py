# -*- coding: utf-8 -*-
"""
End-to-end data & eval pipeline for ED2ED screening on MIMIC-IV-ECG.
- Loads manifest_{train|val|test}.parquet + label_def.csv + pos_weight.npy
- Reads WFDB records from absolute paths, resamples to 500 Hz if needed
- Per-lead Z-score normalization
- Splits each 10s ECG into two 5s segments [0:5s], [5:10s]
- Provides PyTorch Dataset/DataLoader ready for finetuning
- Includes evaluation utilities (AUROC/AUPRC macro/micro, per-label)
- Includes segment→ECG aggregation (max/mean) utilities for logits

Nothing is written into the dataset folders; all outputs under D:\Project\ECG\ecg-cds-ed2ed\runs\...
"""

import os, json, math, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# -----------------------------
# CONFIG (edit if needed)
# -----------------------------
ED2ED_ROOT = Path(r"D:\Project\ECG\ecg-cds-ed2ed\screening_dataset_ed2ed")
ALL2ALL_ROOT = Path(r"D:\Project\ECG\ecg-cds-ed2ed\screening_dataset")  # optional

RUNS_DIR = Path(r"D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# IO performance
NUM_WORKERS = min(8, os.cpu_count() or 4)
PIN_MEMORY = True

# Signal/segment parameters
TARGET_FS = 500        # Hz
SEG_SEC = 5.0          # seconds
N_SEGS = 2             # 10s -> two non-overlapping 5s
N_SAMPLES = int(TARGET_FS * SEG_SEC)  # 2500 samples per segment
N_LEADS_EXPECTED = 12  # standard 12-lead ECG

# -----------------------------
# Utilities
# -----------------------------
def _read_wfdb(dat_path: str, hea_path: str) -> Tuple[np.ndarray, int, List[str]]:
    """
    Read WFDB record given absolute .dat and .hea paths.
    Returns: (signal [T, C], fs, lead_names)
    """
    # wfdb expects "record" path without extension
    rec_stem = os.path.splitext(hea_path)[0]
    rec = wfdb.rdrecord(rec_stem, sampto=None, channels=None)  # lazy read
    fs = int(round(rec.fs))
    sig = rec.p_signal.astype(np.float32)  # [T, C]
    leads = [str(ch) for ch in rec.sig_name]
    return sig, fs, leads

def _resample_to_500(sig: np.ndarray, fs: int) -> np.ndarray:
    """Resample (time, channels) to 500 Hz using polyphase (exact if fs divisible)."""
    if fs == TARGET_FS:
        return sig
    # rational approximation
    # up/down chosen to get close to 500
    g = math.gcd(fs, TARGET_FS)
    up = TARGET_FS // g
    down = fs // g
    # axis=0 -> time axis
    out = resample_poly(sig, up, down, axis=0).astype(np.float32)
    return out

def _zscore_per_lead(sig: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((sig - mu) / sd).astype(np.float32)

def _split_two_segments(sig_10s: np.ndarray) -> np.ndarray:
    """
    Splits a 10s signal (T x C at 500 Hz) into two 5s segments.
    Returns [2, C, 2500]
    """
    T, C = sig_10s.shape
    need = N_SEGS * N_SAMPLES  # 5000
    if T < need:
        # pad by repeating last sample
        pad = np.repeat(sig_10s[-1:, :], repeats=(need - T), axis=0)
        sig_10s = np.concatenate([sig_10s, pad], axis=0)
    elif T > need:
        sig_10s = sig_10s[:need, :]
    seg1 = sig_10s[:N_SAMPLES, :]       # [2500, C]
    seg2 = sig_10s[N_SAMPLES:need, :]   # [2500, C]
    out = np.stack([seg1.T, seg2.T], axis=0).astype(np.float32)  # [2, C, 2500]
    return out

def _load_label_assets(root: Path) -> Tuple[List[str], Dict[str,int], np.ndarray]:
    labels = pd.read_csv(root / "label_def.csv")["label"].tolist()
    with open(root / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    pos_weight = np.load(root / "pos_weight.npy").astype(np.float32)
    return labels, label_map, pos_weight

# -----------------------------
# Dataset
# -----------------------------
class ECGBinaryMultiLabelDataset(Dataset):
    """
    Loads ECGs from manifest parquet:
      - reads absolute dat/hea
      - resamples to 500 Hz
      - per-lead Z-score
      - splits into 2x5s segments -> returns tensor [2, 12, 2500]
      - y: multi-hot vector [K]
    """
    def __init__(self, manifest_pq: Path, label_map: Dict[str, int], n_labels: int):
        self.df = pd.read_parquet(manifest_pq)
        # minimal columns
        need = {"dat_path", "hea_path", "y_multi_hot"}
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"Missing columns in {manifest_pq.name}: {miss}")
        self.label_map = label_map
        self.n_labels = n_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sig, fs, leads = _read_wfdb(row["dat_path"], row["hea_path"])
        if sig.ndim != 2:  # (T, C)
            raise RuntimeError(f"Bad WFDB shape at idx {idx}: {sig.shape}")
        sig = _resample_to_500(sig, fs)
        if sig.shape[1] != N_LEADS_EXPECTED:
            # keep going but warn once
            warnings.warn(f"Non-12-lead record: got {sig.shape[1]} leads", RuntimeWarning)
        sig = _zscore_per_lead(sig)         # (T, C)
        x = _split_two_segments(sig)        # (2, C, 2500)
        # labels (list[int] or python list)
        y_list = row["y_multi_hot"]
        # ensure vector length == K
        y = np.array(y_list, dtype=np.int64)
        if y.shape[0] != self.n_labels:
            raise RuntimeError(f"Label length mismatch: got {y.shape[0]} vs {self.n_labels}")
        return torch.from_numpy(x), torch.from_numpy(y)

def make_loader(manifest_pq: Path, label_map: Dict[str,int], n_labels: int,
                batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = ECGBinaryMultiLabelDataset(manifest_pq, label_map, n_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=drop_last)

# -----------------------------
# Metrics & aggregation
# -----------------------------
def _stack_preds(probs_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(probs_list, axis=0)

def aggregate_segments(logits_2seg: torch.Tensor, how: str = "max") -> torch.Tensor:
    """
    logits_2seg: [B, 2, K] -> returns [B, K]
    """
    if how == "max":
        return logits_2seg.max(dim=1).values
    elif how == "mean":
        return logits_2seg.mean(dim=1)
    else:
        raise ValueError("how must be 'max' or 'mean'")

@torch.no_grad()
def evaluate_multilabel(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    y_true: [N, K] integers 0/1
    y_score: [N, K] floats in [0,1]
    Returns: dict with macro/micro AUROC/AUPRC and per-label lists
    """
    N, K = y_true.shape
    assert y_score.shape == (N, K)
    per_label_auroc, per_label_auprc = [], []
    # per-label
    for k in range(K):
        yk = y_true[:, k]
        sk = y_score[:, k]
        if yk.sum() == 0 or (yk == 0).sum() == 0:
            per_label_auroc.append(np.nan)
        else:
            per_label_auroc.append(roc_auc_score(yk, sk))
        # AP tolerates all-negative
        per_label_auprc.append(average_precision_score(yk, sk))
    # macro
    auroc_macro = np.nanmean(per_label_auroc)
    auprc_macro = np.nanmean(per_label_auprc)
    # micro (flatten)
    auroc_micro = np.nan
    try:
        auroc_micro = roc_auc_score(y_true.ravel(), y_score.ravel())
    except Exception:
        pass
    auprc_micro = average_precision_score(y_true.ravel(), y_score.ravel())
    return {
        "auroc_macro": float(auroc_macro),
        "auprc_macro": float(auprc_macro),
        "auroc_micro": float(auroc_micro) if not math.isnan(auroc_micro) else None,
        "auprc_micro": float(auprc_micro),
        "per_label_auroc": per_label_auroc,
        "per_label_auprc": per_label_auprc,
    }

# -----------------------------
# Head & loss (ready for use when encoder is added)
# -----------------------------
class LinearHead(torch.nn.Module):
    """
    Simple linear classification head that maps encoder features -> K labels.
    Use once you have encoder outputs [B, 2, D]. You can pool over segments first.
    """
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_labels)

    def forward(self, features_2seg: torch.Tensor, pool: str = "max") -> torch.Tensor:
        """
        features_2seg: [B, 2, D]
        Returns logits [B, K] after segment pooling
        """
        if pool == "max":
            f = features_2seg.max(dim=1).values
        elif pool == "mean":
            f = features_2seg.mean(dim=1)
        else:
            raise ValueError("pool must be 'max' or 'mean'")
        return self.fc(f)

def make_bcewithlogits(pos_weight: np.ndarray, device: torch.device):
    w = torch.from_numpy(pos_weight).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=w)

# -----------------------------
# Smoke test (dataset only)
# -----------------------------
def smoke_test(root: Path, batch_size: int = 8):
    labels, label_map, pos_weight = _load_label_assets(root)
    K = len(labels)
    man_train = root / "manifest_train.parquet"
    man_val   = root / "manifest_val.parquet"

    loader = make_loader(man_train, label_map, K, batch_size=batch_size, shuffle=True, drop_last=False)
    x, y = next(iter(loader))
    print("[OK] batch:", x.shape, y.shape)  # [B, 2, C, 2500], [B, K]
    # Save one mini-batch to disk for debug
    out_dir = RUNS_DIR / "debug_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x.npy", x.numpy())
    np.save(out_dir / "y.npy", y.numpy())
    print("[INFO] wrote", out_dir)

if __name__ == "__main__":
    # Default smoke-test on ED2ED; you can switch to ALL2ALL_ROOT if needed.
    smoke_test(ED2ED_ROOT, batch_size=4)
