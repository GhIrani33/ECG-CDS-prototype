# -*- coding: utf-8 -*-
"""
Calibrate per-label thresholds on VAL for ECG-FM + ED2ED clinical screening.

Robust to various 'best_head.pt' formats:
- torch.save(head.state_dict())
- torch.save({'state_dict': head.state_dict(), ...})
- torch.save({'head': head}) OR {'head': head.state_dict()}
- torch.save({'model': {'head.net.0.weight': ...}} / with 'module.' prefixes)
It searches recursively for a state_dict that contains net.0.weight & fc_out.weight
(and strips common prefixes like 'head.' or 'module.' automatically).

Inputs:
  --dataset_dir .../screening_dataset_ed2ed
  --runs_dir    .../runs/ed2ed_k73_finetune
  --ckpt_pre    .../mimic_iv_ecg_physionet_pretrained.pt
  --head_path   .../runs/ed2ed_k73_finetune/best_head.pt

Outputs (in runs_dir/calib):
  thresholds.json
  val_scores_summary.csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wfdb


# -----------------------
# Model I/O
# -----------------------

def build_model_from_checkpoint(ckpt_path: str):
    from fairseq_signals.models import build_model_from_checkpoint as _build
    model = _build(ckpt_path)
    return model

def load_encoder(ckpt_path: str, device: torch.device):
    model = build_model_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model


class HeadMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, K: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.fc_out = nn.Linear(h2, K)

    def forward(self, x):
        return self.fc_out(self.net(x))


# -----------------------
# Robust checkpoint parsing
# -----------------------

_TENSOR_LIKE = (torch.Tensor,)

def _is_state_dict_like(obj) -> bool:
    if isinstance(obj, (dict, OrderedDict)):
        # at least one tensor inside, and keys look like module params
        any_tensor = any(isinstance(v, _TENSOR_LIKE) for v in obj.values())
        any_strkey = all(isinstance(k, str) for k in obj.keys())
        return any_tensor and any_strkey
    return False

def _collect_state_dicts(obj):
    """
    Recursively yield candidate state_dict-like dicts inside obj.
    Handles:
      - direct nn.Module
      - dicts with 'state_dict', 'model', 'head', 'module', arbitrary nesting
    """
    yielded = set()

    def _yield(sd):
        # use id to avoid duplicates
        sid = id(sd)
        if sid not in yielded and _is_state_dict_like(sd):
            yielded.add(sid)
            yield sd

    # Case 1: a module
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        sd = obj.state_dict()
        for x in _yield(sd):
            yield x
        # also inspect attributes that might hold nested modules
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                sub = getattr(obj, name)
            except Exception:
                continue
            if hasattr(sub, "state_dict") and callable(getattr(sub, "state_dict")):
                for x in _yield(sub.state_dict()):
                    yield x

    # Case 2: a dict (nested)
    if isinstance(obj, (dict, OrderedDict)):
        # This dict itself may be a state_dict
        for x in _yield(obj):
            yield x

        # Common keys that might contain state_dict/module
        for k in list(obj.keys()):
            sub = obj[k]
            # module -> take its state_dict
            if hasattr(sub, "state_dict") and callable(getattr(sub, "state_dict")):
                for x in _yield(sub.state_dict()):
                    yield x
            # nested dicts
            if isinstance(sub, (dict, OrderedDict)):
                for x in _collect_state_dicts(sub):
                    for y in _yield(x):
                        yield y

def _strip_common_prefix(keys, target_suffixes):
    """
    For keys like 'head.module.net.0.weight', 'head.module.fc_out.weight'
    find the shortest prefix p such that for each suffix s in target_suffixes
    there exists a key that endswith s and startswith p.
    Return p or ''.
    """
    # Find all candidate prefixes by locating the suffix start
    candidates = []
    for k in keys:
        for s in target_suffixes:
            if k.endswith(s):
                # prefix is k[:-len(s)], possibly empty
                candidates.append(k[: -len(s)] if len(s) > 0 else k)
    # choose the shortest non-conflicting prefix that works for all suffixes
    target_suffixes = list(target_suffixes)
    for p in sorted(set(candidates), key=len):
        ok = True
        for s in target_suffixes:
            if (p + s) not in keys:
                ok = False
                break
        if ok:
            return p
    return ""

def _extract_head_sd_from_any(obj):
    """
    Search recursively for a state_dict which has both net.0.weight and fc_out.weight
    (possibly with prefixes). Returns a cleaned sd with keys:
       'net.0.weight', 'net.0.bias', 'net.3.weight', 'net.3.bias', 'fc_out.weight', 'fc_out.bias'
    Missing keys (e.g., net.3.* in 1-hidden-layer heads) are allowed.
    """
    target_suffixes = ("net.0.weight", "fc_out.weight")
    for sd in _collect_state_dicts(obj):
        keys = list(sd.keys())
        # must contain the required suffixes somewhere
        if not any(k.endswith("net.0.weight") for k in keys):
            continue
        if not any(k.endswith("fc_out.weight") for k in keys):
            continue
        # strip common prefix (e.g., 'head.' / 'module.' / 'model.head.')
        p = _strip_common_prefix(keys, target_suffixes)
        def _unprefix(k):
            return k[len(p):] if p and k.startswith(p) else k
        cleaned = {}
        for k, v in sd.items():
            uk = _unprefix(k)
            if uk.startswith("net.") or uk.startswith("fc_out."):
                cleaned[uk] = v
        # must still contain the two anchor weights
        if "net.0.weight" in cleaned and "fc_out.weight" in cleaned:
            return cleaned

    # If nothing found, raise with a diagnostic of top-level structure
    raise KeyError("Could not locate head state_dict with 'net.0.weight' and 'fc_out.weight' in checkpoint.")

def _infer_head_dims(sd: dict):
    in_dim = sd["net.0.weight"].shape[1]
    h1 = sd["net.0.weight"].shape[0]
    if "net.3.weight" in sd:
        h2 = sd["net.3.weight"].shape[0]
    else:
        h2 = sd["fc_out.weight"].shape[1]
    K = sd["fc_out.weight"].shape[0]
    return in_dim, h1, h2, K

def load_head(head_path: str, enc_out_dim_expected: int, K_expected: int, device: torch.device):
    raw = torch.load(head_path, map_location="cpu")
    sd = _extract_head_sd_from_any(raw)
    in_dim_ckpt, h1_ckpt, h2_ckpt, K_ckpt = _infer_head_dims(sd)

    if in_dim_ckpt != enc_out_dim_expected:
        raise RuntimeError(f"Head enc_out_dim mismatch: ckpt={in_dim_ckpt} vs encoder={enc_out_dim_expected}.")
    if K_ckpt != K_expected:
        raise RuntimeError(f"Head K mismatch: ckpt={K_ckpt} vs dataset K={K_expected}.")

    head = HeadMLP(in_dim_ckpt, h1_ckpt, h2_ckpt, K_ckpt, p=0.2)
    # strict=True: حالا که دقیقاً از روی وزن‌ها معماری را ساختیم
    head.load_state_dict(sd, strict=True)
    head.to(device)
    head.eval()

    with torch.no_grad():
        pnorm = sum(p.norm().item() for p in head.parameters())
    print(f"[INFO] head loaded OK: in={in_dim_ckpt}, h1={h1_ckpt}, h2={h2_ckpt}, K={K_ckpt}, param_norm={pnorm:.3f}")
    return head


# -----------------------
# Data Pipeline (VAL)
# -----------------------

def _zscore_per_lead(x_ct: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x_ct, axis=1, keepdims=True)
    sd = np.nanstd(x_ct, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (x_ct - mu) / sd
    m = np.isfinite(z)
    if not m.all():
        z = np.where(m, z, 0.0)
    return z

def read_wfdb_pair_from_hea(hea_path: str) -> np.ndarray:
    base = os.path.splitext(hea_path)[0]
    sig, meta = wfdb.rdsamp(base)  # sig:(T,C)
    fs = float(meta.fs) if hasattr(meta, "fs") else float(meta["fs"])
    if int(fs) != 500:
        raise ValueError(f"unexpected sample_rate={fs} (expect 500) for {hea_path}")

    x_ct = sig.T.astype(np.float32)  # (C,T)
    C, T = x_ct.shape
    if C < 12:
        raise ValueError(f"expected 12-lead, got C={C} for {hea_path}")
    if T < 5000:
        raise ValueError(f"too short T={T}, need >=5000 for {hea_path}")

    s1 = _zscore_per_lead(x_ct[:, 0:2500])
    s2 = _zscore_per_lead(x_ct[:, 2500:5000])
    x2ct = np.stack([s1, s2], axis=0)  # (2, C, 2500)
    return x2ct[:2, :12, :2500].astype(np.float32)

class ValDataset(Dataset):
    def __init__(self, manifest_path: str, K: int, bad_csv: str = None):
        self.df = pd.read_parquet(manifest_path)
        for c in ("hea_path", "y_multi_hot"):
            if c not in self.df.columns:
                raise KeyError(f"manifest missing required column: '{c}'")
        if bad_csv and os.path.exists(bad_csv):
            bad = pd.read_csv(bad_csv)["hea"].astype(str).tolist()
            before = len(self.df)
            self.df = self.df[~self.df["hea_path"].isin(bad)].reset_index(drop=True)
            print(f"[INFO] VAL drop-bad: {before - len(self.df)} removed (from {before})")
        self.K = K

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        hea_path = r["hea_path"]
        y_arr = np.asarray(r["y_multi_hot"], dtype=np.float32)
        if y_arr.shape[-1] != self.K:
            raise ValueError(f"y length {y_arr.shape[-1]} != K={self.K} for {hea_path}")
        x2ct = read_wfdb_pair_from_hea(hea_path)
        return torch.from_numpy(x2ct), torch.from_numpy(y_arr), hea_path

def collate_fn(batch):
    xs, ys, paths = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,2,12,2500)
    y = torch.stack(ys, dim=0)  # (B,K)
    return x, y, list(paths)


# -----------------------
# Scoring & Thresholds
# -----------------------

@torch.no_grad()
def _extract_feats(encoder, x_bsct: torch.Tensor):
    """
    Accepts (B*S, C, T). Handles return types: Tensor / tuple / dict{'x': Tensor, ...}
    """
    out = encoder.extract_features(source=x_bsct, padding_mask=None, mask=False)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        if "x" in out and isinstance(out["x"], torch.Tensor):
            return out["x"]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Unexpected extract_features output type: {type(out)}")

@torch.no_grad()
def encode_batch(encoder, x2ct: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, S, C, T = x2ct.shape
    x = x2ct.view(B * S, C, T).to(device)
    feats = _extract_feats(encoder, x)  # (B*S, T', D)
    feats = feats.mean(dim=1)          # (B*S, D)
    feats = feats.view(B, S, -1).mean(dim=1)  # (B, D)
    return feats

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.5
    qs = np.linspace(0.01, 0.99, 199)
    cand = np.quantile(y_score, qs)
    best_thr, best_f1 = 0.5, -1.0
    for t in cand:
        yhat = (y_score >= t).astype(np.uint8)
        tp = (yhat & (y_true == 1)).sum()
        fp = (yhat & (y_true == 0)).sum()
        fn = ((1 - yhat) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)

def ppv_target_threshold(y_true: np.ndarray, y_score: np.ndarray, ppv_target: float, rec_min: float = 0.0) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    tp = 0
    Npos = int(y_true.sum())
    for i in range(len(s_sorted)):
        tp += int(y_sorted[i] == 1)
        prec = tp / (i + 1)
        rec  = tp / (Npos + 1e-9)
        thr = s_sorted[i]
        if prec >= ppv_target and rec >= rec_min:
            return float(thr)
    return best_f1_threshold(y_true, y_score)


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--runs_dir",    required=True)
    ap.add_argument("--ckpt_pre",    required=True)
    ap.add_argument("--head_path",   required=True)
    ap.add_argument("--batch_size",  type=int, default=48)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--thr_mode",    default="f1", choices=["f1"])
    ap.add_argument("--ppv_targets", type=float, nargs="*", default=[])
    ap.add_argument("--rec_min",     type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ds_dir  = Path(args.dataset_dir)
    runs    = Path(args.runs_dir)
    calib   = runs / "calib"
    calib.mkdir(parents=True, exist_ok=True)

    manifest_val  = str(ds_dir / "manifest_val.parquet")
    label_def_csv = str(ds_dir / "label_def.csv")

    df_ld = pd.read_csv(label_def_csv)
    label_codes = df_ld["code"].astype(str).tolist() if "code" in df_ld.columns else [f"L{i}" for i in range(len(df_ld))]
    K = len(df_ld)
    print(f"[INFO] K={K} labels (from {label_def_csv})")

    bad_val_csv = str(runs / "bad_val_heas.csv")
    ds = ValDataset(manifest_val, K=K, bad_csv=bad_val_csv)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )

    # Encoder + Head
    encoder = load_encoder(str(args.ckpt_pre), device=device)
    ENC_OUT_DIM = 768  # ECG-FM encoder dim
    head = load_head(str(args.head_path), enc_out_dim_expected=ENC_OUT_DIM, K_expected=K, device=device)
    sigmoid = nn.Sigmoid()

    # Score VAL
    all_scores, all_targets = [], []
    with torch.no_grad():
        for x2ct, yb, _ in tqdm(loader, desc="[VAL] scoring"):
            feats  = encode_batch(encoder, x2ct, device)   # (B, D)
            logits = head(feats)                           # (B, K)
            scores = sigmoid(logits).cpu().numpy()         # (B, K)
            y_true = yb.cpu().numpy().astype(np.float32)   # (B, K)
            all_scores.append(scores)
            all_targets.append(y_true)

    scores  = np.concatenate(all_scores, axis=0)   # (N, K)
    targets = np.concatenate(all_targets, axis=0)  # (N, K)
    N = scores.shape[0]
    print(f"[INFO] VAL scored: N={N}, K={K}")

    # Thresholds
    thr_f1  = np.zeros(K, dtype=np.float32)
    thr_ppv = {f"{t:.2f}": np.zeros(K, dtype=np.float32) for t in args.ppv_targets}

    for k in range(K):
        yk = targets[:, k].astype(np.uint8)
        sk = scores[:, k].astype(np.float32)
        thr_f1[k] = best_f1_threshold(yk, sk)
        for t in args.ppv_targets:
            thr_ppv[f"{t:.2f}"][k] = ppv_target_threshold(yk, sk, ppv_target=t, rec_min=args.rec_min)

    # Save JSON
    out_json = calib / "thresholds.json"
    payload = {
        "K": K,
        "label_codes": label_codes,
        "thr_f1": thr_f1.tolist(),
        "ppv_targets": args.ppv_targets,
        "thr_ppv": {k: v.tolist() for k, v in thr_ppv.items()},
        "rec_min": args.rec_min,
        "val_N": int(N),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {out_json}")

    # Save CSV summary
    df_sum = pd.DataFrame({
        "code": label_codes,
        "prev": targets.mean(axis=0),
        "score_med": np.median(scores, axis=0),
        "thr_f1": thr_f1,
    })
    for t_str, vec in thr_ppv.items():
        df_sum[f"thr_ppv_{t_str}"] = vec
    out_csv = calib / "val_scores_summary.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
