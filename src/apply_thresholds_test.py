import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# --- ECG-FM encoder loader ---
from fairseq_signals.models import build_model_from_checkpoint
# --- WFDB I/O ---
import wfdb

# ---------------- Utils ----------------

def zscore_np(x, axis=-1, eps=1e-6):
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    z = (x - mu) / sd
    return z

def read_wfdb_pair(hea_path, fs_expected=500, seg_len=2500):

    base = hea_path[:-4] if hea_path.lower().endswith(".hea") else hea_path
    sig, fields = wfdb.rdsamp(base)  # sig: (T,C)
    x = sig.astype(np.float32).T     # (C,T)
    C, T = x.shape

    need = seg_len * 2
    if T < need:
        if T <= 0:
            raise ValueError("empty signal")
        rep = int(np.ceil(need / T))
        x = np.tile(x, rep)[:, :need]
        T = x.shape[1]

    s1 = x[:, 0:seg_len]
    s2 = x[:, -seg_len:]

    s1 = zscore_np(s1, axis=1)
    s2 = zscore_np(s2, axis=1)

    s = np.stack([s1, s2], axis=0)  # (2,C,T)
    if not np.isfinite(s).all():
        raise ValueError("non-finite after zscore")
    return torch.from_numpy(s)  # float32

def safe_div(a, b):
    a = float(a); b = float(b)
    return a / b if b > 0 else float('nan')

def f1_from_pr(ppv, rec):
    if (ppv is None) or (rec is None): return float('nan')
    if (ppv + rec) == 0: return float('nan')
    return 2.0 * ppv * rec / (ppv + rec)

# ---------------- Head MLP ----------------

class HeadMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, K: int, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h2, K)  # "fc_out"
        )
    def forward(self, x):
        return self.net(x)

def _extract_head_sd_from_any(raw_obj: dict) -> dict:
    
    if not isinstance(raw_obj, dict):
        raise TypeError("head checkpoint must be a dict (state_dict or wrapper dict)")
    cand = None
    if any(k.startswith("net.") for k in raw_obj.keys()):
        cand = raw_obj
    elif "head" in raw_obj and isinstance(raw_obj["head"], dict):
        cand = raw_obj["head"]
    if cand is None:
        raise KeyError("Could not locate head state_dict (expect top-level 'net.*' or wrapper['head']).")
    if not any(k == "net.0.weight" for k in cand.keys()):
        raise KeyError("State_dict lacks 'net.0.weight'.")
    if not any(k == "net.6.weight" or k == "fc_out.weight" for k in cand.keys()):
        raise KeyError("State_dict lacks final layer weight ('net.6.weight' or 'fc_out.weight').")
    if ("fc_out.weight" in cand) or ("fc_out.bias" in cand):
        remap = {}
        for k, v in cand.items():
            if k.startswith("fc_out."):
                remap["net.6." + k.split(".", 1)[1]] = v
            else:
                remap[k] = v
        cand = remap
    return cand

def infer_head_dims_from_sd(sd: dict):
    in_dim = sd["net.0.weight"].shape[1]
    h1     = sd["net.0.weight"].shape[0]
    h2     = sd["net.3.weight"].shape[0]
    last_w = sd["net.6.weight"] if "net.6.weight" in sd else sd["fc_out.weight"]
    K      = last_w.shape[0]
    return int(in_dim), int(h1), int(h2), int(K)

# ---------------- Encoder wrapper ----------------

@torch.no_grad()
def encode_batch(encoder, x2ct: torch.Tensor, device: torch.device) -> torch.Tensor:
    """x2ct: (B,2,12,2500) -> (B,D) """
    B, S, C, T = x2ct.shape
    x = x2ct.reshape(B * S, C, T).to(device, non_blocking=True)

    out = encoder.extract_features(source=x, padding_mask=None, mask=False)

    if isinstance(out, (list, tuple)):
        feat_t = out[0]
    elif isinstance(out, dict):
        if "x" in out and torch.is_tensor(out["x"]):
            feat_t = out["x"]
        elif "features" in out and torch.is_tensor(out["features"]):
            feat_t = out["features"]
        else:
            raise RuntimeError(f"Unsupported extract_features() dict keys: {list(out.keys())}")
    elif torch.is_tensor(out):
        feat_t = out
    else:
        raise RuntimeError(f"Unsupported extract_features() return type: {type(out)}")

    if feat_t.dim() == 3:
        feat = feat_t.mean(dim=1)      # (B*2,D)
    elif feat_t.dim() == 2:
        feat = feat_t                  # (B*2,D)
    else:
        raise RuntimeError(f"Unexpected feature tensor shape: {tuple(feat_t.shape)}")

    feat = feat.reshape(B, S, -1).mean(dim=1)  # (B,D)
    return feat

# ---------------- Dataset (TEST) ----------------

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str, drop_bad: bool = True, bad_list_csv: str = None):
        df = pd.read_parquet(manifest_path)
        initial = len(df)
        self.bad_set = set()
        if drop_bad and bad_list_csv and os.path.exists(bad_list_csv):
            try:
                bad_df = pd.read_csv(bad_list_csv)
                # ستون استانداردی که قبلاً ساختیم: 'hea_path'
                if "hea_path" in bad_df.columns:
                    self.bad_set = set(bad_df["hea_path"].astype(str).tolist())
                else:
    
                    for col in bad_df.columns:
                        if bad_df[col].dtype == object:
                            self.bad_set |= set(bad_df[col].astype(str).tolist())
                before = len(df)
                df = df[~df["hea_path"].astype(str).isin(self.bad_set)]
                after = len(df)
                print(f"[INFO] bad_test_heas.csv found: removed {before - after} / {before} rows")
            except Exception as e:
                print(f"[WARN] Failed to apply bad_test_heas filter: {e}")
        else:
            if drop_bad:
                print(f"[INFO] bad_test_heas.csv not found at: {bad_list_csv}")
        self.df = df.reset_index(drop=True)
        print(f"[INFO] TEST manifest size: {initial} -> {len(self.df)} after initial filter")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        hea_path = str(r["hea_path"])
        try:
            x2ct = read_wfdb_pair(hea_path)         # (2,12,2500)
            y = np.array(r["y_multi_hot"], dtype=np.int64, copy=True)
            return x2ct, torch.from_numpy(y), hea_path
        except Exception as e:
            return None

def collate(batch):
    xs, ys, paths = [], [], []
    for item in batch:
        if item is None:
            continue
        x, y, p = item
        xs.append(x)
        ys.append(y)
        paths.append(p)
    if len(xs) == 0:
        return None  #ی
    x = torch.stack(xs, dim=0)            # (B,2,12,2500)
    y = torch.stack(ys, dim=0).float()    # (B,K)
    return x, y, paths

# ---------------- Metrics ----------------

def per_label_counts(y_true: np.ndarray, y_pred: np.ndarray):
    TP = (y_true * y_pred).sum(axis=0)
    FP = ((1 - y_true) * y_pred).sum(axis=0)
    FN = (y_true * (1 - y_pred)).sum(axis=0)
    TN = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    return TP, FP, FN, TN

def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list):
    TP, FP, FN, TN = per_label_counts(y_true, y_pred)
    N = y_true.shape[0]
    rows = []
    for i, lab in enumerate(labels):
        tp, fp, fn, tn = [int(TP[i]), int(FP[i]), int(FN[i]), int(TN[i])]
        prev = (tp + fn) / N
        ppv  = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = f1_from_pr(ppv, rec)
        rows.append({
            "label": lab,
            "prevalence": prev,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "PPV": ppv, "Recall": rec, "F1": f1
        })
    df = pd.DataFrame(rows)
    macro = {
        "macro": {
            "PPV": float(np.nanmean(df["PPV"].values)),
            "Recall": float(np.nanmean(df["Recall"].values)),
            "F1": float(np.nanmean(df["F1"].values)),
        }
    }
    TPm, FPm, FNm, TNm = TP.sum(), FP.sum(), FN.sum(), TN.sum()
    ppv_mi = safe_div(TPm, TPm + FPm)
    rec_mi = safe_div(TPm, TPm + FNm)
    f1_mi  = f1_from_pr(ppv_mi, rec_mi)
    micro = {"micro": {"PPV": ppv_mi, "Recall": rec_mi, "F1": f1_mi}}
    return df, macro, micro

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--runs_dir",    required=True)
    ap.add_argument("--ckpt_pre",    required=True, help="ECG-FM pretrained checkpoint (.pt)")
    ap.add_argument("--head_path",   required=True, help="best_head_state_dict_fc.pt یا best_head.pt")
    ap.add_argument("--thresholds",  default=None, help="defaults to runs_dir/calib/thresholds.json")
    ap.add_argument("--batch_size",  type=int, default=48)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    runs_dir    = Path(args.runs_dir)
    thr_path    = Path(args.thresholds) if args.thresholds else (runs_dir / "calib" / "thresholds.json")
    out_dir     = runs_dir / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels & K
    label_def = pd.read_csv(dataset_dir / "label_def.csv")
    if "label" in label_def.columns:
        labels = label_def["label"].astype(str).tolist()
    elif "code" in label_def.columns:
        labels = label_def["code"].astype(str).tolist()
    else:
        raise KeyError("label_def.csv must have 'label' or 'code' column")
    K = len(labels)
    print(f"[INFO] K={K} labels loaded.")

    # Load thresholds.json
    with open(thr_path, "r", encoding="utf-8") as f:
        thr_obj = json.load(f)
    thr_f1   = np.array(thr_obj.get("thr_f1", [0.5]*K), dtype=float)
    thr_ppv  = thr_obj.get("thr_ppv", {})
    thr_p50  = np.array(thr_ppv.get("0.50", [0.5]*K), dtype=float)
    thr_p30  = np.array(thr_ppv.get("0.30", [0.5]*K), dtype=float)
    if not (len(thr_f1)==K and len(thr_p50)==K and len(thr_p30)==K):
        raise ValueError("thresholds.json K mismatch with label_def.csv")
    print(f"[INFO] thresholds loaded from: {thr_path}")

    # Encoder
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    encoder = build_model_from_checkpoint(str(args.ckpt_pre))
    encoder.to(device).eval()
    print(f"[INFO] Encoder ready on {device}, type={type(encoder)}")

    # Head (robust load)
    raw = torch.load(str(args.head_path), map_location="cpu")
    sd  = _extract_head_sd_from_any(raw)
    in_dim, h1, h2, K_sd = infer_head_dims_from_sd(sd)
    if K_sd != K:
        raise ValueError(f"Head K ({K_sd}) != labels K ({K})")
    head = HeadMLP(in_dim, h1, h2, K)
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        raise RuntimeError(f"Head load mismatch. missing={missing}, unexpected={unexpected}")
    head.to(device).eval()
    print(f"[INFO] Head ready: in={in_dim}, h1={h1}, h2={h2}, K={K}")

    # Dataset / loader
    manifest_test = dataset_dir / "manifest_test.parquet"
    bad_test_csv  = runs_dir / "bad_test_heas.csv"
    ds = TestDataset(str(manifest_test), drop_bad=True, bad_list_csv=str(bad_test_csv))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    print(f"[INFO] TEST size after filter: N={len(ds)}")
    skipped = 0

    # Score TEST
    all_logits = []
    all_y      = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[TEST] scoring"):
            if batch is None:
                skipped += args.batch_size  
                continue
            x2ct, yb, _paths = batch
            if x2ct.numel() == 0:
                continue
            feats = encode_batch(encoder, x2ct, device)   # (B,D)
            logits = head(feats)                          # (B,K)
            all_logits.append(logits.cpu())
            all_y.append(yb)

    if len(all_logits) == 0:
        raise RuntimeError("No valid TEST samples to score.")

    logits = torch.cat(all_logits, dim=0).float().numpy()    # (N,K)
    y_true = torch.cat(all_y,      dim=0).float().numpy()    # (N,K)
    probs  = 1.0 / (1.0 + np.exp(-logits))                   # sigmoid

    def eval_and_save(thr_vec: np.ndarray, tag: str):
        y_pred = (probs >= thr_vec.reshape(1, -1)).astype(np.int64)
        df, macro, micro = per_label_metrics(y_true, y_pred, labels)
        out_csv = out_dir / f"test_per_label_{tag}.csv"
        df.to_csv(out_csv, index=False)
        return {"tag": tag, "macro": macro["macro"], "micro": micro["micro"], "csv": str(out_csv)}

    summary = {
        "N_test_after_filter": int(y_true.shape[0]),
        "K": K,
        "skipped_samples_est": int(skipped),
        "reports": []
    }
    summary["reports"].append(eval_and_save(thr_f1,  "f1"))
    summary["reports"].append(eval_and_save(thr_p50, "ppv50"))
    summary["reports"].append(eval_and_save(thr_p30, "ppv30"))

    out_json = out_dir / "test_macro_micro.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {out_json}")
    for r in summary["reports"]:
        print(f"[OK] wrote {r['csv']}  (macro={r['macro']}, micro={r['micro']})")

if __name__ == "__main__":
    main()
