# -*- coding: utf-8 -*-
"""
Fine-tune ECG-FM (ED2ED, K=73 default) — Head-Only (default)

Data assumptions (as built earlier):
  D:\Project\ECG\ecg-cds-ed2ed\screening_dataset_ed2ed\
    - manifest_{train,val,test}.parquet  (contains hea_path, y_multi_hot)
    - label_def.csv  (column 'label')
    - pos_weight.npy
Optional bad-record CSVs (if exist):
  D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed_k73_finetune\bad_{train,val,test}_heas.csv

Usage (PowerShell):
  python "D:\Project\ECG\ecg-cds-ed2ed\finetune_ed2ed_ecgfm.py" `
    --dataset_dir "D:\Project\ECG\ecg-cds-ed2ed\screening_dataset_ed2ed" `
    --runs_dir    "D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed_k73_finetune" `
    --ckpt_pre    "D:\Project\ECG\ecg-cds-ed2ed\mimic_iv_ecg_physionet_pretrained.pt" `
    --epochs 20 --batch_size 64 --lr_head 1e-3 --use_asl 1

"""
import os, json, math, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from fairseq_signals.models import build_model_from_checkpoint
torch.backends.cudnn.benchmark = True

# --------------------
# Utils
# --------------------
def _norm(p): return os.path.normcase(os.path.normpath(p))
def set_seed(seed=1337):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zscore(x, eps=1e-7):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + eps
    return (x - mu) / sd

def read_wfdb_pair(hea_path, seg_len=2500):
    rec = wfdb.rdrecord(os.path.splitext(hea_path)[0])
    sig = rec.p_signal.T.astype(np.float32)         # (C, T_raw)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    need = 2 * seg_len
    if sig.shape[1] < need:
        pad = np.repeat(sig[:, -1:], repeats=(need - sig.shape[1]), axis=1)
        sig = np.concatenate([sig, pad], axis=1)
    s1 = zscore(sig[:, :seg_len])
    s2 = zscore(sig[:, seg_len: seg_len*2])
    x  = np.stack([s1, s2], axis=0).astype(np.float32)  # (2, 12, 2500)
    x  = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def load_exclusions(csv_path: Path):
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col = "hea_path" if "hea_path" in df.columns else df.columns[0]
        return set(_norm(s) for s in df[col].astype(str))
    return set()

# --------------------
# Dataset
# --------------------
class ED2EDDataset(Dataset):
    def __init__(self, manifest_path: Path, exclude_set=None):
        df = pd.read_parquet(manifest_path)
        if exclude_set:
            df = df.loc[~df["hea_path"].astype(str).apply(_norm).isin(exclude_set)].reset_index(drop=True)
        self.df = df

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hea = row["hea_path"]
        x   = read_wfdb_pair(hea)                         # (2,12,2500)
        # y_multi_hot stored as JSON-like string or list
        y   = row["y_multi_hot"]
        if isinstance(y, str):
            import ast; y = ast.literal_eval(y)
        y = np.asarray(y, dtype=np.float32)
        if not np.isfinite(x).all():  # just in case
            raise ValueError("non-finite after zscore")
        return x, y

def collate_bsct(batch):
    xs = [torch.from_numpy(b[0]) for b in batch]  # (2,12,2500)
    ys = [torch.from_numpy(b[1]) for b in batch]
    x  = torch.stack(xs, dim=0)                   # (B,2,12,2500)
    y  = torch.stack(ys, dim=0)                   # (B,K)
    return x, y

# --------------------
# Head (MLP) and Losses
# --------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim:int, k:int, h1:int=1024, h2:int=512, p:float=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h2, k),
        )
    def forward(self, x): return self.net(x)

class AsymmetricLoss(nn.Module):
    # ASL from: https://arxiv.org/abs/2009.14119 (multi-label)
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps
    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid; xs_neg = 1.0 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)
        loss_pos = -torch.pow(1 - pt_pos, self.gp) * torch.log(pt_pos.clamp(min=self.eps))
        loss_neg = -torch.pow(1 - pt_neg, self.gn) * torch.log(pt_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg
        return loss.mean()

# --------------------
# Encoder feature extraction (BSxCT)
# --------------------
def encode_bsct(backbone, x_bsct, device):
    # x: (B,2,12,2500) -> (B*2, 12, 2500)
    B,S,C,T = x_bsct.shape
    inp = x_bsct.reshape(B*S, C, T).to(device)
    pm  = torch.zeros((B*S, T), dtype=torch.bool, device=device)
    with torch.no_grad():
        out = backbone.extract_features(source=inp, padding_mask=pm, mask=False)
    if isinstance(out, dict):
        y = out.get("x", None)
        if y is None:
            # fallback: first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    y = v; break
    elif isinstance(out, (tuple, list)):
        y = out[0]
    else:
        y = out
    # y: (B*S, T', D) or (B*S, D, T') → میانگین زمانی
    if y.dim() == 3 and y.shape[1] > y.shape[2]:  # (B*S, D, T')
        y = y.transpose(1, 2)
    feat = y.mean(dim=1)             # (B*S, D)
    feat = feat.reshape(B, S, -1).mean(dim=1)  # (B, D)
    return feat

# --------------------
# L2-SP regularizer (optional)
# --------------------
def capture_init_weights(modules):
    init = {}
    for name, p in modules:
        if p.requires_grad:
            init[name] = p.detach().clone()
    return init

def l2sp_penalty(modules, init_dict, alpha):
    if alpha <= 0: return 0.0
    reg = 0.0
    for name, p in modules:
        if p.requires_grad and name in init_dict:
            reg = reg + (p - init_dict[name]).pow(2).sum()
    return alpha * reg

# --------------------
# Metrics
# --------------------
@torch.no_grad()
def evaluate(loader, backbone, head, device):
    head.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        feat = encode_bsct(backbone, xb, device)     # (B, D)
        logits = head(feat)                          # (B, K)
        prob = torch.sigmoid(logits)
        ys.append(yb.cpu().numpy()); ps.append(prob.cpu().numpy())
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    # macro AUROC/AUPRC
    aurocs, auprcs = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]; p = P[:, j]
        if y.max() > 0 and y.min() < 1:
            try: aurocs.append(roc_auc_score(y, p))
            except: pass
            try: auprcs.append(average_precision_score(y, p))
            except: pass
    macro_auroc = float(np.nanmean(aurocs)) if aurocs else float("nan")
    macro_auprc = float(np.nanmean(auprcs)) if auprcs else float("nan")
    return {"macro_auroc": macro_auroc, "macro_auprc": macro_auprc}

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=r"D:\Project\ECG\ecg-cds-ed2ed\screening_dataset_ed2ed")
    ap.add_argument("--runs_dir",    default=r"D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed_k73_finetune")
    ap.add_argument("--ckpt_pre",    default=r"D:\Project\ECG\ecg-cds-ed2ed\mimic_iv_ecg_physionet_pretrained.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)   # Windows-safe
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_enc",  type=float, default=0.0)   # 0 → فریز کامل
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_asl", type=int, default=1)
    ap.add_argument("--asl_gp", type=float, default=1.0)
    ap.add_argument("--asl_gn", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--unfreeze", type=str, default="")     # e.g. "encoder.layers.10,encoder.layers.11,encoder.layer_norm"
    ap.add_argument("--l2sp", type=float, default=0.0)      # e.g. 1e-4
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN = Path(args.runs_dir); RUN.mkdir(parents=True, exist_ok=True)

    # Load labels & pos_weight
    labels = pd.read_csv(Path(args.dataset_dir)/"label_def.csv")["label"].tolist()
    K = len(labels)
    pos_weight = torch.from_numpy(np.load(Path(args.dataset_dir)/"pos_weight.npy")).float().to(device)

    # Exclusions (if any)
    excl_train = load_exclusions(RUN/"bad_train_heas.csv")
    excl_val   = load_exclusions(RUN/"bad_val_heas.csv")
    excl_test  = load_exclusions(RUN/"bad_test_heas.csv")

    # Datasets & Loaders
    dtr = ED2EDDataset(Path(args.dataset_dir)/"manifest_train.parquet", exclude_set=excl_train)
    dva = ED2EDDataset(Path(args.dataset_dir)/"manifest_val.parquet",   exclude_set=excl_val)
    dte = ED2EDDataset(Path(args.dataset_dir)/"manifest_test.parquet",  exclude_set=excl_test)

    train_loader = DataLoader(dtr, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_bsct)
    val_loader   = DataLoader(dva, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_bsct)
    test_loader  = DataLoader(dte, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_bsct)

    # Backbone
    backbone = build_model_from_checkpoint(args.ckpt_pre).to(device).eval()
    print("[INFO] ECG-FM loaded via fairseq_signals.models.build_model_from_checkpoint.")

    # Detect feature dim
    xb, yb = next(iter(train_loader))
    with torch.no_grad():
        feat = encode_bsct(backbone, xb.to(device), device)
    D = int(feat.shape[1])
    print(f"[INFO] encoder feat dim = {D}")

    # Head
    head = MLPHead(D, K, h1=1024, h2=512, p=0.3).to(device)
    # Loss
    if args.use_asl:
        criterion = AsymmetricLoss(gamma_pos=args.asl_gp, gamma_neg=args.asl_gn, clip=args.asl_clip)
        print(f"[INFO] Using ASL(gp={args.asl_gp}, gn={args.asl_gn}, clip={args.asl_clip})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("[INFO] Using BCEWithLogits(pos_weight)")

    # Freeze all encoder by default
    for p in backbone.parameters():
        p.requires_grad = False

    # Optional Partial-Unfreeze
    unfrozen_param_names = []
    init_snapshot = {}
    if args.lr_enc > 0.0 and args.unfreeze.strip():
        allow = [u.strip() for u in args.unfreeze.split(",") if u.strip()]
        for name, module in backbone.named_modules():
            if any(name == a or name.startswith(a + ".") for a in allow):
                for pn, p in module.named_parameters(recurse=True):
                    p.requires_grad = True
                    fullname = f"{name}.{pn}" if name else pn
                    unfrozen_param_names.append(fullname)
        if args.l2sp > 0.0:
            init_snapshot = capture_init_weights(
                [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
            )
        print(f"[INFO] Partial-Unfreeze enabled on {len(unfrozen_param_names)} params; LR_enc={args.lr_enc}, L2-SP={args.l2sp}")
    else:
        print("[INFO] Encoder frozen (Head-only).")

    # Optimizer with two groups
    params = [{"params": head.parameters(), "lr": args.lr_head}]
    if args.lr_enc > 0.0 and unfrozen_param_names:
        params.append({"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_enc})
    optim = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val = -1.0
    best_path = RUN/"best_head.pt"

    def run_epoch(loader, train=True):
        nonlocal init_snapshot
        head.train(mode=train)
        running = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                feat = encode_bsct(backbone, xb, device)     # (B, D)
                logits = head(feat)
                loss = criterion(logits, yb)
                # L2-SP penalty (only when encoder partially unfrozen)
                if args.lr_enc > 0.0 and args.l2sp > 0.0 and init_snapshot:
                    reg = l2sp_penalty([(n,p) for n,p in backbone.named_parameters() if p.requires_grad],
                                       init_snapshot, args.l2sp)
                    loss = loss + reg
            if train:
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                # also clip encoder if unfrozen
                if args.lr_enc > 0.0 and unfrozen_param_names:
                    nn.utils.clip_grad_norm_([p for p in backbone.parameters() if p.requires_grad], args.grad_clip)
                scaler.step(optim)
                scaler.update()
            running += loss.detach().item() * xb.shape[0]
            n += xb.shape[0]
        return running / max(n,1)

    t0 = time.time()
    for ep in range(1, args.epochs+1):
        tr_loss = run_epoch(train_loader, train=True)
        val_metrics = evaluate(val_loader, backbone, head, device)
        macro_auroc = val_metrics["macro_auroc"]
        macro_auprc = val_metrics["macro_auprc"]
        sched.step(macro_auprc if not math.isnan(macro_auprc) else 0.0)

        # Log
        for i, g in enumerate(optim.param_groups):
            print(f"[EP {ep:02d}] loss={tr_loss:.4f}  val/macro(AUROC={macro_auroc:.4f}, AUPRC={macro_auprc:.4f})  lr[{i}]={g['lr']:.2e}")

        # Save best head only
        if macro_auprc > best_val:
            best_val = macro_auprc
            torch.save({
                "head": head.state_dict(),
                "feat_dim": D,
                "K": K,
                "labels": labels
            }, best_path)
            print(f"[SAVE] best_head.pt (macro-AUPRC={macro_auprc:.4f})")

    # TEST with best
    state = torch.load(best_path, map_location="cpu")
    head.load_state_dict(state["head"], strict=False)
    test_metrics = evaluate(test_loader, backbone, head, device)
    print(f"[TEST] macro(AUROC={test_metrics['macro_auroc']:.4f}, AUPRC={test_metrics['macro_auprc']:.4f})")
    print(f"[DONE] total_time={(time.time()-t0)/60:.1f}m")

if __name__ == "__main__":
    main()
