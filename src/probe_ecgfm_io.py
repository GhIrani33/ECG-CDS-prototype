# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import torch
from fairseq_signals.models import build_model_from_checkpoint

# ===== Paths =====
CKPT = r"D:\Project\ECG\ecg-cds-ed2ed\mimic_iv_ecg_physionet_pretrained.pt"
DEBUG_X = r"D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed\debug_batch\x.npy"
OUT = Path(r"D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed\io_probe")
OUT.mkdir(parents=True, exist_ok=True)

def _extract_feat_from_output(out):
    """
    Robustly extract the feature tensor from model output.
    - dict: try 'x' / 'features' / 'feature', else first Tensor-like value
    - tuple/list: take [0]
    - tensor: return as-is
    """
    import torch
    if isinstance(out, dict):
        for k in ("x", "features", "feature"):
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        # fallback: first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise RuntimeError("extract_features returned dict but no tensor value was found.")
    elif isinstance(out, (tuple, list)):
        return out[0]
    elif isinstance(out, torch.Tensor):
        return out
    else:
        raise RuntimeError(f"Unsupported output type from extract_features: {type(out)}")

def main():
    # 1) Load model on CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(CKPT).to(device).eval()

    # 2) Load debug batch: (B, 2, 12, 2500)  -> make BSxCT
    x = np.load(DEBUG_X).astype(np.float32)
    B, S, C, T = x.shape
    assert (S, C, T) == (2, 12, 2500), f"Unexpected x shape: {x.shape}"
    x = torch.from_numpy(x).to(device)

    # canonical layout for conv1d: (batch, channels=12, time=2500)
    inp = x.reshape(B * S, C, T)        # BSxCT
    pm = torch.zeros((B * S, T), dtype=torch.bool, device=device)  # no padding

    # 3) Forward with proper keywords + no masking
    with torch.no_grad():
        out = model.extract_features(source=inp, padding_mask=pm, mask=False)
        feat = _extract_feat_from_output(out)  # (BS, T', D) or (BS, D, T')
        if feat.dim() == 3 and feat.shape[1] > feat.shape[2]:
            # (BS, D, T') -> (BS, T', D)
            feat = feat.transpose(1, 2)

    info = {
        "layout": "BSxCT",
        "feat_shape": tuple(feat.shape),  # (BS, T', D)
    }
    (OUT / "io_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] layout=BSxCT -> feature shape=", info["feat_shape"])
    print("[INFO] wrote", OUT / "io_choice.json")

if __name__ == "__main__":
    main()
