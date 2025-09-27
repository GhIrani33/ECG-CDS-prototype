# -*- coding: utf-8 -*-
"""
Phase-1 Analysis (No retraining)
- Reads TEST per-label CSVs at three operating points (F1-opt / PPV=0.50 / PPV=0.30)
- Reads label_def.csv and label_stats.csv
- Standardizes columns, reconstructs counts if missing
- Computes 95% Wilson CIs for PPV and Recall
- Computes macro/micro aggregates
- Computes a simple random baseline (same alert rate)
- Writes combined CSVs + summary JSON + a readable TXT

Usage (PowerShell, Windows):
python ".\analyze_phase1.py" `
  --dataset_dir   "D:\Project\ECG\ecg-cds-ed2ed\screening_dataset_ed2ed_k72" `
  --runs_dir      "D:\Project\ECG\ecg-cds-ed2ed\runs\ed2ed_k73_finetune" `
  --ops           "f1,ppv50,ppv30" `
  --min_recall    0.10

"""

import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# -----------------------------
# Utilities
# -----------------------------

def ensure_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")

def read_parquet_rows(p: Path) -> int:
    df = pd.read_parquet(p)
    return len(df)

def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float,float]:
    """95% Wilson CI for a binomial proportion."""
    if trials <= 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~ 95%
    phat = successes / trials
    denom = 1 + (z**2)/trials
    center = (phat + (z**2)/(2*trials)) / denom
    margin = (z * math.sqrt( (phat*(1-phat)/trials) + (z**2)/(4*trials**2) )) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def to_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def standardize_per_label(df_raw: pd.DataFrame,
                          label_def: Optional[pd.DataFrame],
                          label_stats: Optional[pd.DataFrame],
                          N_test: int,
                          op_name: str) -> pd.DataFrame:
    """
    Normalize columns to:
    ['label','desc','thr','n','n_pos','pred_pos','tp','fp','fn',
     'prevalence','alert_rate','ppv','recall','f1',
     'ppv_lo','ppv_hi','rec_lo','rec_hi',
     'baseline_ppv','baseline_recall','baseline_f1','op']
    Reconstruct counts if missing.
    """
    df = to_lower_cols(df_raw)

    # map label text/desc from label_def if available
    desc_map = None
    if label_def is not None:
        df_ld = to_lower_cols(label_def)
        lab_col = pick_first(df_ld, ['label','code','icd','icd_code'])
        desc_col = pick_first(df_ld, ['text','desc','description','name'])
        if lab_col:
            if desc_col:
                desc_map = dict(zip(df_ld[lab_col].astype(str), df_ld[desc_col].astype(str)))
            else:
                desc_map = dict(zip(df_ld[lab_col].astype(str), ['']*len(df_ld)))
    # prevalence from label_stats if available
    prev_map = None
    if label_stats is not None:
        df_ls = to_lower_cols(label_stats)
        lab_col2 = pick_first(df_ls, ['label','code','icd','icd_code'])
        prev_col = pick_first(df_ls, ['train_prev','prev','prevalence'])
        if lab_col2 and prev_col:
            prev_map = dict(zip(df_ls[lab_col2].astype(str), df_ls[prev_col].astype(float)))

    # detect columns present
    label_col = pick_first(df, ['label','code','icd','icd_code','name'])
    if not label_col:
        raise KeyError(f"{op_name}: cannot find label column")

    thr_col   = pick_first(df, ['thr','threshold','thresh'])
    ppv_col   = pick_first(df, ['ppv','precision'])
    rec_col   = pick_first(df, ['recall','tpr','sensitivity','sens'])
    f1_col    = pick_first(df, ['f1','f1_score'])

    n_col     = pick_first(df, ['n','n_total','total'])          # total test after filter
    npos_col  = pick_first(df, ['n_pos','npos','support','pos']) # true positives in GT
    pred_col  = pick_first(df, ['pred_pos','n_pred','alerts'])   # predicted positives
    tp_col    = pick_first(df, ['tp'])
    fp_col    = pick_first(df, ['fp'])
    fn_col    = pick_first(df, ['fn'])

    prev_col  = pick_first(df, ['prevalence','prev'])
    ar_col    = pick_first(df, ['alert_rate','pred_rate'])

    # start build
    out = pd.DataFrame()
    out['label'] = df[label_col].astype(str)

    # description
    if desc_map:
        out['desc'] = out['label'].map(desc_map).fillna('')
    else:
        out['desc'] = ''

    # threshold
    out['thr'] = df[thr_col] if thr_col else np.nan

    # copy metrics if present
    out['ppv']   = df[ppv_col] if ppv_col else np.nan
    out['recall']= df[rec_col] if rec_col else np.nan
    out['f1']    = df[f1_col] if f1_col else np.nan

    # counts if present
    out['n']       = df[n_col]   if n_col   else N_test
    out['n_pos']   = df[npos_col] if npos_col else np.nan
    out['pred_pos']= df[pred_col] if pred_col else np.nan
    out['tp']      = df[tp_col]   if tp_col   else np.nan
    out['fp']      = df[fp_col]   if fp_col   else np.nan
    out['fn']      = df[fn_col]   if fn_col   else np.nan

    # prevalence / alert_rate
    out['prevalence'] = df[prev_col] if prev_col else np.nan
    out['alert_rate'] = df[ar_col]   if ar_col   else np.nan

    # backfill prevalence from label_stats if needed
    if out['prevalence'].isna().any() and prev_map is not None:
        out['prevalence'] = out['prevalence'].fillna(out['label'].map(prev_map))

    # reconstruct missing pieces
    # Step 1: n_pos from prevalence & N
    mask_npos_missing = out['n_pos'].isna()
    if mask_npos_missing.any():
        out.loc[mask_npos_missing, 'n_pos'] = (out.loc[mask_npos_missing, 'prevalence'] * out.loc[mask_npos_missing, 'n']).round()

    # Step 2: TP from recall * n_pos
    mask_tp_missing = out['tp'].isna()
    if mask_tp_missing.any():
        out.loc[mask_tp_missing, 'tp'] = (out.loc[mask_tp_missing, 'recall'] * out.loc[mask_tp_missing, 'n_pos']).round()

    # Step 3: pred_pos from TP / PPV
    mask_pred_missing = out['pred_pos'].isna()
    if mask_pred_missing.any():
        out.loc[mask_pred_missing, 'pred_pos'] = (out.loc[mask_pred_missing, 'tp'] / out.loc[mask_pred_missing, 'ppv']).replace([np.inf, -np.inf], np.nan).round()

    # Step 4: FP, FN
    mask_fp_missing = out['fp'].isna()
    if mask_fp_missing.any():
        out.loc[mask_fp_missing, 'fp'] = (out['pred_pos'] - out['tp']).round()
    mask_fn_missing = out['fn'].isna()
    if mask_fn_missing.any():
        out.loc[mask_fn_missing, 'fn'] = (out['n_pos'] - out['tp']).round()

    # Step 5: derive alert_rate if missing
    if out['alert_rate'].isna().any():
        out['alert_rate'] = out['pred_pos'] / out['n']

    # Recompute ppv/recall/f1 from counts to be consistent
    with np.errstate(divide='ignore', invalid='ignore'):
        ppv_new = out['tp'] / (out['tp'] + out['fp'])
        rec_new = out['tp'] / (out['n_pos'])
        f1_new  = 2 * (ppv_new * rec_new) / (ppv_new + rec_new)
    out['ppv']   = ppv_new.replace([np.inf, -np.inf], np.nan)
    out['recall']= rec_new.replace([np.inf, -np.inf], np.nan)
    out['f1']    = f1_new.replace([np.inf, -np.inf], np.nan)

    # Wilson CI for PPV (tp / (tp+fp)) and Recall (tp / n_pos)
    ppv_lo, ppv_hi, rec_lo, rec_hi = [], [], [], []
    for _, r in out.iterrows():
        tp = int(max(0, round(r['tp'] if pd.notna(r['tp']) else 0)))
        pred = int(max(0, round((r['tp'] + r['fp']) if (pd.notna(r['tp']) and pd.notna(r['fp'])) else r['pred_pos'])))
        npos = int(max(0, round(r['n_pos'] if pd.notna(r['n_pos']) else 0)))
        lo1, hi1 = wilson_ci(tp, pred) if pred > 0 else (np.nan, np.nan)
        lo2, hi2 = wilson_ci(tp, npos) if npos > 0 else (np.nan, np.nan)
        ppv_lo.append(lo1); ppv_hi.append(hi1); rec_lo.append(lo2); rec_hi.append(hi2)
    out['ppv_lo'] = ppv_lo; out['ppv_hi'] = ppv_hi
    out['rec_lo'] = rec_lo; out['rec_hi'] = rec_hi

    # Baseline (random with same alert rate): PPV ≈ prevalence ; Recall ≈ alert_rate
    base_ppv = out['prevalence'].astype(float)
    base_rec = out['alert_rate'].astype(float)
    base_f1 = (2 * base_ppv * base_rec) / (base_ppv + base_rec)
    out['baseline_ppv'] = base_ppv
    out['baseline_recall'] = base_rec
    out['baseline_f1'] = base_f1

    out['op'] = op_name
    # order columns
    cols = ['label','desc','thr','n','n_pos','pred_pos','tp','fp','fn',
            'prevalence','alert_rate','ppv','ppv_lo','ppv_hi',
            'recall','rec_lo','rec_hi','f1',
            'baseline_ppv','baseline_recall','baseline_f1','op']
    out = out[cols]
    # enforce numeric
    num_cols = [c for c in cols if c not in ['label','desc','op']]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors='coerce')

    return out

def macro_micro(df_std: pd.DataFrame) -> Dict[str, float]:
    # macro
    macro_ppv = df_std['ppv'].mean(skipna=True)
    macro_rec = df_std['recall'].mean(skipna=True)
    macro_f1  = df_std['f1'].mean(skipna=True)
    # micro (sum counts)
    TP = df_std['tp'].sum(skipna=True)
    FP = df_std['fp'].sum(skipna=True)
    FN = df_std['fn'].sum(skipna=True)
    micro_ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    micro_rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    micro_f1  = (2 * micro_ppv * micro_rec) / (micro_ppv + micro_rec) if (micro_ppv + micro_rec) > 0 else np.nan
    return {
        'macro': {'PPV': float(macro_ppv), 'Recall': float(macro_rec), 'F1': float(macro_f1)},
        'micro': {'PPV': float(micro_ppv), 'Recall': float(micro_rec), 'F1': float(micro_f1)},
        'TP_sum': int(TP), 'FP_sum': int(FP), 'FN_sum': int(FN)
    }

def top_k_tables(df_std: pd.DataFrame, min_recall: float, k: int = 10) -> Dict[str, list]:
    # Top by F1
    top_f1 = (df_std[['label','desc','ppv','recall','f1']]
              .sort_values('f1', ascending=False)
              .head(k)
              .round(6)
              .to_dict(orient='records'))
    # Top by PPV subject to recall >= min_recall
    df_ppv = df_std[df_std['recall'] >= min_recall]
    top_ppv = (df_ppv[['label','desc','ppv','recall','f1']]
              .sort_values('ppv', ascending=False)
              .head(k)
              .round(6)
              .to_dict(orient='records'))
    return {'top_f1': top_f1, 'top_ppv_with_min_recall': top_ppv}

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True, type=str)
    ap.add_argument('--runs_dir',    required=True, type=str)
    ap.add_argument('--ops',         type=str, default="f1,ppv50,ppv30",
                    help="Comma-separated list among: f1,ppv50,ppv30")
    ap.add_argument('--min_recall',  type=float, default=0.10,
                    help="Min recall for PPV-based top list")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    runs_dir    = Path(args.runs_dir)
    calib_dir   = runs_dir / "calib"
    out_dir     = calib_dir / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Determine N_test from manifest
    manifest_test = dataset_dir / "manifest_test.parquet"
    ensure_exists(manifest_test, "manifest_test.parquet")
    N_test = read_parquet_rows(manifest_test)

    # 2) Read label_def / label_stats if available
    label_def_path   = dataset_dir / "label_def.csv"
    label_stats_path = dataset_dir / "label_stats.csv"
    label_def = pd.read_csv(label_def_path) if label_def_path.exists() else None
    label_stats = pd.read_csv(label_stats_path) if label_stats_path.exists() else None

    # 3) Read per-op CSVs
    op_map = {
        'f1':    calib_dir / "test_per_label_f1.csv",
        'ppv50': calib_dir / "test_per_label_ppv50.csv",
        'ppv30': calib_dir / "test_per_label_ppv30.csv",
    }
    ops = [o.strip().lower() for o in args.ops.split(',') if o.strip()]
    selected = []
    for o in ops:
        if o not in op_map:
            print(f"[WARN] Unknown op '{o}' (allowed: f1,ppv50,ppv30). Skipped.", file=sys.stderr)
            continue
        ensure_exists(op_map[o], f"per-label CSV for {o}")
        selected.append(o)
    if not selected:
        raise RuntimeError("No valid operating points found to analyze.")

    # 4) Process each op
    combined_info = {}
    for o in selected:
        df_raw = pd.read_csv(op_map[o])
        df_std = standardize_per_label(df_raw, label_def, label_stats, N_test, op_name=o)

        # aggregates
        agg = macro_micro(df_std)
        # top lists
        tops = top_k_tables(df_std, min_recall=args.min_recall, k=10)

        # write combined CSV
        out_csv = out_dir / f"combined_{o}.csv"
        df_std.to_csv(out_csv, index=False)
        combined_info[o] = {
            'csv': str(out_csv),
            'macro': agg['macro'],
            'micro': agg['micro'],
            'TP_sum': agg['TP_sum'],
            'FP_sum': agg['FP_sum'],
            'FN_sum': agg['FN_sum'],
            'top': tops
        }
        print(f"[OK] wrote {out_csv}")

    # 5) Macro/micro summary (if macro_micro.json exists, we include it)
    mm_json = calib_dir / "test_macro_micro.json"
    mm = None
    if mm_json.exists():
        try:
            mm = json.loads(Path(mm_json).read_text(encoding='utf-8'))
        except Exception:
            mm = None

    # 6) Final summary JSON
    summary = {
        'N_test': N_test,
        'ops_analyzed': selected,
        'per_op': combined_info,
        'macro_micro_original': mm
    }
    out_json = out_dir / "phase1_summary.json"
    Path(out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[OK] wrote {out_json}")

    # 7) Readable TXT
    lines = []
    lines.append("=== Phase-1 Analysis (No Retraining) ===")
    lines.append(f"N_test: {N_test}")
    for o in selected:
        info = combined_info[o]
        lines.append(f"\n[Operating Point: {o}]")
        lines.append(f"CSV: {info['csv']}")
        lines.append(f"Macro: PPV={info['macro']['PPV']:.4f}  Recall={info['macro']['Recall']:.4f}  F1={info['macro']['F1']:.4f}")
        lines.append(f"Micro: PPV={info['micro']['PPV']:.4f}  Recall={info['micro']['Recall']:.4f}  F1={info['micro']['F1']:.4f}")
        lines.append("Top-10 by F1:")
        for r in info['top']['top_f1']:
            lines.append(f"  {r['label']:<6}  F1={r['f1']:.4f}  PPV={r['ppv']:.4f}  R={r['recall']:.4f}  {r.get('desc','')}")
        lines.append(f"Top-10 by PPV (Recall≥{args.min_recall:.2f}):")
        for r in info['top']['top_ppv_with_min_recall']:
            lines.append(f"  {r['label']:<6}  PPV={r['ppv']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  {r.get('desc','')}")
    out_txt = out_dir / "phase1_readme.txt"
    Path(out_txt).write_text("\n".join(lines), encoding='utf-8')
    print(f"[OK] wrote {out_txt}")
    print("[DONE]")

if __name__ == '__main__':
    main()
