# ECG-CDS-ED2ED: Clinical ICD-10 Prediction from 12-lead ECG (ED setting)


**Status:** Research/Feasibility • Windows/PowerShell–first, fully reproducible
**Encoder:** [ECG-FM](https://github.com/bowang-lab/ECG-FM?tab=readme-ov-file) (pretrained on PhysioNet Challenge ECG)
**Datasets:**

* MIMIC-IV-ECG (public): [https://physionet.org/content/mimic-iv-ecg/1.0/](https://physionet.org/content/mimic-iv-ecg/1.0/)
* MIMIC-IV-ECG-EXT-ICD labels (credentialed): [https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)

This repository provides a **reproducible pipeline** to evaluate whether ECG alone can screen for **ED discharge ICD-10 diagnoses** (ED→ED, abbreviated **ED2ED**). We re-use a **pretrained ECG-FM** encoder (originally fine-tuned for ECG diagnosis) and conduct **head-only fine-tuning** for a **73-label** ICD-10 set (5-char truncation, min freq ≥ 1000). We further **calibrate per-label thresholds** on validation (F1-optimal, PPV≈0.50, PPV≈0.30) and report **test** performance with macro/micro and per-label metrics.

> **Important:** ECG-FM’s original fine-tuning targets **ECG rhythm/morphology**. Here we evaluate a **different task**—predicting **clinical diagnoses** assigned at ED discharge—purely from ECG. This is an **exploratory CDS feasibility** study, **not** a clinical tool.

---

## TL;DR (What to know in 30 seconds)

* **Task:** Predict **ED discharge ICD-10 codes** from a single 12-lead ECG (ED2ED label space).
* **Modeling:** **Frozen ECG-FM encoder** + lightweight MLP head, **ASL (focal-style)** loss, **attention pooling**.
* **QC:** Filtered ~1% records with non-finite samples; 12-lead @ 500 Hz confirmed.
* **Calibration:** Per-label thresholds from **validation**; report **test** metrics at three operating points.
* **Headline finding:** For **Atrial Fibrillation (I48.91)** the model achieves **PPV ≈ 0.49, Recall ≈ 0.45, F1 ≈ 0.47** (F1-optimal), making it a promising **ED screening** signal.
* **Global (73-label) performance is low** (e.g., macro AUPRC ≈ 0.06), consistent with the difficulty of predicting broad **clinical diagnoses** from ECG alone.
* **Positioning:** This is a **focused, evidence-based feasibility**—*AFib* is promising; many other ICD-10 labels are not.

---

## What’s in this repository

* **Dataset construction (ED2ED)**: `build_screening_dataset_ed2ed_k27.py` (your working version may be `build_screening_dataset.py` in the repo; keep the one you actually used). Produces:

  * `manifest_train/val/test.parquet` with `hea_path`, `dat_path`, `labels_list`, `y_multi_hot`
  * `label_def.csv`, `label_map.json`, `label_stats.csv`, `dataset_stats.json`, `pos_weight.npy`
* **I/O verification for ECG-FM**: `probe_ecgfm_io.py` (discovers correct input layout; confirms feature dim = 768).
* **Fine-tuning**: `finetune_ed2ed_ecgfm.py` (frozen encoder, ASL loss, attention pooling, AMP, grad-clip).
* **Calibration on VAL**: `calibrate_thresholds_val.py` (writes `calib/thresholds.json` and validation summaries).
* **Evaluation on TEST**: `apply_thresholds_test.py` (per-label PPV/Recall/F1 at F1-opt, PPV≈0.50, PPV≈0.30).
* **Phase-1 analysis**: `analyze_phase1.py` (aggregates VAL/TEST summaries, writes CSV/JSON).

All scripts are **Windows PowerShell** friendly and tested on:

* **Windows 10 Pro**, Python 3.9, **PyTorch 2.1.2 + CUDA 11.8**, **fairseq-signals** (editable install)
* GPU: **RTX 3060 Ti (8GB VRAM)**; System RAM ≈ 12–16 GB

---

## Data & Access

* **ECG waveforms (public):** [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/)
* **ED discharge ICD-10 labels (credentialed):** [mimic-iv-ecg-ext-icd-labels 1.0.1](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)

  * We restrict to **ED2ED**: both ECG acquisition and ICD assignment occur within the ED visit.
  * Labels are truncated to **5 characters**; frequency threshold **≥ 1000** yields **K = 73** labels.

**Splits:** folds by `study_id` with ED temporal de-duplication on VAL/TEST (first ECG per stay).
**QC:** ~**1%** of records excluded (non-finite samples after WFDB read + z-score check).

---

## Model & Training

* **Encoder:** [ECG-FM](https://github.com/bowang-lab/ECG-FM?tab=readme-ov-file) (pretrained; loaded via `fairseq_signals.models.build_model_from_checkpoint`).

  * Feature dimension **D=768** (confirmed).
* **Head:** MLP with attention pooling: `AttnPool(12→D) → [1024→512] → K` (73).
* **Loss:** **ASL** (focal-style) with `γ_pos=1.0, γ_neg=4.0`, small logit clip=0.05; **pos_weight** from TRAIN prevalence.
* **Optimization:** Adam (head LR **1e-3**), AMP on, grad-clip=1.0, cosine LR decay optional.
* **Batching:** 48–64 on 8GB VRAM; input layout **B×S×C×T = (B,2,12,2500)** (two 5-s crops @ 500 Hz).
* **Why frozen encoder:** empirical stability + speed on Windows GPU; **partial unfreeze** of last two transformer blocks was tested but did **not** outperform the best head-only run in this setting.

---

## Calibration & Evaluation

* **Calibration (VAL):** compute **per-label thresholds** at:

  * **F1-optimal** (max F1 on VAL)
  * **PPV≈0.50** target (maximize recall under PPV ≥ 0.5 if attainable)
  * **PPV≈0.30** target
  * Save as `runs/<exp>/calib/thresholds.json`
* **Evaluation (TEST):** apply thresholds to get **per-label** PPV/Recall/F1 and **macro/micro** aggregates.

  * We also provide combined summaries for **Phase-1** (val+test CSVs and a JSON summary).

---

## Key Results (TEST)

**Global 73-label performance** is modest—consistent with the difficulty of mapping a single ECG to broad ED diagnoses:

* **At F1-optimal thresholds:** macro {PPV≈**0.066**, Recall≈**0.251**, F1≈**0.095**}; micro {PPV≈**0.073**, Recall≈**0.362**, F1≈**0.122**}.
* **At PPV≈0.50 targets:** macro {PPV≈**0.076**, Recall≈**0.236**, F1≈**0.085**}.
* **At PPV≈0.30 targets:** macro {PPV≈**0.080**, Recall≈**0.220**, F1≈**0.077**}.

> Note: macro PPV does **not** equal the target—many labels cannot reach PPV=0.5 under practical thresholds; the macro average reflects that.

**Per-label highlight (proof-of-concept):**

* **I48.91 (Atrial Fibrillation)**

  * **F1-opt:** **PPV ≈ 0.49, Recall ≈ 0.45, F1 ≈ 0.47**
  * **High-precision mode (PPV≈0.80+):** recall drops substantially, as expected.
  * Interpretation: AFib has **direct ECG manifestation** (rhythm irregularity), hence learnable from ECG alone.

Other clinically relevant labels (e.g., **I21.3 STEMI**, **R00.1 Bradycardia**, **I10 Hypertension**) show weaker or mixed results—aligned with the reality that **many ED discharge diagnoses are not ECG-centric** and require clinical context.

---

## Intended Use & Scope

* **Intended use (research):** Feasibility of **ECG-only** signals for **ED CDS** under strict constraint of **low alarm fatigue** (prefer higher PPV).
* **Not intended** for real-world clinical decision-making. No guarantees of safety/effectiveness.
* **Most promising use case:** **AFib screening** in ED triage as a **focused tool**, not as an all-ICD predictor.

---

## Reproducibility (Windows/PowerShell quickstart)

1. **Install core dependencies**

* Python 3.9, PyTorch 2.1.2+cu118
* `git clone https://github.com/Jwoo5/fairseq-signals` → `pip install -e .`

2. **Prepare data**

* Download **MIMIC-IV-ECG** (public) and **mimic-iv-ecg-ext-icd-labels** (credentialed).
* Update paths in dataset builder and run:

  ```
  python build_screening_dataset.py `
    --mimic_ecg_root "D:\path\to\mimic_iv_ecg" `
    --icd_ext_root  "D:\path\to\mimic_iv_ecg_ext_icd" `
    --out_dir       "D:\Project\ECG\new\screening_dataset_ed2ed_k72" `
    --min_freq 1000 --truncate_len 5 --split ED2ED
  ```

3. **Probe encoder I/O & feature dim**

```
python probe_ecgfm_io.py --ckpt_pre "...\mimic_iv_ecg_physionet_pretrained.pt"
```

4. **Fine-tune (head-only, ASL)**

```
python finetune_ed2ed_ecgfm.py `
  --dataset_dir "...\screening_dataset_ed2ed_k72" `
  --runs_dir    "...\runs\ed2ed_k73_finetune" `
  --ckpt_pre    "...\mimic_iv_ecg_physionet_pretrained.pt" `
  --epochs 20 --batch_size 48 --lr_head 1e-3 --use_asl 1 --device cuda
```

5. **Calibrate thresholds on VAL**

```
python calibrate_thresholds_val.py `
  --dataset_dir "...\screening_dataset_ed2ed_k72" `
  --runs_dir    "...\runs\ed2ed_k73_finetune" `
  --ckpt_pre    "...\mimic_iv_ecg_physionet_pretrained.pt" `
  --head_path   "...\runs\ed2ed_k73_finetune\best_head_state_dict_fc.pt" `
  --batch_size 48 --device cuda --thr_mode f1 --ppv_targets 0.5 0.3 --rec_min 0.05
```

6. **Evaluate on TEST**

```
python apply_thresholds_test.py `
  --dataset_dir "...\screening_dataset_ed2ed_k72" `
  --runs_dir    "...\runs\ed2ed_k73_finetune" `
  --ckpt_pre    "...\mimic_iv_ecg_physionet_pretrained.pt" `
  --head_path   "...\runs\ed2ed_k73_finetune\best_head_state_dict_fc.pt" `
  --batch_size 48 --device cuda
```

Artifacts land under `runs/.../calib/` (per-label CSVs, macro/micro JSON, thresholds).

---

## Limitations & Ethical Considerations

* **ECG-only** features are **insufficient** for many ED discharge diagnoses; expect **low PPV** globally.
* Label noise and clinical coding practices may decouple ICD assignment from contemporaneous ECG findings.
* Model trained and evaluated on a **single health system cohort** (MIMIC); **external validity** not established.
* This work is **research-only**; **not** a medical device; **do not** use to make clinical decisions.

---

## How this differs from the original ECG-FM paper

* ECG-FM’s reported fine-tuning targets **ECG diagnostic tasks** (rhythm/morphology)—a closer match to the pretraining domain.
* Our work targets **clinical discharge ICD-10** (ED2ED), which often encode **non-ECG factors**; thus lower performance is expected and observed.
* The value proposition emerges for **ECG-centric labels** (e.g., **AFib**) rather than broad ICD-10 coverage.

---

## Repository Structure (suggested)

```
ecg-cds-ed2ed/
  README.md
  doc/
    MODEL_CARD.md
  scripts/
    build_screening_dataset.py
    probe_ecgfm_io.py
    finetune_ed2ed_ecgfm.py
    calibrate_thresholds_val.py
    apply_thresholds_test.py
    analyze_phase1.py
  runs/                # ignored by .gitignore
  data/                # ignored
  examples/            # tiny sample outputs (optional, committed)
  LICENSE
```

---

## License

MIT License (see `LICENSE`).
Datasets are provided by PhysioNet under their respective terms. Access to the **ICD label set** requires PhysioNet **credentialed access**.

---

## Acknowledgements

* **ECG-FM**: [https://github.com/bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM)
* PhysioNet MIMIC-IV-ECG and extended ICD label set maintainers
* This repo is Windows-first but scripts are standard Python and can be adapted to Linux.

---





