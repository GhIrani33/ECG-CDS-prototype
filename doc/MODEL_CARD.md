
# MODEL CARD — ECG-CDS-ED2ED

**Model family:** ECG-FM encoder (pretrained) + lightweight MLP head
**Task:** Predict **ED discharge ICD-10** labels from a single 12-lead ECG (**ED→ED**, “ED2ED”)
**Status:** Research/feasibility (not a medical device)

---

## 1) Intended Use

* **Primary intent (research):** Assess feasibility of **ECG-only** signals for **ED clinical decision support (CDS)** across a curated **73-label** ICD-10 subset.
* **Clinical scope:** Screening/triage support where **low alarm fatigue** (higher PPV) is prioritized.
* **Most promising use case:** **Atrial Fibrillation (AFib, I48.91)** detection—an ECG-centric condition.

> **Not for clinical use.** This system has not undergone external validation, regulatory review, or prospective clinical assessment.

---

## 2) Out-of-Scope / Non-Goals

* Not a general ICD-10 coder. Most ED discharge diagnoses are **not** ECG-deterministic.
* No clinical metadata (e.g., vitals, age/sex) is used.
* No external validation; results are from MIMIC only.

---

## 3) Data

* **Waveforms:** MIMIC-IV-ECG v1.0 (public)
  [https://physionet.org/content/mimic-iv-ecg/1.0/](https://physionet.org/content/mimic-iv-ecg/1.0/)
* **ED discharge labels:** MIMIC-IV-ECG-EXT-ICD v1.0.1 (**credentialed access**)
  [https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)

### Label design (ED2ED)

* ECG acquired in ED and ICD-10 assigned at ED discharge (**ED→ED**).
* ICD-10 codes truncated to **5 characters**.
* Frequency threshold **≥ 1000** → **K = 73** labels.
* Multilabel setting (each ECG may map to multiple ICD-10 codes).

### Quality control

* Standard WFDB read; Z-score per lead.
* ~**1%** records excluded due to non-finite samples.

---

## 4) Model

* **Encoder:** ECG-FM (pretrained; loaded via `fairseq_signals.models.build_model_from_checkpoint`), feature dim **D = 768**.
* **Head:** Attention pooling (12-lead → D) → MLP **[1024 → 512 → K]** with dropout.
* **Loss:** **ASL** (focal-style) with γ_pos=1.0, γ_neg=4.0, small logit-clip=0.05; **pos_weight** from TRAIN prevalence.
* **Optimization:** Adam; head LR ≈ 1e-3; AMP; grad-clip=1.0.
* **Freezing:** Encoder **frozen** in the main run (partial unfreeze of late transformer blocks was tested and did not beat the best head-only run in this setting).

---

## 5) Training & Calibration

* **Splits:** TRAIN/VAL/TEST from the ED2ED cohort (by study_id; temporal de-duplication on VAL/TEST).
* **Batching:** 48–64 on 8 GB VRAM; input layout **B×S×C×T = (B,2,12,2500)** (two 5-s crops @ 500 Hz).
* **Calibration (per-label):** Thresholds tuned on **VAL** for three operating points:

  1. **F1-optimal**, 2) **PPV≈0.50**, 3) **PPV≈0.30**.
     Saved to `runs/.../calib/thresholds.json`.

---

## 6) Evaluation (TEST)

We report **per-label** PPV/Recall/F1 and **macro/micro** aggregates at each operating point. Artifacts are included under `runs/.../calib/` (CSV + JSON).

### Global summary (73 labels, ECG-only)

* **At F1-optimal thresholds (on VAL):**
  **Macro:** PPV ≈ **0.066**, Recall ≈ **0.251**, F1 ≈ **0.095**
  **Micro:** PPV ≈ **0.073**, Recall ≈ **0.362**, F1 ≈ **0.122**
* **At PPV≈0.50 target:** Macro PPV ≈ **0.076**, Recall ≈ **0.236**, F1 ≈ **0.085**
* **At PPV≈0.30 target:** Macro PPV ≈ **0.080**, Recall ≈ **0.220**, F1 ≈ **0.077**

> Interpretation: Broad clinical ICD-10 prediction from ECG alone yields **low global performance**, consistent with label non-specificity to ECG.

### Per-label highlight

* **AFib (I48.91)** emerged as the **strongest** ECG-centric label in Phase-1 analysis (see per-label CSVs).
* Labels like **STEMI (I21.3)** and **Bradycardia (R00.1)** showed mixed results; further error analysis is warranted.

> **Confidence intervals:** not estimated in this release. (Bootstrap CIs can be added in a subsequent update.)

---

## 7) Fairness & Safety

* **Fairness/bias:** No demographic features were used; subgroup performance was **not** analyzed.
* **Safety:** This is a **research tool**. Do **not** use predictions to change patient care. Alarm policies must be defined with clinicians to prevent alert fatigue.
* **Data governance:** Users must obtain appropriate PhysioNet credentials for the ICD label set and comply with the data use agreements.

---

## 8) Limitations

* Many ED discharge ICD-10 codes reflect **clinical context** rather than ECG morphophysiology; ECG-only signals are insufficient.
* Single-center retrospective data (MIMIC) → limited external validity.
* No CI estimates; no external test set; no comparison to clinician performance in this release.

---

## 9) Hardware / Software

* **OS:** Windows 10 Pro; **GPU:** RTX 3060 Ti (8 GB); **RAM:** ~12–16 GB
* **PyTorch:** 2.1.2 + CUDA 11.8
* **fairseq-signals:** editable install

---

## 10) How to Use (in brief)

* Build the ED2ED dataset from MIMIC-IV-ECG + credentialed ICD labels (scripts provided).
* Fine-tune the **head** on TRAIN; calibrate thresholds on VAL; evaluate on TEST.
* Use **per-label thresholds** matched to your operating point (F1-opt, PPV≈0.50, PPV≈0.30).
* Inspect per-label CSVs; for CDS prototyping, focus on ECG-centric labels (e.g., **AFib**).

---

## 11) Ethical & Legal

* **License:** MIT for code (see `LICENSE`).
* **Data:** PhysioNet licenses/DUAs apply; credentialed access is required for the ICD label set.
* **Clinical use:** Prohibited without appropriate validation, approval, and governance.

---

## 12) Acknowledgements & Links

* **ECG-FM:** [https://github.com/bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM)
* **MIMIC-IV-ECG:** [https://physionet.org/content/mimic-iv-ecg/1.0/](https://physionet.org/content/mimic-iv-ecg/1.0/)
* **MIMIC-IV-ECG-EXT-ICD:** [https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/)

---

### Change Log (high-level)

* v0.1 (this release): Frozen ECG-FM + ASL head; ED2ED dataset; per-label thresholding; Phase-1 analysis; Windows/PowerShell recipes.
