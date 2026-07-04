# AM-DiMNet

**HRV-Anchored Autonomic–Morphological Structured Factorization for Interpretable
Multimodal Cardiovascular Risk Prediction**

Reference PyTorch implementation.
Repository: https://github.com/am-dimnet/cardio-deterioration

---

## Overview

AM-DiMNet predicts **near-term cardiovascular deterioration** in the ICU from
routinely monitored waveforms and derived features. Given a short observation
window (5 min), it estimates the risk that a composite deterioration endpoint
(cardiac arrest / CPR, death, vasoactive escalation, acute heart failure,
malignant arrhythmia, or urgent cardiovascular intervention — Table 1) occurs
within a prediction horizon `[window_end + gap, window_end + gap + horizon]`
(default gap 1.5 h, horizon 24 h; Eq. 2–3).

The model is built for **interpretability and robustness** rather than raw
capacity:

- **Quality-aware gated fusion.** Per-modality encoders (ECG, PPG, optional PCG,
  plus proxy / HRV / clinical vectors) are combined by a gate that weights each
  modality by its signal-quality index (SQI) and availability mask, so degraded
  or missing channels are down-weighted rather than trusted blindly (Eq. 14–15).
- **Structured latent factorization.** The fused embedding is factorized into
  five interpretable factors — *autonomic*, *morphological*, *activity*,
  *baseline*, *noise* — and only the autonomic, morphological, and baseline
  factors feed the risk head (Eq. 16–17).
- **HRV anchoring.** The autonomic factor must reconstruct a 10-dimensional HRV
  feature vector, tethering it to physiology (Eq. 18–19).
- **Activity de-confounding.** A motion/activity proxy (derived without a
  dedicated accelerometer) anchors an activity factor and, via **gradient
  reversal**, pushes the autonomic and morphological factors to be
  activity-invariant, so motion artifact does not masquerade as risk (Eq. 22–23).
- **Orthogonality + calibration.** A cross-factor orthogonality penalty keeps the
  factors disentangled (Eq. 24–25), and a Brier term plus post-hoc temperature
  scaling keep the predicted probabilities well-calibrated (Eq. 27, 35).

The repository is **fully runnable without any patient data**: a self-contained
`SyntheticCohort` reproduces the batch contract, and a CPU smoke test exercises
the entire method end to end.

> **On reported numbers.** This code does not embed the manuscript's performance
> figures. On synthetic data it produces qualitatively sensible behavior
> (finite/decreasing loss, above-chance discrimination, calibratable
> probabilities). Reproducing the paper's numbers requires the credentialed
> MIMIC-IV data (see below).

---

## Install

Python 3.10+ and PyTorch 2.x. No GPU needed for the smoke test.

```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate am-dimnet
```

---

## Quickstart

```bash
python -m am_dimnet.smoke_test
```

This builds `Config.default()`, creates a tiny patient-disjoint synthetic
cohort, runs an `AMDiMNet` forward pass and asserts all output shapes, computes
`AMDiMNetLoss` and takes a couple of optimizer steps (loss finite and
decreasing), then runs temperature calibration plus the full metric suite
(AUROC / AUPRC / ECE-15 / operating point) on the tiny test set. On success it
prints `SMOKE TEST PASSED`. Runs on CPU in under ~1 minute.

### Tests

A minimal, fast, CPU-only `pytest` suite (`tests/`) checks the config/constants
against the manuscript, the batch/data contract, the model forward + loss
scaling, metric correctness (event-frequency ECE, AUROC, operating point,
patient aggregation, temperature calibration), the endpoint item IDs (Table 1)
and labelling rule, and every baseline's forward pass. It also runs the smoke
test as a case. No external data required.

```bash
pip install pytest
pytest            # ~40 tests, ~12 s on CPU
```

---

## Full MIMIC-IV reproduction pipeline

Complete PhysioNet credentialing first (`docs/data_access.md`); the datasets are
not redistributed here. Then run the stages in order — each script is a thin
wrapper (`set -euo pipefail`) that reads paths/DB settings from
`configs/mimic.yaml`. Full step-by-step details are in `docs/reproducibility.md`.

```
extract  ->  preprocess  ->  split  ->  train  ->  calibrate  ->  evaluate
```

```bash
# 1. Extract candidate ICU stays + waveform linkage; apply Table-12 cohort flow;
#    build composite endpoint labels (Table 1, Eq. 2-3).
scripts/run_extract.sh    configs/mimic.yaml

# 2. Bandpass + window (5 min / 60 s step) + features (HRV Eq. 5, proxy Eq. 7-9,
#    SQI Eq. 10) + build batch-contract records, then patient-disjoint splits.
scripts/run_preprocess.sh configs/mimic.yaml

# 3. Train across the 5 seeds (AdamW, early stop on val AUROC, modality dropout,
#    GRL warmup). One checkpoint per seed.
scripts/run_train.sh      configs/mimic.yaml

# 4. 
