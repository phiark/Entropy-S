# Resolution-aware EDL experiment package

This folder contains the next stage after the rewritten paper: runnable CIFAR-10 / SVHN diagnostics code, plus a synthetic demo that reproduces the intended pattern offline.

Current status:

- `CE baseline` is stable and reaches normal CIFAR-10 accuracy.
- `EDL from CE finetune` is usable for diagnostics.
- `pure EDL from scratch` is currently not a reliable mainline recipe in this repo and should be treated as a debugging path, not the default experiment.

## Documentation map

- Project linkage across the paper, audit report, code, and current results: [REPORT_PROJECT_LINKAGE.md](/Users/zero_lab/Documents/resolution_edl_experiment/REPORT_PROJECT_LINKAGE.md)
- Next experiment design based on the structural audit: [EXPERIMENT_V2_PLAN.md](/Users/zero_lab/Documents/resolution_edl_experiment/EXPERIMENT_V2_PLAN.md)
- Repair log for implementation-level fixes: [REPAIR_REPORT.md](/Users/zero_lab/Documents/resolution_edl_experiment/REPAIR_REPORT.md)
- Original manuscript draft: [report/V3.tex](/Users/zero_lab/Documents/resolution_edl_experiment/report/V3.tex)
- Structural audit report: [report/entropy_s_structural_evaluation_report.tex](/Users/zero_lab/Documents/resolution_edl_experiment/report/entropy_s_structural_evaluation_report.tex)

## Files

- `train_edl_cifar10.py` trains an evidential ResNet-18 on CIFAR-10.
- `evaluate_resolution_diagnostics.py` runs the paper's main evaluation on CIFAR-10 vs SVHN.
- `synthetic_resolution_demo.py` generates an offline toy benchmark with entropy matching, AUROC tables, and risk-coverage curves.
- `models.py` implements a self-contained CIFAR-style ResNet-18.
- `utils_evidential.py` contains the evidential loss, TFU diagnostics, matching, and evaluation helpers.

## Recommended workflow

```bash
# 1. Train a strong classifier first.
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_ce \
  --loss ce \
  --epochs 100

# 2. Finetune with EDL from the CE checkpoint.
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_edl_from_ce \
  --loss edl \
  --init-checkpoint ./runs/cifar10_ce/best.pt \
  --no-train-augment \
  --epochs 30

# 3. Define a fixed cohort with a reference checkpoint.
python evaluate_resolution_diagnostics.py \
  --data-dir ./data \
  --checkpoint ./runs/cifar10_ce/best.pt \
  --out-dir ./runs/cifar10_eval_ce_reference \
  --betas 0.1 0.5 0.75

# 4. Reuse the same cohort for later checkpoint comparisons.
python evaluate_resolution_diagnostics.py \
  --data-dir ./data \
  --checkpoint ./runs/cifar10_edl_from_ce/best.pt \
  --out-dir ./runs/cifar10_eval_from_ce \
  --cohort-manifest ./runs/cifar10_eval_ce_reference/cohort_manifest.json \
  --betas 0.1 0.5 0.75
```

## Pure EDL debugging path

If you want to inspect the unstable path directly:

```bash
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_edl \
  --loss edl \
  --epochs 100
```

For EDL debugging, `--no-train-augment` is useful. In this codebase, EDL is much more sensitive to the default random crop/flip augmentation than CE.

## Current findings

- `pure EDL from scratch` reached only about `49.6%` validation accuracy and produced unreliable downstream diagnostics.
- `CE baseline` reached about `93.1%` validation accuracy.
- `EDL from CE finetune` reached about `91.0%` validation accuracy.
- Once the classifier is strong, the diagnostics recover to a reasonable range, but the current results do not yet show a strong, stable "EDL clearly beats CE" conclusion.
- The repository's detailed repair log is in [REPAIR_REPORT.md](/Users/zero_lab/Documents/resolution_edl_experiment/REPAIR_REPORT.md).

Outputs include:

- `cohort_manifest.json` for fixed benchmark reuse
- `matched_discrimination/matched_scatter*.png`
- `matched_discrimination/matched_auroc.csv`
- `selective_rejection/risk_coverage_summary.csv`
- `selective_rejection/risk_coverage_eta_*.png`
- `selective_rejection/beta_scan_ablation.csv`

## Synthetic fallback

If you want a no-download, low-compute sanity check, run:

```bash
python synthetic_resolution_demo.py --out-dir ./synthetic_demo
```

That script simulates two EDL-like cohorts:

- **ID-hard:** high evidence, low vacuity, high conflict between top classes
- **OOD:** low evidence, high vacuity, diffuse beliefs

It then entropy-matches the cohorts by ordinary predictive entropy, so the classic scalar is deliberately neutralized. The expected pattern is the whole point of the paper: OOD clusters at low `r`, while ID-hard clusters at higher `r` but larger conditional conflict `H_cont`.

## Notes

1. `projected_entropy_beta_0p1` corresponds to the EDL-compatible choice `beta = 1 / K` for `K = 10` classes.
2. AUROCs in `matched_auroc.csv` are reported with the best monotone orientation because some diagnostics are naturally large on OOD (vacuity) while others are naturally large on ID-hard (dissonance).
3. Risk-coverage uses the natural uncertainty direction: larger score means more reason to abstain.
4. In the current implementation, `vacuity` and `1 - r` are the same quantity, so they should not be interpreted as independent evidence.
5. The real CIFAR-10/SVHN script assumes a healthy `torch` / `torchvision` installation. Because Python packaging is a small industrial accident, the synthetic script is included so you can validate the logic even when your local torchvision build is grumpy.
