# Resolution-aware EDL experiment package

This folder contains the next stage after the rewritten paper: runnable experiment code for the EDL-based CIFAR-10/SVHN protocol, plus a synthetic demo that works offline and reproduces the paper's intended diagnostic pattern.

## Files

- `train_edl_cifar10.py` trains an evidential ResNet-18 on CIFAR-10.
- `evaluate_resolution_diagnostics.py` runs the paper's main evaluation on CIFAR-10 vs SVHN.
- `synthetic_resolution_demo.py` generates an offline toy benchmark with entropy matching, AUROC tables, and risk-coverage curves.
- `models.py` implements a self-contained CIFAR-style ResNet-18.
- `utils_evidential.py` contains the evidential loss, TFU diagnostics, matching, and evaluation helpers.

## Real experiment workflow

```bash
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_edl \
  --epochs 100

python evaluate_resolution_diagnostics.py \
  --data-dir ./data \
  --checkpoint ./runs/cifar10_edl/best.pt \
  --out-dir ./runs/cifar10_eval \
  --betas 0.1 0.5 0.75
```

## Recommended training comparisons

If the pure EDL run looks weak, separate classifier quality from the EDL objective:

```bash
# 1. Plain cross-entropy baseline.
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_ce \
  --loss ce \
  --epochs 100

# 2. EDL finetune from the CE checkpoint.
python train_edl_cifar10.py \
  --data-dir ./data \
  --out-dir ./runs/cifar10_edl_from_ce \
  --loss edl \
  --init-checkpoint ./runs/cifar10_ce/best.pt \
  --epochs 30
```

For EDL debugging, `--no-train-augment` is useful. In this codebase, EDL is much more sensitive to the default random crop/flip augmentation than CE.

Outputs include:

- `matched_scatter.png` for the main figure
- `matched_auroc.csv` for the entropy-matched table
- `risk_coverage_summary.csv` and `risk_coverage_eta_*.png`
- `beta_scan_ablation.csv`

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
4. The real CIFAR-10/SVHN script assumes a healthy `torch` / `torchvision` installation. Because Python packaging is a small industrial accident, the synthetic script is included so you can validate the logic even when your local torchvision build is grumpy.
