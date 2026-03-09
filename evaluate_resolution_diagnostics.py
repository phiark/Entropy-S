from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import cifar_resnet18
from utils_evidential import (
    alpha_to_metrics,
    build_dataloader_kwargs,
    completion_score,
    entropy_matched_indices,
    format_beta,
    get_default_device,
    oriented_auroc,
    risk_coverage_curve,
    selective_risk_at_coverage,
    set_seed,
)


def build_eval_loaders(data_dir: str, batch_size: int, num_workers: int):
    try:
        from torchvision import datasets, transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for the real CIFAR-10/SVHN experiment. "
            "Install a matching torch/torchvision pair in your environment."
        ) from e

    cifar_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    svhn_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )

    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=cifar_transform)
    svhn_test = datasets.SVHN(root=data_dir, split="test", download=True, transform=svhn_transform)

    loader_kwargs = build_dataloader_kwargs(num_workers)
    cifar_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, **loader_kwargs)
    svhn_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return cifar_loader, svhn_loader


@torch.no_grad()
def infer_alpha(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_alpha: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        logits = model(images)
        evidence = torch.nn.functional.softplus(logits)
        alpha = evidence + 1.0
        all_alpha.append(alpha.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_alpha, axis=0), np.concatenate(all_labels, axis=0)


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_main_scatter(r: np.ndarray, h_cont: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.0))
    mask_id = labels == 0
    mask_ood = labels == 1
    plt.scatter(r[mask_id], h_cont[mask_id], s=12, alpha=0.55, label="CIFAR-10 ID-hard")
    plt.scatter(r[mask_ood], h_cont[mask_ood], s=12, alpha=0.55, label="SVHN OOD")
    plt.xlabel("resolution ratio r")
    plt.ylabel("content entropy H_cont")
    plt.title("Entropy-matched diagnostic scatter")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_risk_coverage(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.0))
    for name, (coverage, risk) in curves.items():
        plt.plot(coverage, risk, label=name)
    plt.xlabel("coverage")
    plt.ylabel("selective risk")
    plt.title("Risk-coverage curves")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resolution-aware diagnostics on CIFAR-10/SVHN.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./runs/cifar10_edl_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cohort-size", type=int, default=5000)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--coverage-target", type=float, default=0.8)
    parser.add_argument("--etas", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    parser.add_argument("--betas", type=float, nargs="*", default=[0.1, 0.5, 0.75])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = get_default_device()
    print(f"Using device: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = cifar_resnet18(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])

    cifar_loader, svhn_loader = build_eval_loaders(args.data_dir, args.batch_size, args.num_workers)
    cifar_alpha, cifar_labels = infer_alpha(model, cifar_loader, device)
    svhn_alpha, svhn_labels = infer_alpha(model, svhn_loader, device)

    cifar_metrics = alpha_to_metrics(cifar_alpha)
    svhn_metrics = alpha_to_metrics(svhn_alpha)

    id_hard_idx = np.argsort(cifar_metrics.margin)[: args.cohort_size]
    ood_idx = rng.choice(np.arange(len(svhn_alpha)), size=min(args.cohort_size, len(svhn_alpha)), replace=False)

    id_pred_entropy = cifar_metrics.pred_entropy[id_hard_idx]
    ood_pred_entropy = svhn_metrics.pred_entropy[ood_idx]
    combined_entropy = np.concatenate([id_pred_entropy, ood_pred_entropy])
    combined_labels = np.concatenate([np.zeros_like(id_pred_entropy, dtype=int), np.ones_like(ood_pred_entropy, dtype=int)])
    matched = entropy_matched_indices(combined_entropy, combined_labels, args.num_bins, rng)

    id_size = len(id_hard_idx)
    matched_id_local = matched[combined_labels[matched] == 0]
    matched_ood_local = matched[combined_labels[matched] == 1] - id_size
    matched_id_idx = id_hard_idx[matched_id_local]
    matched_ood_idx = ood_idx[matched_ood_local]

    matched_r = np.concatenate([cifar_metrics.r[matched_id_idx], svhn_metrics.r[matched_ood_idx]])
    matched_h_cont = np.concatenate([cifar_metrics.h_cont[matched_id_idx], svhn_metrics.h_cont[matched_ood_idx]])
    matched_labels = np.concatenate([
        np.zeros(len(matched_id_idx), dtype=int),
        np.ones(len(matched_ood_idx), dtype=int),
    ])

    plot_main_scatter(matched_r, matched_h_cont, matched_labels, out_dir / "matched_scatter.png")

    # Train/test split for the pair probe.
    feature_matrix = np.stack([matched_r, matched_h_cont], axis=1)
    all_indices = np.arange(len(matched_labels))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.3,
        stratify=matched_labels,
        random_state=args.seed,
    )
    x_train, x_test = feature_matrix[train_idx], feature_matrix[test_idx]
    y_train, y_test = matched_labels[train_idx], matched_labels[test_idx]
    pair_probe = LogisticRegression(max_iter=2000, random_state=args.seed)
    pair_probe.fit(x_train, y_train)
    pair_prob = pair_probe.predict_proba(x_test)[:, 1]

    heldout_index = np.zeros(len(matched_labels), dtype=bool)
    heldout_index[test_idx] = True

    matched_pred_entropy = np.concatenate([
        cifar_metrics.pred_entropy[matched_id_idx],
        svhn_metrics.pred_entropy[matched_ood_idx],
    ])
    matched_vacuity = np.concatenate([
        cifar_metrics.vacuity[matched_id_idx],
        svhn_metrics.vacuity[matched_ood_idx],
    ])
    matched_dissonance = np.concatenate([
        cifar_metrics.dissonance[matched_id_idx],
        svhn_metrics.dissonance[matched_ood_idx],
    ])
    matched_r_all = matched_r

    auroc_rows: List[Dict[str, object]] = []

    for name, score in [
        ("predictive_entropy", matched_pred_entropy[heldout_index]),
        ("vacuity", matched_vacuity[heldout_index]),
        ("dissonance", matched_dissonance[heldout_index]),
        ("resolution_ratio_r", matched_r_all[heldout_index]),
    ]:
        auroc, orientation = oriented_auroc(y_test, score)
        auroc_rows.append({"score": name, "auroc_best_orientation": auroc, "orientation": orientation})

    for beta in args.betas:
        q_id, h_id = completion_score(alpha_to_metrics(cifar_alpha[matched_id_idx]), beta)
        q_ood, h_ood = completion_score(alpha_to_metrics(svhn_alpha[matched_ood_idx]), beta)
        score = np.concatenate([h_id, h_ood])[heldout_index]
        auroc, orientation = oriented_auroc(y_test, score)
        auroc_rows.append(
            {
                "score": f"projected_entropy_beta_{format_beta(beta)}",
                "auroc_best_orientation": auroc,
                "orientation": orientation,
            }
        )

    auroc_rows.append(
        {
            "score": "pair_r_hcont_logistic_probe",
            "auroc_best_orientation": float(roc_auc_score(y_test, pair_prob)),
            "orientation": 1,
        }
    )
    save_csv(auroc_rows, out_dir / "matched_auroc.csv")

    # Mixed-stream risk/coverage.
    risk_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    id_error = (cifar_metrics.top1 != cifar_labels).astype(int)
    id_uncertainty = {
        "predictive_entropy": cifar_metrics.pred_entropy,
        "vacuity": cifar_metrics.vacuity,
        "dissonance": cifar_metrics.dissonance,
        "resolution_ratio_r": 1.0 - cifar_metrics.r,
    }
    for beta in args.betas:
        _, h_id = completion_score(cifar_metrics, beta)
        id_uncertainty[f"projected_entropy_beta_{format_beta(beta)}"] = h_id

    # Fit a pair probe on matched data once and reuse as an uncertainty score on the mixed stream.
    def pair_score_from_metrics(metrics) -> np.ndarray:
        return pair_probe.predict_proba(np.stack([metrics.r, metrics.h_cont], axis=1))[:, 1]

    id_uncertainty["pair_r_hcont_logistic_probe"] = pair_score_from_metrics(cifar_metrics)

    def mixed_stream_score(metrics, name: str) -> np.ndarray:
        if name == "predictive_entropy":
            return metrics.pred_entropy
        if name == "resolution_ratio_r":
            return 1.0 - metrics.r
        if name == "pair_r_hcont_logistic_probe":
            return pair_score_from_metrics(metrics)
        if name.startswith("projected_entropy_beta_"):
            beta_str = name.split("projected_entropy_beta_")[-1].replace("p", ".").replace("m", "-")
            beta = float(beta_str)
            _, h = completion_score(metrics, beta)
            return h
        return getattr(metrics, name)

    for eta in args.etas:
        total = min(len(cifar_alpha), len(svhn_alpha))
        n_ood = int(round(total * eta))
        n_id = total - n_ood
        id_choice = rng.choice(np.arange(len(cifar_alpha)), size=n_id, replace=False)
        ood_choice = rng.choice(np.arange(len(svhn_alpha)), size=n_ood, replace=False)

        mixed_errors = np.concatenate([id_error[id_choice], np.ones(n_ood, dtype=int)])
        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, score in id_uncertainty.items():
            ood_score = mixed_stream_score(svhn_metrics, name)[ood_choice]
            mixed_score = np.concatenate([score[id_choice], ood_score])
            coverage, risk = risk_coverage_curve(mixed_score, mixed_errors)
            curves[name] = (coverage, risk)
            risk_rows.append(
                {
                    "eta": eta,
                    "score": name,
                    "coverage_target": args.coverage_target,
                    "selective_risk": selective_risk_at_coverage(mixed_score, mixed_errors, args.coverage_target),
                }
            )
            if name.startswith("projected_entropy_beta_"):
                ablation_rows.append(
                    {
                        "setting": f"eta={eta}",
                        "beta": name.split("projected_entropy_beta_")[-1],
                        "matched_auroc": next(
                            row["auroc_best_orientation"]
                            for row in auroc_rows
                            if row["score"] == name
                        ),
                        "selective_risk_at_target_coverage": selective_risk_at_coverage(mixed_score, mixed_errors, args.coverage_target),
                    }
                )

        if eta == args.etas[0]:
            plot_risk_coverage(curves, out_dir / f"risk_coverage_eta_{eta:.2f}.png")

    save_csv(risk_rows, out_dir / "risk_coverage_summary.csv")
    save_csv(ablation_rows, out_dir / "beta_scan_ablation.csv")

    summary = {
        "matched_id_count": int(len(matched_id_idx)),
        "matched_ood_count": int(len(matched_ood_idx)),
        "checkpoint": args.checkpoint,
        "betas": args.betas,
        "etas": args.etas,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
