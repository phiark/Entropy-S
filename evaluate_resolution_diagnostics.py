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
    EvidentialMetrics,
    alpha_to_metrics,
    build_dataloader_kwargs,
    completion_score,
    entropy_matched_indices,
    format_beta,
    get_default_device,
    oriented_auroc,
    risk_coverage_curve,
    save_json,
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


def plot_scatter(
    r: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.0))
    mask_id = labels == 0
    mask_ood = labels == 1
    plt.scatter(r[mask_id], y[mask_id], s=12, alpha=0.55, label="CIFAR-10 ID-hard")
    plt.scatter(r[mask_ood], y[mask_ood], s=12, alpha=0.55, label="SVHN OOD")
    plt.xlabel("resolution ratio r")
    plt.ylabel(ylabel)
    plt.title(title)
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


def normalize_index_array(values: object, upper_bound: int, name: str) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D index array.")
    if indices.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise ValueError(f"{name} contains out-of-range indices for upper bound {upper_bound}.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    return indices


def build_generated_cohort_manifest(
    cifar_metrics: EvidentialMetrics,
    svhn_metrics: EvidentialMetrics,
    cohort_size: int,
    num_bins: int,
    rng: np.random.Generator,
    checkpoint: str,
    seed: int,
) -> Dict[str, object]:
    id_hard_idx = np.argsort(cifar_metrics.margin)[:cohort_size]
    ood_idx = rng.choice(
        np.arange(len(svhn_metrics.alpha)),
        size=min(cohort_size, len(svhn_metrics.alpha)),
        replace=False,
    )

    id_pred_entropy = cifar_metrics.pred_entropy[id_hard_idx]
    ood_pred_entropy = svhn_metrics.pred_entropy[ood_idx]
    combined_entropy = np.concatenate([id_pred_entropy, ood_pred_entropy])
    combined_labels = np.concatenate(
        [
            np.zeros_like(id_pred_entropy, dtype=int),
            np.ones_like(ood_pred_entropy, dtype=int),
        ]
    )
    matched = entropy_matched_indices(combined_entropy, combined_labels, num_bins, rng)

    id_size = len(id_hard_idx)
    matched_id_local = matched[combined_labels[matched] == 0]
    matched_ood_local = matched[combined_labels[matched] == 1] - id_size
    matched_id_idx = id_hard_idx[matched_id_local]
    matched_ood_idx = ood_idx[matched_ood_local]

    return {
        "source": "generated",
        "checkpoint_used_to_define_cohort": checkpoint,
        "seed": seed,
        "cohort_size": int(cohort_size),
        "num_bins": int(num_bins),
        "id_hard_idx": id_hard_idx.tolist(),
        "ood_idx": ood_idx.tolist(),
        "matched_id_idx": matched_id_idx.tolist(),
        "matched_ood_idx": matched_ood_idx.tolist(),
    }


def load_or_create_cohort_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    rng: np.random.Generator,
    cifar_metrics: EvidentialMetrics,
    svhn_metrics: EvidentialMetrics,
) -> Tuple[Dict[str, object], Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.cohort_manifest:
        with Path(args.cohort_manifest).open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["source"] = "loaded"
        manifest["source_path"] = str(Path(args.cohort_manifest).resolve())
    else:
        manifest = build_generated_cohort_manifest(
            cifar_metrics=cifar_metrics,
            svhn_metrics=svhn_metrics,
            cohort_size=args.cohort_size,
            num_bins=args.num_bins,
            rng=rng,
            checkpoint=args.checkpoint,
            seed=args.seed,
        )

    id_hard_idx = normalize_index_array(manifest["id_hard_idx"], len(cifar_metrics.alpha), "id_hard_idx")
    ood_idx = normalize_index_array(manifest["ood_idx"], len(svhn_metrics.alpha), "ood_idx")
    matched_id_idx = normalize_index_array(manifest["matched_id_idx"], len(cifar_metrics.alpha), "matched_id_idx")
    matched_ood_idx = normalize_index_array(manifest["matched_ood_idx"], len(svhn_metrics.alpha), "matched_ood_idx")

    if matched_id_idx.size != matched_ood_idx.size:
        raise ValueError("matched_id_idx and matched_ood_idx must have the same length.")

    save_path = Path(args.save_cohort_manifest) if args.save_cohort_manifest else out_dir / "cohort_manifest.json"
    normalized_manifest = dict(manifest)
    normalized_manifest["id_hard_idx"] = id_hard_idx.tolist()
    normalized_manifest["ood_idx"] = ood_idx.tolist()
    normalized_manifest["matched_id_idx"] = matched_id_idx.tolist()
    normalized_manifest["matched_ood_idx"] = matched_ood_idx.tolist()
    normalized_manifest["matched_count"] = int(matched_id_idx.size)
    save_json(normalized_manifest, save_path)
    return normalized_manifest, save_path, id_hard_idx, ood_idx, matched_id_idx, matched_ood_idx


def pair_score_from_metrics(
    metrics: EvidentialMetrics,
    pair_probe: LogisticRegression,
    weighted_content: bool,
) -> np.ndarray:
    content = metrics.r * metrics.h_cont if weighted_content else metrics.h_cont
    return pair_probe.predict_proba(np.stack([metrics.r, content], axis=1))[:, 1]


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
    parser.add_argument(
        "--cohort-manifest",
        type=str,
        default=None,
        help="Optional fixed cohort manifest produced by a reference checkpoint.",
    )
    parser.add_argument(
        "--save-cohort-manifest",
        type=str,
        default=None,
        help="Optional path for the cohort manifest. Defaults to <out-dir>/cohort_manifest.json.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    matched_dir = out_dir / "matched_discrimination"
    selective_dir = out_dir / "selective_rejection"
    matched_dir.mkdir(parents=True, exist_ok=True)
    selective_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = get_default_device()
    print(f"Using device: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = cifar_resnet18(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])

    cifar_loader, svhn_loader = build_eval_loaders(args.data_dir, args.batch_size, args.num_workers)
    cifar_alpha, cifar_labels = infer_alpha(model, cifar_loader, device)
    svhn_alpha, _ = infer_alpha(model, svhn_loader, device)

    cifar_metrics = alpha_to_metrics(cifar_alpha)
    svhn_metrics = alpha_to_metrics(svhn_alpha)

    cohort_manifest, cohort_manifest_path, _, _, matched_id_idx, matched_ood_idx = load_or_create_cohort_manifest(
        args=args,
        out_dir=out_dir,
        rng=rng,
        cifar_metrics=cifar_metrics,
        svhn_metrics=svhn_metrics,
    )

    matched_r = np.concatenate([cifar_metrics.r[matched_id_idx], svhn_metrics.r[matched_ood_idx]])
    matched_h_cont = np.concatenate([cifar_metrics.h_cont[matched_id_idx], svhn_metrics.h_cont[matched_ood_idx]])
    matched_r_hcont_weighted = matched_r * matched_h_cont
    matched_labels = np.concatenate(
        [
            np.zeros(len(matched_id_idx), dtype=int),
            np.ones(len(matched_ood_idx), dtype=int),
        ]
    )

    plot_scatter(
        matched_r,
        matched_h_cont,
        matched_labels,
        ylabel="content entropy H_cont",
        title="Entropy-matched diagnostic scatter",
        path=matched_dir / "matched_scatter.png",
    )
    plot_scatter(
        matched_r,
        matched_r_hcont_weighted,
        matched_labels,
        ylabel="weighted content entropy r * H_cont",
        title="Entropy-matched geometry-aware scatter",
        path=matched_dir / "matched_scatter_weighted.png",
    )

    all_indices = np.arange(len(matched_labels))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.3,
        stratify=matched_labels,
        random_state=args.seed,
    )
    heldout_index = np.zeros(len(matched_labels), dtype=bool)
    heldout_index[test_idx] = True
    y_test = matched_labels[test_idx]

    raw_feature_matrix = np.stack([matched_r, matched_h_cont], axis=1)
    weighted_feature_matrix = np.stack([matched_r, matched_r_hcont_weighted], axis=1)

    pair_probe_raw = LogisticRegression(max_iter=2000, random_state=args.seed)
    pair_probe_raw.fit(raw_feature_matrix[train_idx], matched_labels[train_idx])
    pair_prob_raw = pair_probe_raw.predict_proba(raw_feature_matrix[test_idx])[:, 1]

    pair_probe_weighted = LogisticRegression(max_iter=2000, random_state=args.seed)
    pair_probe_weighted.fit(weighted_feature_matrix[train_idx], matched_labels[train_idx])
    pair_prob_weighted = pair_probe_weighted.predict_proba(weighted_feature_matrix[test_idx])[:, 1]

    matched_pred_entropy = np.concatenate(
        [
            cifar_metrics.pred_entropy[matched_id_idx],
            svhn_metrics.pred_entropy[matched_ood_idx],
        ]
    )
    matched_vacuity = np.concatenate(
        [
            cifar_metrics.vacuity[matched_id_idx],
            svhn_metrics.vacuity[matched_ood_idx],
        ]
    )
    matched_dissonance = np.concatenate(
        [
            cifar_metrics.dissonance[matched_id_idx],
            svhn_metrics.dissonance[matched_ood_idx],
        ]
    )

    auroc_rows: List[Dict[str, object]] = []
    for name, score in [
        ("predictive_entropy", matched_pred_entropy[heldout_index]),
        ("vacuity", matched_vacuity[heldout_index]),
        ("dissonance", matched_dissonance[heldout_index]),
        ("resolution_ratio_r", matched_r[heldout_index]),
        ("r_hcont_weighted", matched_r_hcont_weighted[heldout_index]),
    ]:
        auroc, orientation = oriented_auroc(y_test, score)
        auroc_rows.append({"score": name, "auroc_best_orientation": auroc, "orientation": orientation})

    for beta in args.betas:
        _, h_id = completion_score(alpha_to_metrics(cifar_alpha[matched_id_idx]), beta)
        _, h_ood = completion_score(alpha_to_metrics(svhn_alpha[matched_ood_idx]), beta)
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
            "auroc_best_orientation": float(roc_auc_score(y_test, pair_prob_raw)),
            "orientation": 1,
        }
    )
    auroc_rows.append(
        {
            "score": "pair_r_rhcont_logistic_probe",
            "auroc_best_orientation": float(roc_auc_score(y_test, pair_prob_weighted)),
            "orientation": 1,
        }
    )

    save_csv(auroc_rows, matched_dir / "matched_auroc.csv")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "cohort_manifest": str(cohort_manifest_path),
            "matched_id_count": int(len(matched_id_idx)),
            "matched_ood_count": int(len(matched_ood_idx)),
            "heldout_count": int(test_idx.size),
            "scores": [row["score"] for row in auroc_rows],
        },
        matched_dir / "summary.json",
    )

    risk_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    id_error = (cifar_metrics.top1 != cifar_labels).astype(int)
    id_uncertainty = {
        "predictive_entropy": cifar_metrics.pred_entropy,
        "vacuity": cifar_metrics.vacuity,
        "dissonance": cifar_metrics.dissonance,
        "resolution_ratio_r": 1.0 - cifar_metrics.r,
        "r_hcont_weighted": cifar_metrics.r * cifar_metrics.h_cont,
        "pair_r_hcont_logistic_probe": pair_score_from_metrics(cifar_metrics, pair_probe_raw, weighted_content=False),
        "pair_r_rhcont_logistic_probe": pair_score_from_metrics(cifar_metrics, pair_probe_weighted, weighted_content=True),
    }
    for beta in args.betas:
        _, h_id = completion_score(cifar_metrics, beta)
        id_uncertainty[f"projected_entropy_beta_{format_beta(beta)}"] = h_id

    def mixed_stream_score(metrics: EvidentialMetrics, name: str) -> np.ndarray:
        if name == "predictive_entropy":
            return metrics.pred_entropy
        if name == "resolution_ratio_r":
            return 1.0 - metrics.r
        if name == "r_hcont_weighted":
            return metrics.r * metrics.h_cont
        if name == "pair_r_hcont_logistic_probe":
            return pair_score_from_metrics(metrics, pair_probe_raw, weighted_content=False)
        if name == "pair_r_rhcont_logistic_probe":
            return pair_score_from_metrics(metrics, pair_probe_weighted, weighted_content=True)
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
                        "selective_risk_at_target_coverage": selective_risk_at_coverage(
                            mixed_score,
                            mixed_errors,
                            args.coverage_target,
                        ),
                    }
                )

        plot_risk_coverage(curves, selective_dir / f"risk_coverage_eta_{eta:.2f}.png")

    save_csv(risk_rows, selective_dir / "risk_coverage_summary.csv")
    save_csv(ablation_rows, selective_dir / "beta_scan_ablation.csv")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "cohort_manifest": str(cohort_manifest_path),
            "coverage_target": args.coverage_target,
            "etas": args.etas,
            "scores": list(id_uncertainty.keys()),
        },
        selective_dir / "summary.json",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "cohort_manifest": str(cohort_manifest_path),
        "cohort_source": cohort_manifest.get("source"),
        "matched_id_count": int(len(matched_id_idx)),
        "matched_ood_count": int(len(matched_ood_idx)),
        "betas": args.betas,
        "etas": args.etas,
        "matched_output_dir": str(matched_dir),
        "selective_output_dir": str(selective_dir),
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
