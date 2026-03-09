from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from utils_evidential import (
    alpha_to_metrics,
    completion_score,
    entropy_matched_indices,
    format_beta,
    oriented_auroc,
    risk_coverage_curve,
    selective_risk_at_coverage,
)


def sample_id_hard_alpha(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    total_evidence = rng.lognormal(mean=4.0, sigma=0.35, size=n)
    alpha = np.ones((n, k), dtype=np.float64)
    for i in range(n):
        pair = rng.choice(k, size=2, replace=False)
        weights = np.zeros(k, dtype=np.float64)
        main_share = rng.uniform(0.38, 0.47)
        second_share = rng.uniform(0.28, 0.38)
        remaining = max(0.0, 1.0 - main_share - second_share)
        weights[pair[0]] = main_share
        weights[pair[1]] = second_share
        tail = rng.dirichlet(np.full(k - 2, 1.5)) * remaining
        tail_indices = [idx for idx in range(k) if idx not in pair]
        weights[tail_indices] = tail
        evidence = total_evidence[i] * weights
        alpha[i] = evidence + 1.0
    return alpha


def sample_ood_alpha(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    total_evidence = rng.lognormal(mean=2.0, sigma=0.45, size=n)
    alpha = np.ones((n, k), dtype=np.float64)
    for i in range(n):
        dominant = rng.integers(0, k)
        second = int((dominant + rng.integers(1, k)) % k)
        weights = np.zeros(k, dtype=np.float64)
        main_share = rng.uniform(0.50, 0.78)
        second_share = rng.uniform(0.03, 0.15)
        remaining = max(0.0, 1.0 - main_share - second_share)
        weights[dominant] = main_share
        weights[second] = second_share
        tail = rng.dirichlet(np.full(k - 2, 1.0)) * remaining
        tail_indices = [idx for idx in range(k) if idx not in (dominant, second)]
        weights[tail_indices] = tail
        evidence = total_evidence[i] * weights
        alpha[i] = evidence + 1.0
    return alpha


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(r: np.ndarray, h_cont: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(r[labels == 0], h_cont[labels == 0], s=10, alpha=0.55, label="ID-hard")
    plt.scatter(r[labels == 1], h_cont[labels == 1], s=10, alpha=0.55, label="OOD")
    plt.xlabel("resolution ratio r")
    plt.ylabel("content entropy H_cont")
    plt.title("Synthetic entropy-matched diagnostic scatter")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_entropy_overlap(id_entropy: np.ndarray, ood_entropy: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(id_entropy, bins=30, alpha=0.55, density=True, label="ID-hard")
    plt.hist(ood_entropy, bins=30, alpha=0.55, density=True, label="OOD")
    plt.xlabel("predictive entropy")
    plt.ylabel("density")
    plt.title("Synthetic predictive-entropy overlap before matching")
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
    plt.title("Synthetic risk-coverage curves")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic resolution-aware diagnostics demo.")
    parser.add_argument("--out-dir", type=str, default="./synthetic_demo")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-id", type=int, default=6000)
    parser.add_argument("--n-ood", type=int, default=6000)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=18)
    parser.add_argument("--betas", type=float, nargs="*", default=[0.1, 0.5, 0.75])
    parser.add_argument("--coverage-target", type=float, default=0.8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    id_alpha = sample_id_hard_alpha(args.n_id, args.num_classes, rng)
    ood_alpha = sample_ood_alpha(args.n_ood, args.num_classes, rng)
    id_metrics = alpha_to_metrics(id_alpha)
    ood_metrics = alpha_to_metrics(ood_alpha)

    plot_entropy_overlap(id_metrics.pred_entropy, ood_metrics.pred_entropy, out_dir / "predictive_entropy_overlap.png")

    combined_entropy = np.concatenate([id_metrics.pred_entropy, ood_metrics.pred_entropy])
    combined_labels = np.concatenate([
        np.zeros(args.n_id, dtype=int),
        np.ones(args.n_ood, dtype=int),
    ])
    matched = entropy_matched_indices(combined_entropy, combined_labels, args.num_bins, rng)
    matched_labels = combined_labels[matched]

    matched_r = np.concatenate([id_metrics.r, ood_metrics.r])[matched]
    matched_h_cont = np.concatenate([id_metrics.h_cont, ood_metrics.h_cont])[matched]
    matched_pred_entropy = combined_entropy[matched]
    matched_vacuity = np.concatenate([id_metrics.vacuity, ood_metrics.vacuity])[matched]
    matched_dissonance = np.concatenate([id_metrics.dissonance, ood_metrics.dissonance])[matched]
    matched_error_proxy = matched_labels.copy()  # accepting OOD is treated as an error proxy in the mixed stream.

    plot_scatter(matched_r, matched_h_cont, matched_labels, out_dir / "matched_scatter.png")

    feature_matrix = np.stack([matched_r, matched_h_cont], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        matched_labels,
        test_size=0.3,
        stratify=matched_labels,
        random_state=args.seed,
    )
    pair_probe = LogisticRegression(max_iter=2000, random_state=args.seed)
    pair_probe.fit(x_train, y_train)
    pair_prob = pair_probe.predict_proba(x_test)[:, 1]

    heldout_index = np.zeros(len(matched_labels), dtype=bool)
    for row in x_test:
        match = np.all(np.isclose(feature_matrix, row[None, :], atol=1e-12), axis=1)
        first = np.flatnonzero(match & (~heldout_index))
        if len(first):
            heldout_index[first[0]] = True

    auroc_rows: List[Dict[str, object]] = []
    for name, score in [
        ("predictive_entropy", matched_pred_entropy[heldout_index]),
        ("vacuity", matched_vacuity[heldout_index]),
        ("dissonance", matched_dissonance[heldout_index]),
        ("resolution_ratio_r", matched_r[heldout_index]),
    ]:
        auroc, orientation = oriented_auroc(y_test, score)
        auroc_rows.append({"score": name, "auroc_best_orientation": auroc, "orientation": orientation})

    for beta in args.betas:
        q_id, h_id = completion_score(id_metrics, beta)
        q_ood, h_ood = completion_score(ood_metrics, beta)
        matched_h_beta = np.concatenate([h_id, h_ood])[matched][heldout_index]
        auroc, orientation = oriented_auroc(y_test, matched_h_beta)
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

    mixed_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    risk_rows: List[Dict[str, object]] = []
    uncertainty_scores = {
        "predictive_entropy": matched_pred_entropy,
        "vacuity": matched_vacuity,
        "dissonance": matched_dissonance,
        "resolution_ratio_r": 1.0 - matched_r,
        "pair_r_hcont_logistic_probe": pair_probe.predict_proba(feature_matrix)[:, 1],
    }
    for beta in args.betas:
        _, h_id = completion_score(id_metrics, beta)
        _, h_ood = completion_score(ood_metrics, beta)
        uncertainty_scores[f"projected_entropy_beta_{format_beta(beta)}"] = np.concatenate([h_id, h_ood])[matched]

    for name, score in uncertainty_scores.items():
        coverage, risk = risk_coverage_curve(score, matched_error_proxy)
        mixed_curves[name] = (coverage, risk)
        risk_rows.append(
            {
                "score": name,
                "coverage_target": args.coverage_target,
                "selective_risk": selective_risk_at_coverage(score, matched_error_proxy, args.coverage_target),
            }
        )
    save_csv(risk_rows, out_dir / "risk_coverage_summary.csv")
    plot_risk_coverage(mixed_curves, out_dir / "risk_coverage.png")

    summary = {
        "n_id": args.n_id,
        "n_ood": args.n_ood,
        "matched_total": int(len(matched)),
        "matched_id": int((matched_labels == 0).sum()),
        "matched_ood": int((matched_labels == 1).sum()),
        "pair_probe_auroc": float(roc_auc_score(y_test, pair_prob)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
