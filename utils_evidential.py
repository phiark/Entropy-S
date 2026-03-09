from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader_kwargs(num_workers: int) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


EPS = 1e-12


def binary_entropy_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0 - EPS)
    return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))


def categorical_entropy_np(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, 1.0)
    return -(probs * np.log2(probs)).sum(axis=1)


def format_beta(beta: float) -> str:
    return f"{beta:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1))."""
    num_classes = alpha.shape[1]
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    second = -torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
    third = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return (first + second + third).squeeze(1)


def evidential_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    annealing_steps: int,
    evidence_fn: str = "softplus",
) -> torch.Tensor:
    if evidence_fn == "relu":
        evidence = F.relu(logits)
    elif evidence_fn == "exp":
        evidence = torch.exp(torch.clamp(logits, -10.0, 10.0))
    else:
        evidence = F.softplus(logits)

    alpha = evidence + 1.0
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    probs = alpha / sum_alpha
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

    sq_error = (one_hot - probs).pow(2)
    variance = alpha * (sum_alpha - alpha) / (sum_alpha.pow(2) * (sum_alpha + 1.0))
    data_fit = (sq_error + variance).sum(dim=1)

    anneal = min(1.0, float(epoch + 1) / float(max(1, annealing_steps)))
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    kl_reg = dirichlet_kl_to_uniform(alpha_tilde)
    return (data_fit + anneal * kl_reg).mean()


@dataclass
class EvidentialMetrics:
    alpha: np.ndarray
    probs: np.ndarray
    belief: np.ndarray
    uncertainty: np.ndarray
    top1: np.ndarray
    top1_prob: np.ndarray
    second_prob: np.ndarray
    margin: np.ndarray
    pred_entropy: np.ndarray
    vacuity: np.ndarray
    dissonance: np.ndarray
    p_t: np.ndarray
    p_f: np.ndarray
    p_u: np.ndarray
    r: np.ndarray
    tau: np.ndarray
    h_cont: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "top1": self.top1,
            "top1_prob": self.top1_prob,
            "second_prob": self.second_prob,
            "margin": self.margin,
            "pred_entropy": self.pred_entropy,
            "vacuity": self.vacuity,
            "dissonance": self.dissonance,
            "p_t": self.p_t,
            "p_f": self.p_f,
            "p_u": self.p_u,
            "r": self.r,
            "tau": self.tau,
            "h_cont": self.h_cont,
        }


def dissonance_from_belief_np(belief: np.ndarray) -> np.ndarray:
    n, k = belief.shape
    b_i = belief[:, :, None]
    b_j = belief[:, None, :]
    denom = b_i + b_j
    balance = 1.0 - np.abs(b_i - b_j) / np.clip(denom, EPS, None)
    balance = np.where(denom > EPS, balance, 0.0)
    for idx in range(k):
        balance[:, idx, idx] = 0.0
    numerator = (belief[:, None, :] * balance).sum(axis=2)
    denominator = belief.sum(axis=1, keepdims=True) - belief
    out = np.where(denominator > EPS, belief * numerator / np.clip(denominator, EPS, None), 0.0)
    return out.sum(axis=1)


def alpha_to_metrics(alpha: np.ndarray) -> EvidentialMetrics:
    alpha = np.asarray(alpha, dtype=np.float64)
    evidence = np.clip(alpha - 1.0, 0.0, None)
    sum_alpha = alpha.sum(axis=1, keepdims=True)
    probs = alpha / np.clip(sum_alpha, EPS, None)
    belief = evidence / np.clip(sum_alpha, EPS, None)
    num_classes = alpha.shape[1]
    uncertainty = np.full(alpha.shape[0], float(num_classes)) / np.clip(sum_alpha[:, 0], EPS, None)

    top1 = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top1_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2]
    margin = top1_prob - second_prob

    rows = np.arange(alpha.shape[0])
    p_t = belief[rows, top1]
    p_f = belief.sum(axis=1) - p_t
    p_u = uncertainty
    r = 1.0 - p_u
    tau = np.divide(p_t, r, out=np.zeros_like(p_t), where=r > EPS)
    h_cont = binary_entropy_np(tau)
    pred_entropy = categorical_entropy_np(probs)
    vacuity = p_u
    dissonance = dissonance_from_belief_np(belief)

    return EvidentialMetrics(
        alpha=alpha,
        probs=probs,
        belief=belief,
        uncertainty=uncertainty,
        top1=top1,
        top1_prob=top1_prob,
        second_prob=second_prob,
        margin=margin,
        pred_entropy=pred_entropy,
        vacuity=vacuity,
        dissonance=dissonance,
        p_t=p_t,
        p_f=p_f,
        p_u=p_u,
        r=r,
        tau=tau,
        h_cont=h_cont,
    )


def completion_score(metrics: EvidentialMetrics, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    q_beta = metrics.p_t + beta * metrics.p_u
    h_beta = binary_entropy_np(q_beta)
    return q_beta, h_beta


def entropy_matched_indices(
    entropy: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    entropy = np.asarray(entropy)
    labels = np.asarray(labels)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(entropy, quantiles)
    if np.unique(edges).shape[0] < 3:
        edges = np.linspace(entropy.min(), entropy.max() + 1e-8, num_bins + 1)
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    chosen: List[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (entropy >= low) & (entropy < high)
        id_idx = np.flatnonzero(mask & (labels == 0))
        ood_idx = np.flatnonzero(mask & (labels == 1))
        take = min(len(id_idx), len(ood_idx))
        if take == 0:
            continue
        chosen.extend(rng.choice(id_idx, size=take, replace=False).tolist())
        chosen.extend(rng.choice(ood_idx, size=take, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def oriented_auroc(labels: np.ndarray, score: np.ndarray) -> Tuple[float, int]:
    labels = np.asarray(labels)
    score = np.asarray(score)
    raw = roc_auc_score(labels, score)
    if raw >= 0.5:
        return float(raw), 1
    return float(1.0 - raw), -1


def train_pair_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float = 0.3,
) -> Tuple[LogisticRegression, Dict[str, np.ndarray], Dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, y_train)
    test_prob = clf.predict_proba(x_test)[:, 1]
    auroc = roc_auc_score(y_test, test_prob)
    return clf, {"x_test": x_test, "y_test": y_test, "test_prob": test_prob}, {"auroc": float(auroc)}


def risk_coverage_curve(
    uncertainty: np.ndarray,
    errors: np.ndarray,
    num_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    uncertainty = np.asarray(uncertainty)
    errors = np.asarray(errors).astype(np.float64)
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    cum_errors = np.cumsum(sorted_errors)
    n = len(sorted_errors)
    accepted = np.arange(1, n + 1)
    coverage = accepted / float(n)
    risk = cum_errors / accepted
    if num_points and n > num_points:
        keep = np.linspace(0, n - 1, num_points).astype(int)
        coverage = coverage[keep]
        risk = risk[keep]
    return coverage, risk


def selective_risk_at_coverage(
    uncertainty: np.ndarray,
    errors: np.ndarray,
    target_coverage: float,
) -> float:
    coverage, risk = risk_coverage_curve(uncertainty, errors, num_points=0)
    idx = int(np.argmin(np.abs(coverage - target_coverage)))
    return float(risk[idx])


def save_json(data: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
