from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import cifar_resnet18
from utils_evidential import build_dataloader_kwargs, evidential_classification_loss, get_default_device, set_seed


def compute_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
    epoch: int,
    annealing_steps: int,
) -> torch.Tensor:
    if loss_name == "ce":
        return nn.functional.cross_entropy(logits, targets)
    return evidential_classification_loss(
        logits=logits,
        targets=targets,
        epoch=epoch,
        annealing_steps=annealing_steps,
    )


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int, train_augment: bool = True):
    try:
        from torchvision import datasets, transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for the real CIFAR-10/SVHN experiment. "
            "Install a matching torch/torchvision pair in your environment."
        ) from e

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    if train_augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        train_transform = eval_transform

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    eval_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)
    train_size = int(0.9 * len(train_dataset))
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(0)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(eval_dataset, val_indices)

    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    loader_kwargs = build_dataloader_kwargs(num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    epoch: int,
    annealing_steps: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = compute_classification_loss(logits, labels, loss_name, epoch, annealing_steps)
        preds = logits.argmax(dim=1)
        total_loss += loss.detach() * images.size(0)
        total_correct += (preds == labels).sum()
        total += images.size(0)
    if total == 0:
        return 0.0, 0.0
    return (total_loss / total).item(), (total_correct.float() / total).item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an evidential ResNet-18 on CIFAR-10.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./runs/cifar10_edl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--annealing-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--loss", type=str, choices=("edl", "ce"), default="edl")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"Using device: {device}")
    print(f"Training loss: {args.loss}")
    print(f"Train augmentation: {'off' if args.no_train_augment else 'on'}")
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        train_augment=not args.no_train_augment,
    )

    model = cifar_resnet18(num_classes=10).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Initialized model from {args.init_checkpoint}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        total = 0
        correct = torch.zeros((), dtype=torch.int64, device=device)
        progress = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = compute_classification_loss(logits, labels, args.loss, epoch, args.annealing_steps)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum()
                total += images.size(0)
                running_loss += loss.detach() * images.size(0)
                if step % max(1, args.log_interval) == 0 or step == len(train_loader):
                    progress.set_postfix(
                        loss=(running_loss / max(total, 1)).item(),
                        acc=(correct.float() / max(total, 1)).item(),
                    )

        scheduler.step()
        train_loss = (running_loss / max(total, 1)).item()
        train_acc = (correct.float() / max(total, 1)).item()
        val_loss, val_acc = evaluate(model, val_loader, device, args.loss, epoch, args.annealing_steps)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "history": history,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, out_dir / "best.pt")

    best_checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device, args.loss, args.epochs - 1, args.annealing_steps)
    print(
        json.dumps(
            {
                "best_epoch": best_checkpoint["epoch"],
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
    )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
