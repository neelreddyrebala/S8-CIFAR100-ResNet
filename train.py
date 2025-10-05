import argparse, os, time, math
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from data import get_loaders
from models.resnet_cifar import resnet18_cifar, resnet34_cifar
from engine import train_one_epoch, validate
from utils import set_seed

def get_model(name: str, num_classes=100, drop_p=0.0):
    name = name.lower()
    if name == "resnet18":
        return resnet18_cifar(num_classes=num_classes, drop_p=drop_p)
    elif name == "resnet34":
        return resnet34_cifar(num_classes=num_classes, drop_p=drop_p)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument("--max-lr", type=float, default=0.1, help="OneCycle max LR")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--mix-prob", type=float, default=0.6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--logdir", type=str, default="logs")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = get_model(args.model, num_classes=100, drop_p=args.dropout).to(device)

    train_loader, val_loader = get_loaders(batch_size=args.batch_size, workers=args.workers)

    # Optimizer & OneCycleLR
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        three_phase=False,
        anneal_strategy="cos"
    )

    scaler = GradScaler()
    best_top1 = 0.0
    log_md = os.path.join(args.logdir, "training_log.md")
    with open(log_md, "w", encoding="utf-8") as f:
        f.write("| epoch | lr | train_loss | train_top1 | train_top5 | val_loss | val_top1 | val_top5 |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_classes=100, mix_prob=args.mix_prob, scaler=scaler, smoothing=0.1)
        val_stats   = validate(model, val_loader, device, epoch)

        current_lr = scheduler.get_last_lr()[0]

        # Save last
        last_path = os.path.join(args.outdir, f"last_{args.model}_cifar100.pth")
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_stats["top1"],
                    "args": vars(args)}, last_path)

        # Save best
        if val_stats["top1"] > best_top1:
            best_top1 = val_stats["top1"]
            best_path = os.path.join(args.outdir, f"best_{args.model}_cifar100.pth")
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_top1": val_stats["top1"],
                        "args": vars(args)}, best_path)

        # Append Markdown log
        with open(log_md, "a", encoding="utf-8") as f:
            f.write(f"| {epoch} | {current_lr:.6f} | {train_stats['loss']:.4f} | {100*train_stats['top1']:.2f} | {100*train_stats['top5']:.2f} | {val_stats['loss']:.4f} | {100*val_stats['top1']:.2f} | {100*val_stats['top5']:.2f} |\n")

    print(f"Training complete. Best Top-1: {100*best_top1:.2f}%")

if __name__ == "__main__":
    main()
