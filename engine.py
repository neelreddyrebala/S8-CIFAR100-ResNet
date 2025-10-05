from typing import Dict
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils import accuracy, maybe_mix, LabelSmoothingCE

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, num_classes=100, mix_prob=0.6, scaler: GradScaler=None, smoothing=0.1):
    model.train()
    loss_fn = LabelSmoothingCE(smoothing=smoothing, num_classes=num_classes)
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        images, soft_targets = maybe_mix(images, targets, num_classes=num_classes, p_mix=mix_prob)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = loss_fn(logits, soft_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            top1, top5 = accuracy(logits.float(), targets, topk=(1, 5))
            bs = images.size(0)
            running["loss"] += loss.item() * bs
            running["top1"] += top1 * bs / 100.0
            running["top5"] += top5 * bs / 100.0
            running["n"]    += bs
            pbar.set_postfix(loss=running["loss"]/running["n"], top1=100*running["top1"]/running["n"], lr=scheduler.get_last_lr()[0] if scheduler else 0.0)

    for k in ("loss","top1","top5"):
        running[k] = running[k] / running["n"]
    return running

@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "n": 0}
    loss_fn = torch.nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]")
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        top1, top5 = accuracy(logits.float(), targets, topk=(1, 5))
        bs = images.size(0)
        running["loss"] += loss.item() * bs
        running["top1"] += top1 * bs / 100.0
        running["top5"] += top5 * bs / 100.0
        running["n"]    += bs
        pbar.set_postfix(loss=running["loss"]/running["n"], top1=100*running["top1"]/running["n"])
    for k in ("loss","top1","top5"):
        running[k] = running[k] / running["n"]
    return running
