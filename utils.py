import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int = 42):
    import numpy as np, random as pyrand
    import torch
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=100):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, target):
        # target may be soft (mixup/cutmix) or hard indices
        if target.dtype in (torch.float16, torch.float32):
            log_probs = F.log_softmax(x, dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
            return loss
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        log_probs = F.log_softmax(x, dim=-1)
        return -(true_dist * log_probs).sum(dim=-1).mean()

def _rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=0.2, num_classes=100):
    if alpha <= 0.:
        return x, y
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y

def cutmix_data(x, y, alpha=1.0, num_classes=100):
    if alpha <= 0.:
        return x, y
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = _rand_bbox(W, H, lam)
    x2 = x[index, :].clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_b = torch.nn.functional.one_hot(y[index], num_classes=num_classes).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    return x, mixed_y

def maybe_mix(inputs, targets, num_classes=100, p_mix=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
    if torch.rand(()) > p_mix:
        return inputs, targets  # no mix
    if torch.rand(()) < 0.5:
        return mixup_data(inputs, targets, alpha=mixup_alpha, num_classes=num_classes)
    else:
        return cutmix_data(inputs, targets, alpha=cutmix_alpha, num_classes=num_classes)
