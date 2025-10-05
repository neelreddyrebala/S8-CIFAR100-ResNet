# ERA V4 — Session 8: CIFAR-100 ResNet (from scratch) + Hugging Face Space

## What’s inside
- `train.py` — Train a CIFAR-100 classifier (ResNet-18/34) from scratch with OneCycle policy, MixUp/CutMix, Label Smoothing, AMP.
- `gradcam.py` — Generate Grad-CAM heatmaps for trained checkpoints.
- `models/resnet_cifar.py` — ResNet for CIFAR (3×3 stem, no maxpool).
- `data.py` — Dataloaders & augmentations (RandAugment, RandomErasing).
- `engine.py` — Train/validate loops with metrics.
- `utils.py` — MixUp/CutMix, LabelSmoothingCE, accuracy helpers, seed utils.
- `labels/cifar100.txt` — Class names for inference apps.
- `hf_space/` — Drop-in Gradio app for Hugging Face Spaces.

## Quickstart (Windows / Cursor)
```powershell
cd "C:\Work\TSAI\ERAV4 Session 8 Assignment"
# Copy this repo's contents here

py -m venv .venv
.\.venv\Scripts\activate

py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### Train (100 epochs, batch 128)
```powershell
py train.py --epochs 100 --batch-size 128 --model resnet18 --max-lr 0.1 --workers 4
```

> If you run out of VRAM, try `--batch-size 64` and/or `--model resnet34 --max-lr 0.05`.

### Logs (Markdown)
The training script writes `logs/training_log.md` with one line per epoch (Epoch, LR, Train/Val loss, Top-1/Top-5).
Submit this file as per assignment.

### Checkpoints
- Best model (by validation Top-1): `checkpoints/best_{model}_cifar100.pth`
- Last epoch: `checkpoints/last_{model}_cifar100.pth`

### Grad-CAM
```powershell
# Example: visualize 16 samples from the test set
py gradcam.py --ckpt checkpoints/best_resnet18_cifar100.pth --num-samples 16
```
Output will be saved under `outputs/gradcam_grid.png`.

### Hugging Face Space (Gradio)
- Create a **Space** (Gradio + Python).
- Copy `hf_space/app.py`, `hf_space/requirements.txt`, `models/resnet_cifar.py`, `labels/cifar100.txt` and your checkpoint (rename to `weights.pth`) into the Space repo root.
- Commit & push. Space will build and serve a UI that returns **Top-5 predictions**.

---
**Note:** No pretrained weights are used. This trains ResNet from scratch.
