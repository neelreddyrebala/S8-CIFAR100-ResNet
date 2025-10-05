import argparse, os
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets
from models.resnet_cifar import resnet18_cifar, resnet34_cifar

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def get_model(name, num_classes=100):
    if name.lower() == "resnet18":
        return resnet18_cifar(num_classes=num_classes)
    else:
        return resnet34_cifar(num_classes=num_classes)

class GradCAM:
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        target_layer = dict([*self.model.named_modules()])[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, index=None):
        logits = self.model(x)
        if index is None:
            index = logits.argmax(dim=1)
        selected = logits.gather(1, index.view(-1,1)).squeeze()
        self.model.zero_grad()
        selected.backward(torch.ones_like(selected), retain_graph=True)

        grads = self.gradients   # [B, C, H, W]
        acts  = self.activations # [B, C, H, W]
        weights = grads.mean(dim=(2,3), keepdim=True)  # GAP over H,W

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        # normalize to [0,1]
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1,1,1,1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return logits, cam

def denorm(x):
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(CIFAR100_STD,  device=x.device).view(1,3,1,1)
    return x * std + mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet34"])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out", type=str, default="outputs/gradcam_grid.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Data
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfms)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.num-samples, shuffle=True, num_workers=2)

    images, targets = next(iter(loader))
    images, targets = images.to(device), targets.to(device)

    gc = GradCAM(model, target_layer_name="layer4")
    logits, cams = gc(images)
    probs = torch.softmax(logits, dim=-1).max(dim=1)[0]

    # overlay
    images_denorm = denorm(images).clamp(0,1)
    cams = F.interpolate(cams, size=images_denorm.shape[-2:], mode="bilinear", align_corners=False)
    overlay = (0.5*images_denorm + 0.5*cams.repeat(1,3,1,1)).clamp(0,1)

    grid = make_grid(overlay, nrow=int(args.num-samples ** 0.5))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_image(grid, args.out)
    print(f"Saved Grad-CAM grid to {args.out}")

if __name__ == "__main__":
    main()
