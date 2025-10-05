import os, torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from models.resnet_cifar import resnet18_cifar, resnet34_cifar

LABELS = [x.strip() for x in open(os.path.join(os.path.dirname(__file__), "labels", "cifar100.txt"), "r", encoding="utf-8").read().splitlines()]
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def load_model(weights_path="weights.pth", model="resnet18", device="cpu"):
    if model == "resnet18":
        net = resnet18_cifar(num_classes=100)
    else:
        net = resnet34_cifar(num_classes=100)
    state = torch.load(weights_path, map_location=device)
    sd = state.get("model", state)  # allow raw state_dict too
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()
    return net

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18")
WEIGHTS = os.environ.get("WEIGHTS", "weights.pth")
net = load_model(WEIGHTS, MODEL_NAME, device)

tfms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

def predict(image: Image.Image, topk: int = 5):
    x = tfms(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=-1)[0]
        topk_probs, topk_idx = probs.topk(topk)
    results = [(LABELS[i], float(topk_probs[j])) for j, i in enumerate(topk_idx)]
    return results

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Upload an image"), gr.Slider(1, 10, value=5, step=1, label="Top-K")],
    outputs=gr.Label(label="Top-K Predictions"),
    title="CIFAR-100 ResNet (from scratch)",
    description="Upload an image. The model was trained from scratch on CIFAR-100 (32×32).",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
