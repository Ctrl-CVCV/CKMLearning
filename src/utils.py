from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, epoch, loss):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "loss": float(loss),
                "latent_dim": model.latent_dim}, path)


def load_model(checkpoint, device):
    from src.model import SpatialCVAE
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
        latent_dim = state.get("latent_dim", 64)
    else:
        sd = state
        latent_dim = sd["q_mu.weight"].shape[0]
    model = SpatialCVAE(latent_dim=latent_dim)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def tensor_to_pil(t):
    t = t.detach().cpu().clamp(0, 1)
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8"))


def psnr_torch(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    import math
    return 10 * math.log10(1.0 / mse)


def plot_losses(csv_path, output_path):
    import csv
    epochs, losses = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Spatial CVAE training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


@torch.no_grad()
def infer_tiled(model, image, tile_size=256, overlap=32):
    """Run full-resolution conditioning image in overlapping tiles to limit memory use."""
    _, _, height, width = image.shape
    if tile_size <= overlap:
        raise ValueError("tile_size must be larger than overlap")
    stride = tile_size - overlap
    output = torch.zeros_like(image)
    weight_sum = torch.zeros((1, 1, height, width), device=image.device, dtype=image.dtype)
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(height - tile_size, 0))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(width - tile_size, 0))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            y2, x2 = min(y + tile_size, height), min(x + tile_size, width)
            patch = image[:, :, y:y2, x:x2]
            pred = model.infer(patch)
            ph, pw = pred.shape[-2:]
            wy = torch.ones(ph, device=image.device, dtype=image.dtype)
            wx = torch.ones(pw, device=image.device, dtype=image.dtype)
            ramp = min(overlap, ph // 2, pw // 2)
            if ramp > 0:
                edge = torch.linspace(0.05, 1.0, ramp, device=image.device, dtype=image.dtype)
                if y > 0:
                    wy[:ramp] = edge
                if y2 < height:
                    wy[-ramp:] = edge.flip(0)
                if x > 0:
                    wx[:ramp] = edge
                if x2 < width:
                    wx[-ramp:] = edge.flip(0)
            weight = (wy[:, None] * wx[None, :])[None, None]
            output[:, :, y:y2, x:x2] += pred * weight
            weight_sum[:, :, y:y2, x:x2] += weight
    return (output / weight_sum.clamp_min(1e-8)).clamp(0, 1)
