"""生成式超分演示：对同一张 LR，采样多个 z ~ p(z|c) 生成多张不同的 HR。

展示空间隐变量 VAE 的生成能力（这是 VAE 相比确定性 CNN 的独特价值）。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio
from torchvision.transforms import functional as TF

from src.dataset import EXTS
from src.model import SpatialCVAE
from src.utils import get_device, tensor_to_pil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/spatial_vae.pt")
    p.add_argument("--input", default=None, help="HR image path; default = first held-out valid image")
    p.add_argument("--output_dir", default="outputs/demo")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max_side", type=int, default=800)
    args = p.parse_args()

    device = get_device()
    model = SpatialCVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.input:
        img = Image.open(args.input).convert("RGB")
    else:
        paths = sorted(x for x in Path("data/DIV2K_valid_HR").rglob("*") if x.suffix.lower() in EXTS)
        img = Image.open(paths[120]).convert("RGB")  # first held-out image

    w, h = img.size
    w, h = w - w % 2, h - h % 2
    img = img.crop((0, 0, w, h))
    if max(w, h) > args.max_side:
        s = args.max_side / max(w, h)
        w2, h2 = round(w * s) // 2 * 2, round(h * s) // 2 * 2
        img = img.resize((w2, h2), Image.BICUBIC)
        w, h = w2, h2

    lr = img.resize((w // 2, h // 2), Image.BICUBIC)
    bic = lr.resize((w, h), Image.BICUBIC)
    c = TF.to_tensor(bic).unsqueeze(0).to(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        mean = tensor_to_pil(model.infer(c)[0])
        p_mu, p_logvar = model.encode_prior(c)
        samples = [tensor_to_pil(model.decode(c, model.reparameterize(p_mu, p_logvar))[0])
                   for _ in range(args.samples)]

    bic.save(out / "demo_bicubic.png")
    mean.save(out / "demo_mean.png")
    for i, s in enumerate(samples):
        s.save(out / f"demo_sample{i + 1}.png")

    lr_near = lr.resize((w, h), Image.NEAREST)
    panels = [("LR", lr_near), ("Bicubic", bic), ("Mean z=mu_p", mean)] + \
             [(f"Sample {i + 1}", s) for i, s in enumerate(samples)]
    label_h = 24
    canvas = Image.new("RGB", (w * len(panels), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        canvas.paste(panel, (i * w, label_h))
        draw.text((i * w + 6, 6), label, fill="black")
    canvas.save(out / "demo_grid.png")

    gt = np.asarray(img).astype("uint8")
    def psnr(pil):
        return peak_signal_noise_ratio(gt, np.asarray(pil).astype("uint8"), data_range=255)

    print(f"PSNR: bicubic={psnr(bic):.3f}  mean={psnr(mean):.3f}")
    for i, s in enumerate(samples):
        print(f"  sample{i + 1}={psnr(s):.3f}")
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
