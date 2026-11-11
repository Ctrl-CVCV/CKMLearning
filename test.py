import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from src.utils import get_device, load_model, tensor_to_pil, infer_tiled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output_dir", default="outputs/test")
    args = p.parse_args()

    device = get_device()
    model = load_model(args.checkpoint, device)

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    w2, h2 = w - (w % 2), h - (h % 2)
    img = img.crop((0, 0, w2, h2))

    lr = img.resize((w2 // 2, h2 // 2), Image.BICUBIC)
    lr_up = lr.resize((w2, h2), Image.BICUBIC)

    lr_t = TF.to_tensor(lr_up).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = infer_tiled(model, lr_t)[0]
    sr = tensor_to_pil(sr_t)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    lr.save(out / f"{stem}_lr.png")
    lr_up.save(out / f"{stem}_bicubic.png")
    sr.save(out / f"{stem}_vae.png")

    # Side‑by‑side comparison with labels.
    lr_nearest = lr.resize((w2, h2), Image.NEAREST)
    panels = [("LR (nearest enlarged)", lr_nearest), ("Bicubic x2", lr_up), ("CVAE x2", sr)]
    # first panel intentionally shows the bicubic‑sized input condition; raw LR is saved separately.
    label_h = 28
    canvas = Image.new("RGB", (w2 * 3, h2 + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        canvas.paste(panel, (i * w2, label_h))
        draw.text((i * w2 + 8, 7), label, fill="black")
    canvas.save(out / f"{stem}_comparison.png")
    print("Saved results to", out)


if __name__ == "__main__":
    main()

