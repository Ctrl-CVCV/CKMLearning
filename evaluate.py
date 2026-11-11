import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.utils import get_device, load_model, tensor_to_pil, infer_tiled
from src.dataset import EXTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--hr_dir", default="data/DIV2K_valid_HR")
    p.add_argument("--output_dir", default="outputs/evaluation")
    args = p.parse_args()
    device = get_device()
    model = load_model(args.checkpoint, device)
    paths = sorted(x for x in Path(args.hr_dir).rglob("*") if x.suffix.lower() in EXTS)
    if not paths:
        raise RuntimeError(f"No HR images found in {args.hr_dir}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(paths, desc="Evaluating"):
        hr = Image.open(path).convert("RGB")
        w, h = hr.size
        w, h = w - w % 2, h - h % 2
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // 2, h // 2), Image.Resampling.BICUBIC)
        bic = lr.resize((w, h), Image.Resampling.BICUBIC)
        lr_up_t = TF.to_tensor(bic).unsqueeze(0).to(device)
        with torch.no_grad():
            sr = tensor_to_pil(infer_tiled(model, lr_up_t)[0])
        gt_arr = np.asarray(hr).astype(np.uint8)
        bic_arr = np.asarray(bic).astype(np.uint8)
        sr_arr = np.asarray(sr).astype(np.uint8)
        gt_f = gt_arr.astype(np.float32)
        bic_psnr = peak_signal_noise_ratio(gt_arr, bic_arr, data_range=255)
        sr_psnr = peak_signal_noise_ratio(gt_arr, sr_arr, data_range=255)
        bic_ssim = structural_similarity(gt_arr, bic_arr, channel_axis=2, data_range=255)
        sr_ssim = structural_similarity(gt_arr, sr_arr, channel_axis=2, data_range=255)
        bic_mae = np.mean(np.abs(gt_f - bic_arr.astype(np.float32)))
        sr_mae = np.mean(np.abs(gt_f - sr_arr.astype(np.float32)))
        rows.append([path.name, bic_psnr, sr_psnr, bic_ssim, sr_ssim, bic_mae, sr_mae])
        sr.save(out / f"{path.stem}_vae.png")
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "bicubic_psnr", "vae_psnr", "bicubic_ssim", "vae_ssim",
                         "bicubic_mae", "vae_mae"])
        writer.writerows(rows)
        means = np.array([r[1:] for r in rows], dtype=float).mean(axis=0)
        writer.writerow(["MEAN", *means.tolist()])
    print("Mean Bicubic PSNR / VAE PSNR:", means[0], "/", means[1])
    print("Mean Bicubic SSIM / VAE SSIM:", means[2], "/", means[3])
    print("Mean Bicubic MAE  / VAE MAE :", means[4], "/", means[5])
    print("Metrics saved to", out / "metrics.csv")


if __name__ == "__main__":
    main()
