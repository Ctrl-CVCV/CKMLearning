"""正规三段式训练：train / val / test。

从本地 211 张 HR 图中切出：
  - 训练集 171 张
  - 验证集  20 张（每 val_every 个 epoch 评估一次，用于按验证 PSNR 选最优模型）
  - 测试集  20 张（最后单独评估，全程不参与训练）

训练结束后保存 outputs/checkpoints/val_best.pt（按验证集 PSNR 最佳），并在测试集上报 PSNR/SSIM/MAE。
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.dataset import EXTS
from src.model import SpatialCVAE, vae_loss
from src.utils import get_device, seed_everything, save_checkpoint, infer_tiled


class PatchDataset(Dataset):
    def __init__(self, paths, crop_size=192, scale=2, crops_per_img=8):
        self.crop_size = crop_size
        self.scale = scale
        self.crops_per_img = crops_per_img
        self.imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            if w < crop_size or h < crop_size:
                r = max(crop_size / w, crop_size / h)
                im = im.resize((round(w * r), round(h * r)), Image.Resampling.BICUBIC)
            self.imgs.append(np.asarray(im, dtype=np.uint8))

    def __len__(self):
        return len(self.imgs) * self.crops_per_img

    def __getitem__(self, idx):
        arr = self.imgs[idx % len(self.imgs)]
        h, w = arr.shape[:2]
        s = self.crop_size
        top = random.randint(0, h - s)
        left = random.randint(0, w - s)
        hr = arr[top:top + s, left:left + s]
        hr_t = torch.from_numpy(hr.copy()).permute(2, 0, 1).float() / 255.0
        if random.random() < 0.5:
            hr_t = torch.flip(hr_t, [2])
        if random.random() < 0.5:
            hr_t = torch.flip(hr_t, [1])
        if random.random() < 0.5:
            hr_t = hr_t.transpose(1, 2)
        lr_t = TF.resize(hr_t, [s // self.scale, s // self.scale],
                         interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        lr_up = TF.resize(lr_t, [s, s], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        return lr_up, hr_t


@torch.no_grad()
def eval_one(model, path, device):
    hr = Image.open(path).convert("RGB")
    w, h = hr.size
    w, h = w - w % 2, h - h % 2
    hr = hr.crop((0, 0, w, h))
    lr = hr.resize((w // 2, h // 2), Image.Resampling.BICUBIC)
    bic = lr.resize((w, h), Image.Resampling.BICUBIC)
    lr_t = TF.to_tensor(bic).unsqueeze(0).to(device)
    sr = infer_tiled(model, lr_t)[0]
    sr_np = (sr.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    gt_np = np.asarray(hr).astype("uint8")
    bic_np = np.asarray(bic).astype("uint8")
    psnr = peak_signal_noise_ratio(gt_np, sr_np, data_range=255)
    ssim = structural_similarity(gt_np, sr_np, channel_axis=2, data_range=255)
    mae = float(np.mean(np.abs(gt_np.astype(np.float32) - sr_np.astype(np.float32))))
    bic_psnr = peak_signal_noise_ratio(gt_np, bic_np, data_range=255)
    return psnr, ssim, mae, bic_psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hr_dir", default="data/DIV2K_valid_HR")
    p.add_argument("--train_n", type=int, default=160)
    p.add_argument("--val_n", type=int, default=20)
    p.add_argument("--test_n", type=int, default=31)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--crop_size", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/checkpoints/val_best.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print("Device:", device, flush=True)

    paths = sorted(x for x in Path(args.hr_dir).rglob("*") if x.suffix.lower() in EXTS)
    train_paths = paths[:args.train_n]
    val_paths = paths[args.train_n:args.train_n + args.val_n]
    test_paths = paths[args.train_n + args.val_n:args.train_n + args.val_n + args.test_n]
    print(f"split: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}", flush=True)

    ds = PatchDataset(train_paths, crop_size=args.crop_size, crops_per_img=8)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=(device.type == "cuda"), drop_last=True)
    model = SpatialCVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_psnr = -float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        s_rec = s_kl = 0.0
        for lr_up, hr in loader:
            lr_up, hr = lr_up.to(device), hr.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, q_mu, q_logvar, p_mu, p_logvar = model(lr_up, hr)
                loss, rec, kl = vae_loss(recon, hr, q_mu, q_logvar, p_mu, p_logvar, args.beta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            s_rec += rec.item()
            s_kl += kl.item()
        n = len(loader)
        msg = f"  epoch {epoch}/{args.epochs}: rec={s_rec/n:.6f} kl={s_kl/n:.4f} ({time.time()-t0:.0f}s)"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_m = np.array([eval_one(model, vp, device) for vp in val_paths]).mean(axis=0)
            model.train()
            msg += f" | val_PSNR={val_m[0]:.3f}"
            if val_m[0] > best_val_psnr:
                best_val_psnr = val_m[0]
                save_checkpoint(args.output, model, opt, epoch, val_m[0])
                msg += " (best)"
        print(msg, flush=True)

    # 用最佳验证权重在测试集上评估
    print(f"\nbest val PSNR = {best_val_psnr:.3f}; loading {args.output}", flush=True)
    model.load_state_dict(torch.load(args.output, map_location=device)["model"])
    model.to(device).eval()
    rows = [eval_one(model, tp, device) for tp in tqdm(test_paths, desc="test")]
    m = np.array(rows).mean(axis=0)
    print(f"\n[test set, {len(test_paths)} images] PSNR={m[0]:.3f}  SSIM={m[1]:.4f}  "
          f"MAE={m[2]:.3f}  bicubic_PSNR={m[3]:.3f}", flush=True)


if __name__ == "__main__":
    main()
