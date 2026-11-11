from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DIV2KPatchDataset(Dataset):
    def __init__(self, hr_dir, crop_size=256, scale=2):
        self.hr_dir = Path(hr_dir)
        self.paths = sorted(p for p in self.hr_dir.rglob("*") if p.suffix.lower() in EXTS)
        if not self.paths:
            raise RuntimeError(f"No images found in {self.hr_dir}. Download/unzip DIV2K first.")
        if crop_size % scale:
            raise ValueError("crop_size must be divisible by scale")
        self.crop_size = crop_size
        self.scale = scale

    def __len__(self):
        return len(self.paths) * 8  # random crops/augmentations per source image

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = self.crop_size
        if w < s or h < s:
            ratio = max(s / w, s / h)
            img = img.resize((round(w * ratio), round(h * ratio)), Image.Resampling.BICUBIC)
            w, h = img.size
        left = random.randint(0, w - s)
        top = random.randint(0, h - s)
        hr = img.crop((left, top, left + s, top + s))
        if random.random() < 0.5:
            hr = TF.hflip(hr)
        if random.random() < 0.5:
            hr = TF.vflip(hr)
        if random.random() < 0.5:
            hr = TF.rotate(hr, 90)
        hr_t = TF.to_tensor(hr)
        lr_t = TF.resize(hr_t, [s // self.scale, s // self.scale],
                         interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        lr_up = TF.resize(lr_t, [s, s],
                          interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        return lr_up, hr_t
