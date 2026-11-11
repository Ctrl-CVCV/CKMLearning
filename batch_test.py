import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from src.utils import get_device, load_model, tensor_to_pil, infer_tiled


def process_one(model, device, img_path, out_dir):
    stem = img_path.stem
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    w2, h2 = w - (w % 2), h - (h % 2)
    img = img.crop((0, 0, w2, h2))

    lr = img.resize((w2 // 2, h2 // 2), Image.BICUBIC)
    lr_up = lr.resize((w2, h2), Image.BICUBIC)

    lr_t = TF.to_tensor(lr_up).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = infer_tiled(model, lr_t)[0]
    sr = tensor_to_pil(sr_t)

    lr.save(out_dir / f"{stem}_lr.png")
    lr_up.save(out_dir / f"{stem}_bicubic.png")
    sr.save(out_dir / f"{stem}_vae.png")
    print(f"processed: {img_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--input_dir", required=True, help="存放测试图片的文件夹")
    p.add_argument("--output_dir", default="outputs/batch_test")
    args = p.parse_args()

    device = get_device()
    model = load_model(args.checkpoint, device)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(in_dir.glob(ext))

    if len(files) == 0:
        print("输入文件夹没有找到图片！")
        return

    for f in files:
        process_one(model, device, f, out_dir)
    print(f"\n全部处理完成，共 {len(files)} 张，输出：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
