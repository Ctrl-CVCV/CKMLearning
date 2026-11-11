import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DIV2KPatchDataset
from src.model import SpatialCVAE, vae_loss
from src.utils import seed_everything, get_device, save_checkpoint, plot_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/DIV2K_train_HR")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=64,
                   help="channels of the spatial latent map (z has shape latent_dim x H/8 x W/8)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)
    ds = DIV2KPatchDataset(args.data_dir, args.crop_size, scale=2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)
    model = SpatialCVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"
    best = float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "reconstruction_mse", "kl"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = torch.zeros(3, device=device)
            bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
            for lr_up, hr in bar:
                lr_up, hr = lr_up.to(device), hr.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, q_mu, q_logvar, p_mu, p_logvar = model(lr_up, hr)
                loss, rec, kl = vae_loss(recon, hr, q_mu, q_logvar, p_mu, p_logvar, args.beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sums += torch.tensor([loss.item(), rec.item(), kl.item()], device=device)
                bar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec.item():.4f}")
            means = (sums / len(loader)).cpu().tolist()
            writer.writerow([epoch, *means])
            f.flush()
            print(f"Epoch {epoch}: loss={means[0]:.6f}, rec={means[1]:.6f}, kl={means[2]:.6f}")
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, means[0])
            if means[0] < best:
                best = means[0]
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, means[0])
    plot_losses(log_path, out / "loss_curve.png")
    print(f"Training finished. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
