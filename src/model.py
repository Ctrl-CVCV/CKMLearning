import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class SpatialCVAE(nn.Module):
    """Spatial-latent conditional VAE for x2 super-resolution.

    The latent is a *spatial map* z in R^{latent_dim x H/8 x W/8}, not a single
    global vector. This is the key difference from a naive VAE: a global
    (e.g. 128-dim) latent cannot carry the high-frequency detail needed for SR,
    while a spatial latent can.

      prior       p(z|c)    : encoder on the bicubic condition c alone -> (mu_p, logvar_p)
      posterior   q(z|c,HR) : encoder on [c, HR]                       -> (mu_q, logvar_q)

    Training:   L = MSE(recon, HR) + beta * KL(q(z|c,HR) || p(z|c))
    Inference:  z = mu_p(c)  (deterministic, best PSNR), or
                z ~ p(z|c)   (generative, one LR -> many plausible HRs).
    """
    def __init__(self, latent_dim=64, base_channels=64):
        super().__init__()
        c = base_channels
        # prior encoder (condition only, 3ch) -> H/8
        self.p_enc = nn.Sequential(
            conv_block(3, c, stride=2),
            conv_block(c, c * 2, stride=2),
            conv_block(c * 2, c * 4, stride=2),
            conv_block(c * 4, c * 4, stride=1),
        )
        self.p_mu = nn.Conv2d(c * 4, latent_dim, 3, padding=1)
        self.p_logvar = nn.Conv2d(c * 4, latent_dim, 3, padding=1)
        # posterior encoder ([condition, HR], 6ch) -> H/8
        self.q_enc = nn.Sequential(
            conv_block(6, c, stride=2),
            conv_block(c, c * 2, stride=2),
            conv_block(c * 2, c * 4, stride=2),
            conv_block(c * 4, c * 4, stride=1),
        )
        self.q_mu = nn.Conv2d(c * 4, latent_dim, 3, padding=1)
        self.q_logvar = nn.Conv2d(c * 4, latent_dim, 3, padding=1)
        # decoder: upsample z to full res, concat with condition, refine
        self.latent_dim = latent_dim
        self.dec_in = nn.Conv2d(3 + latent_dim, c * 4, 3, padding=1)
        self.dec1 = conv_block(c * 4, c * 4)
        self.dec2 = conv_block(c * 4, c * 2)
        self.dec3 = conv_block(c * 2, c)
        self.dec4 = conv_block(c, c)
        self.out = nn.Conv2d(c, 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def encode_prior(self, condition):
        x = self.p_enc(condition)
        return self.p_mu(x), self.p_logvar(x).clamp(-12, 12)

    def encode_posterior(self, condition, target):
        x = self.q_enc(torch.cat([condition, target], dim=1))
        return self.q_mu(x), self.q_logvar(x).clamp(-12, 12)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, condition, z):
        z_up = F.interpolate(z, size=condition.shape[-2:], mode="bilinear", align_corners=False)
        x = self.act(self.dec_in(torch.cat([condition, z_up], dim=1)))
        x = self.dec4(self.dec3(self.dec2(self.dec1(x))))
        return (condition + self.out(x)).clamp(0.0, 1.0)

    def forward(self, lr_up, target):
        p_mu, p_logvar = self.encode_prior(lr_up)
        q_mu, q_logvar = self.encode_posterior(lr_up, target)
        z = self.reparameterize(q_mu, q_logvar)
        recon = self.decode(lr_up, z)
        return recon, q_mu, q_logvar, p_mu, p_logvar

    @torch.no_grad()
    def infer(self, lr_up, sample=False):
        """Deterministic (prior mean) by default; sample=True for generative SR."""
        p_mu, p_logvar = self.encode_prior(lr_up)
        z = self.reparameterize(p_mu, p_logvar) if sample else p_mu
        return self.decode(lr_up, z)


def kl_gauss(mu1, logvar1, mu2, logvar2):
    """KL( N(mu1, exp(logvar1)) || N(mu2, exp(logvar2)) ), mean over all elements."""
    return 0.5 * torch.mean(logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1)


def vae_loss(recon, target, q_mu, q_logvar, p_mu, p_logvar, beta=1e-4):
    # MSE reconstruction: PSNR is a monotonic transform of MSE, so this optimizes PSNR.
    rec = F.mse_loss(recon, target)
    # KL against the *conditional* prior p(z|c), not N(0, I).
    kl = kl_gauss(q_mu, q_logvar, p_mu, p_logvar)
    total = rec + beta * kl
    return total, rec.detach(), kl.detach()
