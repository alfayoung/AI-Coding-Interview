"""Reference implementation of DDPM core math."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMRef(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps

        beta = torch.linspace(beta_min, beta_max, n_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.eps_model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps_pred = self.eps_model(x, t_batch)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            mean = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred)

            if t > 0:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean
        return x
