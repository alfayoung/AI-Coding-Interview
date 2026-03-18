"""Reference implementation of Flow Matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingRef(nn.Module):
    def __init__(self, velocity_model: nn.Module):
        super().__init__()
        self.velocity_model = velocity_model

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - t) * x0 + t * x1

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = torch.randn_like(x0)
        t = torch.rand(x0.shape[0], device=x0.device)
        x_t = self.interpolate(x0, x1, t)
        v_pred = self.velocity_model(x_t, t)
        target = x1 - x0
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device, n_steps: int = 100) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), (i + 1) * dt, device=device)
            v = self.velocity_model(x, t)
            x = x - v * dt
        return x
