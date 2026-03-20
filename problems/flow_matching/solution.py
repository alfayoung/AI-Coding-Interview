"""Flow Matching

Implement the core flow matching components:
1. Interpolation: x_t = (1 - t) * x_0 + t * x_1  (where x_1 ~ N(0, I))
2. Velocity target: u_t = x_1 - x_0
3. Training loss: MSE between predicted velocity and target velocity
4. ODE sampling: integrate dx/dt = v(x_t, t) from t=0 to t=1

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 256):
        super().__init__()
        # Time embedding: scalar t → hidden-dim vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 20),
            nn.GELU()
        )

        self.net = nn.Sequential(
            nn.Linear(dim + 20, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time conditioning: broadcast-add to every encoder layer
        t_emb = self.time_mlp(t)  # [B, hidden]
        res = self.net(torch.cat([x, t_emb], dim=-1))
        return res


class FlowMatching(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.model = MLP(dim=dim, hidden=hidden)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute x_t along the linear path from x0 to x1."""
        return x0 * (1 - t) + x1 * t

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss: sample noise x0, time t, predict velocity."""
        B = x1.shape[0]
        t = torch.rand(B, 1, device=x1.device)
        x0 = torch.randn_like(x1)

        xt = self.interpolate(x0, x1, t)
        v_pred = self.model(xt, t)
        v_gt = x1 - x0

        return F.mse_loss(v_pred, v_gt)

    @torch.no_grad()
    def sample(self, n: int, dim: int = 2, device: str = "cpu", n_steps: int = 100) -> torch.Tensor:
        """Generate samples via Euler ODE integration from t=0 to t=1."""
        x = torch.randn(n, dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((n, 1), i * dt, device=device)
            x = x + self.model(x, t) * dt
        return x
