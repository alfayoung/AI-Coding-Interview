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


class FlowMatching(nn.Module):
    def __init__(self, velocity_model: nn.Module):
        """
        Args:
            velocity_model: neural network v(x_t, t) -> velocity
        """
        super().__init__()
        raise NotImplementedError("Implement this!")

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute x_t along the linear path from x0 to x1."""
        raise NotImplementedError("Implement this!")

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss: sample noise x1, time t, predict velocity."""
        raise NotImplementedError("Implement this!")

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device, n_steps: int = 100) -> torch.Tensor:
        """Generate samples via Euler ODE integration from t=1 to t=0."""
        raise NotImplementedError("Implement this!")
