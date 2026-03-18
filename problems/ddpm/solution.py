"""DDPM (Denoising Diffusion Probabilistic Models)

Implement the core DDPM components:
1. Forward diffusion process (q_sample): add noise at timestep t
2. Reverse process loss: predict noise and compute MSE loss
3. Sampling loop: iterative denoising from pure noise

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
"""

import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        """
        Args:
            eps_model: neural network that predicts noise eps(x_t, t) -> eps
            n_steps: number of diffusion steps
            beta_min: starting noise schedule value
            beta_max: ending noise schedule value
        """
        super().__init__()
        raise NotImplementedError("Implement this!")

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: sample x_t given x_0 and timestep t."""
        raise NotImplementedError("Implement this!")

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute training loss: sample t, add noise, predict noise, return MSE."""
        raise NotImplementedError("Implement this!")

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate samples by iterative denoising from x_T ~ N(0, I)."""
        raise NotImplementedError("Implement this!")
