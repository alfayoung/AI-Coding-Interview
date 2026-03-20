"""Swiss Roll flow matching training script.

Trains a velocity network via flow matching on 2D Swiss Roll data,
then generates samples via Euler ODE integration and plots the result.
"""

import torch
from solution import FlowMatching
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll


class Embedding:
    def __init__(self, dim1: int, dim2: int):
        Q = torch.randn(dim1, dim2)
        self.Q, _ = torch.linalg.qr(Q)
    
    def proj(self, x):
        return torch.matmul(x, self.Q)
    
    def unproj(self, x):
        return torch.matmul(x, self.Q.T)

# -- Data --
def sample_swiss_roll(n: int) -> torch.Tensor:
    """Sample 2D Swiss Roll points, normalised to roughly unit variance."""
    data, _ = make_swiss_roll(n, noise=0.5)
    data = data[:, [0, 2]]  # keep the 2D projection
    data = torch.tensor(data, dtype=torch.float32)
    data = (data - data.mean(0)) / data.std(0)
    return data

# -- Main --
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowMatching(32).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    embedder = Embedding(2, 32)

    n_epochs = 200
    batch_size = 64

    for epoch in range(n_epochs):
        x1 = sample_swiss_roll(batch_size).to(device)
        x1 = embedder.proj(x1)

        loss = model(x1)
        
        loss.backward()
        optim.step()
        optim.zero_grad()

        if epoch % 20 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    # -- Visualise --
    real = sample_swiss_roll(2000).numpy()
    gen = model.sample(2000, device=device).cpu()
    gen = embedder.unproj(gen).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.5)
    axes[0].set_title("Swiss Roll (real)")
    axes[1].scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.5)
    axes[1].set_title("Flow Matching (generated)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
