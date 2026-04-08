# Coding Interview Practice — PyTorch From-Scratch Implementations

A collection of **coding interview problems** focused on implementing core deep learning algorithms from scratch in PyTorch. Each problem provides a skeleton (`solution.py`) and a reference implementation (`reference.py`) so the practitioner can attempt the problem, then verify correctness.

## Problem Format

Each problem lives in its own directory under `problems/` with its own `pyproject.toml` for dependencies. The core files are:

| File | Purpose |
|---|---|
| `solution.py` | Skeleton with class/method signatures, docstrings, and `raise NotImplementedError`. The practitioner implements the logic here. |
| `reference.py` | Complete, readable reference implementation used to generate expected outputs. |
| `pyproject.toml` | Per-problem dependencies (the root `pyproject.toml` is the superset of all). |

Depending on the problem, one or both of the following may also be present:

| File | Purpose |
|---|---|
| `test.py` | Checks output match between `solution.py` and `reference.py` (uses `utils.check.check_close` or similar assertions). |
| `train.py` | Trains `solution.py` end-to-end to verify that the implementation converges. |

## Topics Covered

| Category | Problems | Status |
|---|---|---|
| **Attention** | Cross-attention, Causal self-attention, Transformer block, Flash attention | 3/4 done |
| **Math** | Gradient descent (linear regression), Softmax gradient, LayerNorm, BatchNorm | 0/4 |
| **Diffusion Models** | DDPM, Flow Matching | Scaffolded |
| **Reinforcement Learning** | PPO, SAC, TD3 | 0/3 |

See [problem_list.md](problem_list.md) for the live checklist.

## Setup

Requires Python >= 3.12. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync            # install all dependencies from the root pyproject.toml
```

### Key dependencies (root)

- `torch >= 2.10.0`
- `pytest`
- `numpy`, `matplotlib`, `scikit-learn`

Each problem also declares its own minimal dependencies in its `pyproject.toml`.

## Usage

### Solve a problem

1. Open `problems/<topic>/solution.py`.
2. Replace the `NotImplementedError` stubs with your implementation.
3. Verify via the provided `test.py` or `train.py`.

### Run tests

```bash
# Run all tests (pytest discovers test_*.py / test.py inside problems/)
uv run pytest

# Run a specific problem's tests
uv run pytest problems/ddpm/test.py -v
```

### Run training scripts

```bash
# Transformer block
cd problems/transformer_block && uv run python train.py

# Flow matching (trains on Swiss Roll, shows matplotlib plot)
cd problems/flow_matching && uv run python train.py
```

## Adding a New Problem

1. Create a directory under `problems/` (e.g., `problems/layer_norm/`).
2. Add `__init__.py`, `solution.py` (skeleton), `reference.py` (gold implementation), and `pyproject.toml` (problem-specific dependencies).
3. Add `test.py` (output comparison against reference) and/or `train.py` (training convergence check), as appropriate for the problem.
4. Update `problem_list.md`.
