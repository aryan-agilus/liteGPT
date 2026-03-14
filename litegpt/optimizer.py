"""
MuonAdamW — hybrid optimizer from nanochat.

Two param groups:
  - 2D weight matrices (attn QKV/out, MLP fc1/fc2):
      Muon = SGD with momentum + Newton-Schulz orthogonalization.
      Orthogonalization aligns gradient updates with the steepest descent
      direction on the Stiefel manifold, improving training stability.

  - Everything else (embeddings, head, norms, biases, scalars):
      Standard AdamW.

Usage:
    muon, adamw = partition_params(model)
    opt = MuonAdamW(muon_params=muon, adamw_params=adamw, lr=0.01, adamw_lr=3e-4)
"""

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize G via 5 steps of Newton-Schulz iteration.
    Returns a matrix with orthonormal rows (or cols if tall).
    Works on float32; input/output cast handled by caller.
    """
    assert G.ndim >= 2, "G must be at least 2-D"
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.float()
    norm = X.norm()
    if norm < 1e-7:
        return G   # near-zero gradient — skip
    X = X / norm

    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# MuonAdamW
# ---------------------------------------------------------------------------

class MuonAdamW(Optimizer):
    """
    Hybrid optimizer:
      - Muon  for param groups flagged use_muon=True  (2D weight matrices)
      - AdamW for param groups flagged use_muon=False (everything else)

    Args:
        muon_params:    iterable of 2-D weight tensors
        adamw_params:   iterable of all other parameters
        lr:             Muon learning rate
        muon_momentum:  Muon momentum coefficient
        adamw_lr:       AdamW learning rate (scaled by model_dim in train.py)
        adamw_betas:    AdamW beta coefficients
        adamw_eps:      AdamW epsilon
        weight_decay:   L2 regularization (applied in AdamW group)
        ns_steps:       Newton-Schulz iteration steps (5 is sufficient)
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        lr: float = 0.01,
        muon_momentum: float = 0.95,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            muon_momentum=muon_momentum,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        param_groups = [
            {"params": list(muon_params),  "use_muon": True},
            {"params": list(adamw_params), "use_muon": False},
        ]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if use_muon:
                    # ---- Muon update ----------------------------------------
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.mul_(group["muon_momentum"]).add_(p.grad)

                    update = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])

                    # Scale update so its RMS matches a normalized gradient
                    scale = max(1, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update * scale, alpha=-group["lr"])

                else:
                    # ---- AdamW update ----------------------------------------
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    t = state["step"]
                    beta1, beta2 = group["adamw_betas"]
                    eps = group["adamw_eps"]

                    state["exp_avg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        p.grad, p.grad, value=1 - beta2
                    )

                    # Bias-corrected step size
                    step_size = group["adamw_lr"] / (1 - beta1 ** t)
                    denom = (state["exp_avg_sq"].sqrt() / (1 - beta2 ** t) ** 0.5).add_(eps)

                    # Decoupled weight decay
                    p.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Helper: partition model parameters into Muon vs AdamW groups
# ---------------------------------------------------------------------------

def partition_params(model: torch.nn.Module):
    """
    Returns (muon_params, adamw_params).
    Muon: 2-D weight matrices (attn, mlp weights).
    AdamW: embeddings, norms, biases, scalars, output head.
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embeddings and the LM head have ndim==2 but should use AdamW
        # (they live in a different optimization landscape)
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
