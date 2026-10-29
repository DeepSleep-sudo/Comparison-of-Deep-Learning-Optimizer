"""
================================================================================
muon.py  --  Muon (Newton-Schulz orthogonalized momentum)
================================================================================
Operates on 2-D hidden-layer weights and orthogonalises the momentum buffer with
a Newton-Schulz iteration.
================================================================================
"""

import torch
from torch.optim.optimizer import Optimizer


# ------------------------------------------------------------------------------
# Muon  --  Jordan (2024)
# ------------------------------------------------------------------------------
class Muon(Optimizer):
    r"""Muon: MomentUm Orthogonalized by Newton-Schulz.

    Reference: Jordan, K. (2024), "Muon: An optimizer for hidden layers in
    neural networks". Blog write-up: https://kellerjordan.github.io/posts/muon/
    (also see the "Muon is Scalable for LLM Training" follow-up, 2025).

    Idea
    ----
    For a 2-D weight W with gradient G:
        m_t = mu * m_{t-1} + g_t                     (heavy-ball momentum)
        O_t = NewtonSchulz5(m_t)                     (approximate orthogonalisation)
        W_t = W_{t-1} - lr * O_t                     (lr scaled by sqrt(fan_out/fan_in))

    The orthogonalisation maps the momentum matrix to the nearest (scaled)
    orthogonal matrix -- i.e. it *equalises all singular values*. This is a
    cheap approximation to the "spectral" update direction and is closely
    related to a steepest-descent step under the spectral norm.

    Newton-Schulz iteration (5 steps, quintic convergence):
        X_0 = M / ||M||_F
        X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
    with the coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) chosen (in the
    original code) to push the singular values towards 1.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
        """Approximate the orthogonal polar factor of G (i.e. U V^T) with the
        quintic Newton-Schulz iteration. Coefficients from Jordan's Muon code.
        Tensors with ndim > 2 are flattened to 2-D first (as in the reference)."""
        a, b, c = 3.4445, -4.7750, 2.0315
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)             # e.g. conv weight -> matrix
        X = G / (G.norm() + eps)                      # X_0 = M / ||M||_F
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T                               # X X^T
            B = b * A + c * (A @ A)                   # b A + c A^2
            X = a * X + B @ X                         # X_{k+1} = a X + (bA + cA^2) X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() < 2:
                    # 1-D params (biases, norms) fall back to plain momentum SGD.
                    # The Muon write-up explicitly recommends AdamW for these.
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                # momentum EMA:  m_t = mu * m_{t-1} + (1 - mu) * g_t
                buf.lerp_(g, 1 - mu)
                if group["nesterov"]:
                    # Nesterov look-ahead on the momentum
                    m = g.lerp(buf, mu)
                else:
                    m = buf
                O = self._zeropower_via_newtonschulz5(m, steps=ns)
                # lr scaling: keep the update RMS ~ constant across shapes.
                # Reference (Jordan's modded-nanogpt): max(1, fan_out/fan_in)**0.5
                scale = max(1.0, O.size(-2) / O.size(-1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(O.reshape(p.shape), alpha=-lr * scale)
        return loss


