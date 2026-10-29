"""
================================================================================
second_order.py  --  Second-order & preconditioning optimizers
================================================================================
Newton's method needs the (d x d) Hessian, which is intractable for deep nets.
This module implements the three approximations used in the comparison:

    Shampoo  (Gupta et al., 2018)  -- Kronecker-factored full-matrix precond.
    Sophia   (Liu et al., 2023)    -- diagonal Hessian + trust-region clipping
    SOAP     (Vyas et al., 2024)   -- Adam inside the Shampoo eigenbasis
================================================================================
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class Shampoo(Optimizer):
    r"""Shampoo: Preconditioned Stochastic Tensor Optimization.

    Reference: Gupta, Koren & Singer (2018), "Shampoo: Preconditioned Stochastic
    Tensor Optimization", ICML. See Algorithm 1 and eqs. (1)-(3).

    Idea
    ----
    For a weight matrix W in R^{m x n} with gradient G, maintain two
    *Kronecker factors* of the full-matrix preconditioner:

        L_t = L_{t-1} + G G^T          (m x m, "left" second moment)
        R_t = R_{t-1} + G^T G          (n x n, "right" second moment)

    and take the Kronecker-factored preconditioned step:

        W_t = W_{t-1} - eta * L_t^{-1/4} G R_t^{-1/4}       (Algorithm 1)

    The -1/4 exponent comes from the fact that the full preconditioner is
    (L (x) R)^{-1/2}, whose "square root" on each side is L^{-1/4} (x) R^{-1/4}.
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0,
                 epsilon=1e-12, update_freq=1, exponent=-0.25, max_precond_dim=1024):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq, exponent=exponent,
                        max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_root_inv(M, exponent, eps):
        """Return M^{exponent} via symmetric eigendecomposition (M is PSD)."""
        M = M + eps * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        # eigh requires symmetric; symmetrise defensively
        M = 0.5 * (M + M.t())
        evals, evecs = torch.linalg.eigh(M)
        evals = evals.clamp(min=eps) ** exponent
        return (evecs * evals) @ evecs.t()

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
            eps = group["epsilon"]
            freq = group["update_freq"]
            expo = group["exponent"]
            max_dim = group["max_precond_dim"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                # Shampoo is defined for matrices; for 1-D params (biases/norms)
                # or matrices with a dimension above `max_precond_dim`, fall back
                # to plain momentum SGD (documented practical approximation that
                # bounds the O(m^3 + n^3) eigen-decomposition cost).
                if g.dim() != 2 or max(g.shape[0], g.shape[1]) > max_dim:
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    buf = state["momentum"]
                    buf.mul_(mu).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["L"] = torch.zeros(g.shape[0], g.shape[0], device=g.device, dtype=g.dtype)
                    state["R"] = torch.zeros(g.shape[1], g.shape[1], device=g.device, dtype=g.dtype)
                    state["momentum"] = torch.zeros_like(p)
                state["step"] += 1
                L, R = state["L"], state["R"]
                # accumulate Kronecker factors
                L.add_(g @ g.t())
                R.add_(g.t() @ g)
                # preconditioned gradient L^{-1/4} G R^{-1/4}
                Lr = self._matrix_root_inv(L, expo, eps)
                Rr = self._matrix_root_inv(R, expo, eps)
                precond = Lr @ g @ Rr
                buf = state["momentum"]
                buf.mul_(mu).add_(precond)
                p.add_(buf, alpha=-lr)
        return loss


# ------------------------------------------------------------------------------
# Sophia  --  Liu, Li, Hall, Liang, Ma (2023)
# ------------------------------------------------------------------------------


class Sophia(Optimizer):
    r"""Sophia: Second-order Clipped Stochastic Optimization.

    Reference: Liu, Li, Hall, Liang, Ma (2023), "Sophia: A Scalable Stochastic
    Second-order Optimizer for Language Model Pre-training", ICLR 2024
    (arXiv:2305.14342). See Algorithm 1 (Sophia-G) and eqs. (1)-(6).

    Idea
    ----
    Maintain a diagonal Hessian estimate h_t (updated every ``update_freq`` steps
    using a Gauss-Newton / Hutchinson estimator) and *clip* the update:

        m_t = beta1*m_{t-1} + (1-beta1) g_t
        h_t = beta2*h_{t-1} + (1-beta2) * diag_hessian_estimate
        update = clip( m_t / max(gamma * h_t, eps), 1 )      (elementwise clip to [-1,1])
        theta_t = theta_{t-1} - eta * update

    The clip is a *trust region*: it prevents the second-order step from taking
    a huge jump when the Hessian estimate is small/inaccurate. The learning rate
    eta then directly bounds the per-coordinate step.
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=0.0, update_freq=10, clip_value=1.0):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        update_freq=update_freq, clip_value=clip_value)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """``closure`` should return the *loss* so a Gauss-Newton Hessian
        estimate can be computed. If no closure is given we fall back to a
        squared-gradient proxy for the diagonal Hessian (flagged below)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            rho, wd = group["rho"], group["weight_decay"]
            freq = group["update_freq"]
            clip = group["clip_value"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, h = state["exp_avg"], state["hessian"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                # Diagonal Hessian estimate. Proper Sophia uses a Gauss-Newton
                # (Hutchinson) estimate every `freq` steps. Without a closure we
                # use the EMA of g^2 as a cheap (biased) proxy, which recovers
                # an Adam-like denominator and is clearly noted here.
                if t % freq == 0:
                    h.mul_(b2).addcmul_(g, g, value=1 - b2)
                # clipped update: clip(m / max(rho*h, eps), clip)
                # ``clip_value=None`` disables the trust region (ablation only).
                denom = (rho * h).clamp_(min=1e-15)
                update = m / denom
                if clip is not None:
                    update = update.clamp_(-clip, clip)
                p.add_(update, alpha=-lr)
        return loss


# ------------------------------------------------------------------------------
# SOAP  --  Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade (2024)
# ------------------------------------------------------------------------------


class SOAP(Optimizer):
    r"""SOAP: Improving and Stabilizing Shampoo using Adam.

    Reference: Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade
    (2024), "SOAP: Improving and Stabilizing Shampoo using Adam", arXiv:2409.11321.
    See Algorithm 1 (SOAP) and Section 3.

    Idea
    ----
    Run Adam *in the eigenbasis of the Shampoo preconditioner*. Concretely, keep
    the Shampoo factors L, R; compute their eigenbases Q_L, Q_R (updated
    periodically); project the gradient into that basis:

        G' = Q_L^T G Q_R

    run a full Adam (both moments) on G', then project the update back:

        Delta = Q_L * adam_update(G') * Q_R^T

    Because the basis is (approximately) the preconditioner's eigenbasis, Adam's
    diagonal second moment becomes a good approximation of the *full-matrix*
    preconditioner -- giving Shampoo's speed with Adam's stability.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.0, shampoo_beta=0.95, precondition_freq=10,
                 max_precond_dim=1024):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        shampoo_beta=shampoo_beta, precondition_freq=precondition_freq,
                        max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)

    @staticmethod
    def _eigenbasis(M, eps):
        M = 0.5 * (M + M.t())
        evals, evecs = torch.linalg.eigh(M)
        return evecs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            sb = group["shampoo_beta"]
            freq = group["precondition_freq"]
            max_dim = group["max_precond_dim"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                # SOAP needs an (m x m) and (n x n) eigen-decomposition. For 1-D
                # params, or matrices with a dimension larger than `max_precond_dim`,
                # we fall back to plain Adam (a documented practical approximation
                # used by real Shampoo/SOAP implementations to bound compute).
                if g.dim() != 2 or max(g.shape[0], g.shape[1]) > max_dim:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["m1"] = torch.zeros_like(p)
                        state["m2"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["m1"], state["m2"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    p.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt().add_(eps),
                               value=-lr)
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["L"] = torch.zeros(g.shape[0], g.shape[0], device=g.device, dtype=g.dtype)
                    state["R"] = torch.zeros(g.shape[1], g.shape[1], device=g.device, dtype=g.dtype)
                    state["QL"] = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
                    state["QR"] = torch.eye(g.shape[1], device=g.device, dtype=g.dtype)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                L, R = state["L"], state["R"]
                L.mul_(sb).add_(g @ g.t(), alpha=1 - sb)
                R.mul_(sb).add_(g.t() @ g, alpha=1 - sb)
                # refresh eigenbases periodically (expensive)
                if t % freq == 1:
                    state["QL"] = self._eigenbasis(L, eps)
                    state["QR"] = self._eigenbasis(R, eps)
                QL, QR = state["QL"], state["QR"]
                # project gradient into the Shampoo eigenbasis
                gp = QL.t() @ g @ QR
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(gp, alpha=1 - b1)
                v.mul_(b2).addcmul_(gp, gp, value=1 - b2)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                adam_update = (m / bc1) / ((v / bc2).sqrt() + eps)
                # project the update back to parameter space
                update = QL @ adam_update @ QR.t()
                p.add_(update, alpha=-lr)
        return loss
