"""
================================================================================
modern.py  --  Modern optimizers (2019-2024)
================================================================================
    LAMB          (You et al., 2020)        -- layer-wise trust ratio
    Lion          (Chen et al., 2023)       -- sign of an interpolated momentum
    Schedule-Free (Defazio et al., 2024)    -- no learning-rate schedule
================================================================================
"""

import math
import torch
from torch.optim.optimizer import Optimizer


# ------------------------------------------------------------------------------
# LAMB  --  You, Li, Reddi, Hse, Kumar, Bhojanapalli, Song, Demmel, Hsieh (2020)
# ------------------------------------------------------------------------------
class LAMB(Optimizer):
    r"""LAMB: Layer-wise Adaptive Moments optimizer for Batch training.

    Reference: You, Li, Reddi, Hse, Kumar, Bhojanapalli, Song, Demmel, Hsieh
    (2020), "Large Batch Optimization for Deep Learning: Training BERT in 76
    minutes", ICLR. See Algorithm 1 (LAMB) and eqs. (1)-(3).

    Update:
        m_t = beta1*m_{t-1} + (1-beta1) g_t
        v_t = beta2*v_{t-1} + (1-beta2) g_t^2
        r_t = m_hat_t / (sqrt(v_hat_t) + eps) + wd * theta_{t-1}     (AdamW-like direction)
        # layer-wise trust ratio (eq. 2):
        phi(||theta||) = ||theta||          (identity norm, as used in the paper)
        trust_ratio = phi(||theta_{t-1}||) / phi(||r_t||)
        theta_t = theta_{t-1} - eta * trust_ratio * r_t
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, max_trust_ratio=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        max_trust_ratio=max_trust_ratio)
        super().__init__(params, defaults)

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                # AdamW-style direction r_t (with decoupled decay)
                update = (m / bc1) / ((v / bc2).sqrt() + eps) + wd * p
                # layer-wise trust ratio = ||theta|| / ||update||
                w_norm = p.norm()
                u_norm = update.norm()
                if w_norm > 0 and u_norm > 0:
                    trust = (w_norm / u_norm).clamp(max=group["max_trust_ratio"]).item()
                else:
                    trust = 1.0
                p.add_(update, alpha=-lr * trust)
        return loss


class Lion(Optimizer):
    r"""Lion: EvoLved Sign Momentum.

    Reference: Chen, Liang, Huang, Real, Wang, Pham, Dong, Luong, Hsieh, Lu,
    Le (2023), "Symbolic Discovery of Optimization Algorithms", NeurIPS.
    See Algorithm 1 (Lion) and eq. (2).

    Update (Lion, Algorithm 1):
        c_t   = beta1 * m_{t-1} + (1-beta1) * g_t        (interpolation)
        theta_t = theta_{t-1} - eta * ( sign(c_t) + wd * theta_{t-1} )   (eq. 2)
        m_t   = beta2 * m_{t-1} + (1-beta2) * g_t        (update momentum AFTER)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                # update direction: sign(beta1*m + (1-beta1)*g)
                update = m.mul(b1).add_(g, alpha=1 - b1).sign()
                # decoupled weight decay inside the same step
                update.add_(p, alpha=wd)
                p.add_(update, alpha=-lr)
                # store the momentum with beta2 (note: AFTER the update, per Alg. 1)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


class ScheduleFreeAdamW(Optimizer):
    r"""Schedule-Free AdamW.

    Reference: Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky (2024),
    "The Road Less Scheduled", NeurIPS (Oral). See Algorithm 1 and Section 3.

    Idea
    ----
    Standard practice anneals lr -> 0 and averages weights at the end. The
    Schedule-Free method maintains TWO sequences -- the gradient-evaluation
    point y_t and the averaging point x_t -- and produces a final point z_t:

        y_t = (1-beta) z_t + beta x_t          (interpolation, "gradient point")
        g_t = grad at y_t
        (Adam-style) z_{t+1} = z_t - eta * m_hat / (sqrt(v_hat) + eps)
        x_{t+1} = (1 - 1/(t+1)) x_t + 1/(t+1) z_{t+1}    (running average)
        # final output is x (the averaged iterate); no lr schedule needed.
    """

    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, beta=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, beta=beta)
        super().__init__(params, defaults)
        self._training = True

    def train(self):
        """Switch to 'train' mode: parameters hold the y-sequence."""
        self._training = True

    def eval(self):
        """Switch to 'eval' mode: parameters hold the x-sequence (averaged)."""
        self._training = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps, wd, beta = group["eps"], group["weight_decay"], group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.detach().clone()      # z_0 = theta_0
                    state["x"] = p.detach().clone()      # x_0 = theta_0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                z, x = state["z"], state["x"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                # AdamW-style update on z
                z.mul_(1 - lr * wd)
                z.addcdiv_(m / bc1, (v / bc2).sqrt().add_(eps), value=-lr)
                # running average x_{t+1} = (1-1/(t+1)) x_t + 1/(t+1) z_{t+1}
                weight = 1.0 / (t + 1)
                x.lerp_(z, weight)
                # parameter exposed for the *next* gradient = (1-beta) z + beta x
                p.copy_(z.lerp(x, beta))
        return loss
