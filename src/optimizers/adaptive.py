"""
================================================================================
adaptive.py  --  Adaptive (per-coordinate) learning-rate optimizers
================================================================================
Each optimizer rescales the step of every coordinate by a running estimate of
that coordinate's gradient statistics.

    AdaGrad  (Duchi et al., 2011)
    RMSProp  (Tieleman & Hinton, 2012)
    Adam     (Kingma & Ba, 2015)
    AdamW    (Loshchilov & Hutter, 2019)

Notation: g_t = gradient at step t; theta = parameters; eta = learning rate.
Weight decay is decoupled where the source paper defines it (AdamW) and L2-coupled
elsewhere, matching the original papers.
================================================================================
"""

import math
import torch
from torch.optim.optimizer import Optimizer


# ------------------------------------------------------------------------------
# AdaGrad  --  Duchi, Hazan & Singer (2011)
# ------------------------------------------------------------------------------
class AdaGrad(Optimizer):
    r"""Adaptive Gradient (AdaGrad).

    Reference: Duchi, Hazan & Singer (2011), "Adaptive Subgradient Methods for
    Online Learning and Stochastic Optimization", JMLR 12:2121-2159.
    See Algorithm 1 (the "full-matrix" / diagonal version).

    Update (diagonal AdaGrad):
        G_t = G_{t-1} + g_t^2                          (accumulated squared grad)
        theta_t = theta_{t-1} - eta / (sqrt(G_t) + eps) * g_t
    """

    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["sum_sq"] = torch.zeros_like(p)   # G_0 = 0
                G = state["sum_sq"]
                G.addcmul_(g, g)                            # G_t = G_{t-1} + g_t^2
                denom = G.sqrt().add_(eps)                  # sqrt(G_t) + eps
                p.addcdiv_(g, denom, value=-lr)             # theta -= eta * g/denom
        return loss


# ------------------------------------------------------------------------------
# RMSProp  --  Tieleman & Hinton (2012)
# ------------------------------------------------------------------------------


class RMSProp(Optimizer):
    r"""Root Mean Square Propagation.

    Reference: Tieleman & Hinton (2012), "Lecture 6.5 - RMSProp", COURSERA:
    Neural Networks for Machine Learning. (Unpublished lecture notes; the
    algorithm is due to Hinton.) See slide "RMSProp" in the lecture.

    Update:
        v_t = beta * v_{t-1} + (1 - beta) * g_t^2      (EMA of squared grads)
        theta_t = theta_{t-1} - eta / (sqrt(v_t) + eps) * g_t
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, alpha, eps = group["lr"], group["alpha"], group["eps"]
            wd, mu = group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)
                    if mu > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                v = state["square_avg"]
                # v_t = alpha*v_{t-1} + (1-alpha)*g^2
                v.mul_(alpha).addcmul_(g, g, value=1.0 - alpha)
                denom = v.sqrt().add_(eps)               # sqrt(v_t) + eps
                if mu > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).addcdiv_(g, denom)      # m_t = mu*m + g/denom
                    p.add_(buf, alpha=-lr)
                else:
                    p.addcdiv_(g, denom, value=-lr)
        return loss


class Adam(Optimizer):
    r"""Adaptive Moment Estimation (Adam).

    Reference: Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization",
    ICLR. See Algorithm 1.

    Update (Kingma & Ba Algorithm 1):
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t           (1st moment, eq. 1)
        v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2         (2nd moment, eq. 2)
        m_hat = m_t / (1 - beta1^t)                       (bias correction, eq. 3)
        v_hat = v_t / (1 - beta2^t)                       (bias correction)
        theta_t = theta_{t-1} - eta * m_hat / (sqrt(v_hat) + eps)   (eq. 4-5)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                 bias_correction=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        bias_correction=bias_correction)
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
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)       # m
                    state["exp_avg_sq"] = torch.zeros_like(p)    # v
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                # m_t, v_t
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                # bias-corrected step size (``bias_correction=False`` = ablation)
                if group["bias_correction"]:
                    bc1 = 1 - b1 ** t
                    bc2 = 1 - b2 ** t
                else:
                    bc1 = bc2 = 1.0
                if group["amsgrad"]:
                    vmax = state["max_exp_avg_sq"]
                    torch.maximum(vmax, v, out=vmax)             # AMSGrad: v_hat = max
                    denom = vmax.sqrt().add_(eps)
                else:
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                step_size = lr / bc1
                p.addcdiv_(m, denom, value=-step_size)
        return loss


class AdamW(Optimizer):
    r"""Adam with decoupled weight decay (AdamW).

    Reference: Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization",
    ICLR. See Algorithm 2 (AdamW) and Section 2 (why L2 != weight decay under
    adaptive lr).

    Update (Algorithm 2):
        m_t = beta1*m_{t-1} + (1-beta1) g_t
        v_t = beta2*v_{t-1} + (1-beta2) g_t^2
        theta_t = theta_{t-1} - eta * ( m_hat_t/(sqrt(v_hat_t)+eps) + wd * theta_{t-1} )
        i.e. the decay term wd*theta is applied OUTSIDE the adaptive scaling.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                # decoupled weight decay: theta <- theta - eta*wd*theta  (step 1 of Alg. 2)
                p.mul_(1 - lr * wd)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


