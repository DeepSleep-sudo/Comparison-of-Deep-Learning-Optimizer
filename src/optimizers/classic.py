"""
================================================================================
classic.py  --  Classical first-order optimizers
================================================================================
Implements the three foundational optimizers that everything else is built on:

    (1) SGD              -- Robbins & Monro (1951) stochastic approximation
    (2) SGD with Momentum-- Polyak (1964) heavy-ball
    (3) Nesterov (NAG)   -- Nesterov (1983) accelerated gradient

Every class subclasses ``torch.optim.Optimizer`` and mirrors the argument names
of the built-in PyTorch optimizers, so they are drop-in replacements.

Convention used throughout this file (and all other optimizer files):
    theta_t : current parameters
    g_t     : gradient of the mini-batch loss w.r.t. theta_t  (g_t = dL/dtheta_t)
    eta     : learning rate
    t       : step counter (1-indexed inside the update, i.e. t=1 at first step)

Equation numbers in the comments refer to the cited papers.
================================================================================
"""

import torch
from torch.optim.optimizer import Optimizer


# ------------------------------------------------------------------------------
# (1) SGD  --  Robbins & Monro (1951)
# ------------------------------------------------------------------------------
class SGD(Optimizer):
    r"""Vanilla Stochastic Gradient Descent.

    Update rule (Robbins & Monro, 1951, "A Stochastic Approximation Method",
    Annals of Mathematical Statistics 22(3); textbook form):

        theta_{t+1} = theta_t - eta * g_t
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Decoupled weight decay is applied by the *caller* convention in
                # experiments; here we use the classic L2-penalty form for parity
                # with torch.optim.SGD: g <- g + wd * theta  (equivalent to
                # minimising L + 0.5*wd*||theta||^2).
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ---- core update: theta <- theta - eta * g -------------------
                p.add_(g, alpha=-lr)

        return loss


# ------------------------------------------------------------------------------
# (2) SGD with Momentum  --  Polyak (1964) "heavy ball"
# ------------------------------------------------------------------------------


class Momentum(Optimizer):
    r"""SGD with classical (heavy-ball) momentum.

    Reference: Polyak, B. T. (1964). "Some methods of speeding up the convergence
    of iteration methods."  USSR Computational Mathematics and Mathematical Physics.
    (Also popularised for NN training in Rumelhart, Hinton & Williams, 1986.)

    Update rules (Polyak 1964, eq. for the heavy-ball method):

        m_t     = mu * m_{t-1} + g_t                 (momentum buffer, an EMA of g)
        theta_t = theta_{t-1} - eta * m_t
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            damp = group["dampening"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    # first step: initialise the velocity to the raw gradient
                    buf = state["momentum_buffer"] = g.detach().clone()
                else:
                    buf = state["momentum_buffer"]
                    # m_t = mu * m_{t-1} + (1 - dampening) * g_t
                    buf.mul_(mu).add_(g, alpha=1.0 - damp)

                # Nesterov variant: use the "look-ahead" gradient direction.
                if nesterov:
                    g = g.add(buf, alpha=mu)      # g + mu*m_t
                else:
                    g = buf

                p.add_(g, alpha=-lr)

        return loss


# ------------------------------------------------------------------------------
# (3) Nesterov Accelerated Gradient (NAG)
# ------------------------------------------------------------------------------


class Nesterov(Optimizer):
    r"""Nesterov Accelerated Gradient.

    Reference: Nesterov, Y. (1983). "A method of solving a convex programming
    problem with convergence rate O(1/k^2)."  Soviet Mathematics Doklady.
    (Deep-learning formulation follows Sutskever, Martens, Dahl & Hinton, 2013,
    "On the importance of initialization and momentum in deep learning", ICML.)

    Two equivalent views:

    (a) Nesterov's original: evaluate the gradient at the *look-ahead* point
        theta_t + mu * m_t:

            m_{t+1} = mu * m_t + eta * grad L(theta_t + mu * m_t)      (NAG, Sutskever eq. 7)
            theta_{t+1} = theta_t - m_{t+1}

    (b) The form implemented here (matching torch.optim.SGD(nesterov=True)):

            m_t = mu * m_{t-1} + g_t
            g_look = g_t + mu * m_t
            theta_t = theta_{t-1} - eta * g_look
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = g.detach().clone()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(g)          # m_t = mu*m_{t-1} + g_t

                # NAG look-ahead update:
                #   theta <- theta - eta * (g_t + mu * m_t)
                # implemented as two fused adds to avoid an extra allocation.
                p.add_(g, alpha=-lr)
                p.add_(buf, alpha=-lr * mu)

        return loss
