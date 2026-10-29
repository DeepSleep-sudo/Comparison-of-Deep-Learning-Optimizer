"""
================================================================================
Deep-Learning Optimizer Comparison -- fourteen optimizers, one protocol
================================================================================
From-scratch implementations of fourteen optimizers spanning 1951 (SGD) to 2024
(SOAP, Muon, Schedule-Free).  Every class subclasses ``torch.optim.Optimizer``
and is a drop-in replacement for the built-in versions; each docstring maps the
code to the source paper's equations.

The set is chosen to cover five design families, with at least one member of
each and a classical and a modern representative of every family that has both:

    temporal filtering (F1)          SGD, Momentum, NAG, Lion
    diagonal scaling (F2)            AdaGrad, RMSProp, Adam, AdamW
    matrix preconditioning (F3)      Shampoo, Sophia, SOAP
    spectral (F4)                    Muon
    meta-structure (F5)              LAMB, Schedule-Free

Quick usage
-----------
    from src.optimizers import AdamW, Lion, Muon
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

Registry
--------
    OPTIMIZERS : dict name -> (class, default_kwargs)
    ALL_NAMES  : list of registry keys
================================================================================
"""

from .classic import SGD, Momentum, Nesterov
from .adaptive import AdaGrad, RMSProp, Adam, AdamW
from .modern import LAMB, Lion, ScheduleFreeAdamW
from .second_order import Shampoo, Sophia, SOAP
from .muon import Muon

__all__ = [
    "SGD", "Momentum", "Nesterov",
    "AdaGrad", "RMSProp", "Adam", "AdamW",
    "Shampoo", "Sophia", "SOAP",
    "Muon",
    "LAMB", "Lion", "ScheduleFreeAdamW",
]

# Registry used by the experiment scripts.  The dict values are
# (class, default_kwargs); the default learning rate is the value the method's
# own authors recommend, and it is the centre of the three-point grid that the
# experiments sweep around.
OPTIMIZERS = {
    # ---- F1 temporal filtering -------------------------------------------
    "SGD":          (SGD,          dict(lr=1e-2)),
    "Momentum":     (Momentum,     dict(lr=1e-2, momentum=0.9)),
    "Nesterov":     (Nesterov,     dict(lr=1e-2, momentum=0.9)),
    "Lion":         (Lion,         dict(lr=1e-4, weight_decay=1e-2)),
    # ---- F2 diagonal scaling ---------------------------------------------
    "AdaGrad":      (AdaGrad,      dict(lr=1e-1)),
    "RMSProp":      (RMSProp,      dict(lr=1e-2, alpha=0.99)),
    "Adam":         (Adam,         dict(lr=1e-3)),
    "AdamW":        (AdamW,        dict(lr=1e-3, weight_decay=1e-2)),
    # ---- F3 matrix preconditioning ---------------------------------------
    "Shampoo":      (Shampoo,      dict(lr=1e-2)),
    "Sophia":       (Sophia,       dict(lr=1e-4)),
    "SOAP":         (SOAP,         dict(lr=3e-3)),
    # ---- F4 spectral ------------------------------------------------------
    "Muon":         (Muon,         dict(lr=0.02)),
    # ---- F5 meta-structure ------------------------------------------------
    "LAMB":         (LAMB,         dict(lr=1e-3)),
    "ScheduleFree": (ScheduleFreeAdamW, dict(lr=2.5e-3)),
}

ALL_NAMES = list(OPTIMIZERS.keys())

__version__ = "1.0.0"
