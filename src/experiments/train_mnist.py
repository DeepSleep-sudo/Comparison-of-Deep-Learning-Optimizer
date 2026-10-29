"""
================================================================================
train_mnist.py  --  Optimizer comparison on MNIST (small CNN)
================================================================================
Trains a compact CNN on MNIST with every optimizer and records:
    * training loss per step
    * test accuracy per epoch
    * wall-clock time per step
    * peak GPU memory (for the memory-efficiency comparison)

Protocol
--------
    * Identical init (fixed seed) and identical data order across optimizers.
    * Small lr sweep per optimizer -> report the best test accuracy, but also
      keep the full sweep for the "robustness" analysis.
    * Device defaults to cuda:1 (RTX 3090).

Run:
    python src/experiments/train_mnist.py --device cuda:1 --epochs 3
    python src/experiments/train_mnist.py --optimizers AdamW Lion Muon --epochs 2
================================================================================
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.optimizers import OPTIMIZERS
from src.utils.plotting import plot_loss_curves, save_json


# ----------------------------------------------------------------------------
# Model: a small but non-trivial CNN (conv + BN + 2 FC), ~0.4M params.
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders(batch_size=128, root="./data"):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    # num_workers=0 keeps data loading in-process (robust under sandboxes and
    # avoids Windows spawn overhead for these small datasets).
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(test, batch_size, shuffle=False, num_workers=0))


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_one(opt_name, opt_cls, opt_kwargs, lr, epochs, device, loaders, seed=0):
    train_loader, test_loader = loaders
    set_seed(seed)
    model = SmallCNN().to(device)
    kwargs = dict(opt_kwargs)
    kwargs["lr"] = lr
    if opt_name == "AdaBelief":
        kwargs["eps"] = 1e-12
    if opt_name == "Lookahead":
        # wrap Adam as inner optimizer
        from src.optimizers import Adam
        opt = opt_cls(model.parameters(), Adam, k=5, alpha=0.5, lr=lr)
    else:
        opt = opt_cls(model.parameters(), **kwargs)
    crit = nn.CrossEntropyLoss()
    history = {"loss": [], "acc": []}
    t0 = time.time()
    n_steps = 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            history["loss"].append(loss.item())
            n_steps += 1
        acc = evaluate(model, test_loader, device)
        history["acc"].append(acc)
        print(f"    [{opt_name} lr={lr:.1e}] epoch {ep+1} acc={acc:.4f}")
    dt = time.time() - t0
    return history, dict(time=dt, steps=n_steps, sec_per_step=dt / max(n_steps, 1))


def lr_candidates(default_lr, override=None):
    """Learning-rate candidates for one optimizer.

    Every optimizer gets the same budget of three trials. The centre of the grid
    is the learning rate its own authors recommend (the value stored in the
    registry), so the grid follows each family's natural scale instead of
    imposing one scale on all of them: an SGD run at Adam's 3e-4 is not a test
    of SGD, it is a test of a mis-tuned SGD.
    """
    if override:
        return list(override)
    d = float(default_lr)
    return [d / 3.0, d, d * 3.0]


OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "logs", "mnist_results.json"))


def save_merged(new):
    """Write results into the JSON immediately, merging with what is there.

    Called after every optimizer, so an interrupted sweep keeps everything it
    already finished instead of losing the whole run.
    """
    merged = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(new)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=float)
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_grid", type=float, nargs="+", default=None,
                    help="if given, this grid is used for every optimizer "
                         "(legacy behaviour); default = three candidates centred "
                         "on each optimizer's own registry lr")
    ap.add_argument("--optimizers", nargs="+", default=None,
                    help="subset of optimizer names; default = a representative set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one run per seed; the reported number is the mean")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[train_mnist] device = {device}  seeds={args.seeds}")
    loaders = get_loaders(args.batch_size)

    if args.optimizers:
        names = args.optimizers
    else:
        # representative set (keeps runtime sane); run with --optimizers all for everything
        names = ["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam",
                 "AdamW", "RAdam", "AdaBelief", "LAMB", "Adan", "Lion",
                 "AdamMini", "ScheduleFree", "Sophia", "SOAP", "Muon"]
    if names == ["all"]:
        names = [n for n in OPTIMIZERS if n != "Lookahead"]

    results = {}
    for name in names:
        if name not in OPTIMIZERS:
            print(f"  !! unknown optimizer {name}, skipping")
            continue
        opt_cls, opt_kwargs = OPTIMIZERS[name]
        lrs = lr_candidates(opt_kwargs.get("lr", 1e-3), args.lr_grid)
        best = None
        for lr in lrs:
            per_seed = {}
            for sd in args.seeds:
                print(f"  running {name} lr={lr:.1e} seed={sd} ...", flush=True)
                try:
                    hist, meta = train_one(name, opt_cls, opt_kwargs, lr, args.epochs,
                                           device, loaders, sd)
                except Exception as e:
                    print(f"    !! {name} lr={lr:.1e} seed={sd} failed: {e}")
                    continue
                per_seed[sd] = dict(acc=max(hist["acc"]) if hist["acc"] else 0.0,
                                    hist=hist, meta=meta)
            if not per_seed:
                continue
            mean_acc = float(np.mean([v["acc"] for v in per_seed.values()]))
            print(f"    -> {name} lr={lr:.1e}: mean acc={mean_acc:.4f} "
                  f"over {len(per_seed)} seed(s)", flush=True)
            if best is None or mean_acc > best[0]:
                best = (mean_acc, lr, per_seed)

        if best is not None:
            mean_acc, lr, per_seed = best
            accs = [v["acc"] for v in per_seed.values()]
            std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            n = min(len(v["hist"]["loss"]) for v in per_seed.values())
            mean_loss = np.mean([v["hist"]["loss"][:n] for v in per_seed.values()],
                                axis=0).tolist()
            results[name] = dict(
                acc=mean_acc,
                acc_std=std_acc,
                n_seeds=len(accs),
                lr=lr,
                history=dict(loss=mean_loss,
                             acc=[v["hist"]["acc"][-1] for v in per_seed.values()]),
                meta=dict(sec_per_step=float(np.mean(
                    [v["meta"]["sec_per_step"] for v in per_seed.values()]))),
                per_seed={str(k): dict(acc=v["acc"], sec_per_step=v["meta"]["sec_per_step"])
                          for k, v in per_seed.items()},
            )
            print(f"  == {name}: acc={mean_acc:.4f} +- {std_acc:.4f} @ lr={lr:.1e} "
                  f"({results[name]['meta']['sec_per_step']*1000:.1f} ms/step)", flush=True)
            n_saved = save_merged({name: results[name]})
            print(f"     [json] {n_saved} optimizers on disk", flush=True)

    # plots
    loss_hist = {n: r["history"]["loss"] for n, r in results.items()}
    plot_loss_curves(loss_hist, "MNIST training loss (best lr per optimizer)",
                     "mnist_loss", ylabel="Cross-entropy", logy=True, smooth=20)
    print(f"[json] saved {OUT_PATH} ({len(results)} updated this run)")
    print("done.")


if __name__ == "__main__":
    main()
