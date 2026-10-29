"""
================================================================================
time_probe.py  --  clean per-step cost of each optimizer, measured alone
================================================================================
The matrix runs execute two jobs at once, so the ms/step stored in
mnist_results.json / transformer_results.json is inflated by contention and must
not be quoted.  This probe measures the same quantity with nothing else running:
50 steps after 10 warm-up steps, on both models, sequentially.

Run (with no other job on the GPU):
    python src/utils/time_probe.py --device cuda:1
================================================================================
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.optimizers import OPTIMIZERS
from src.experiments.train_mnist import SmallCNN, get_loaders
from src.experiments.train_transformer import TinyGPT, get_corpus, make_batches

NAMES = ["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam", "AdamW",
         "Lion", "Sophia", "SOAP", "Muon", "LAMB", "ScheduleFree", "Shampoo"]


def time_mnist(name, cls, kw, device, warm=10, n=50):
    tr, _ = get_loaders(128)
    model = SmallCNN().to(device)
    kw = dict(kw)
    kw.setdefault("lr", 1e-3)
    opt = cls(model.parameters(), **kw)
    crit = nn.CrossEntropyLoss()
    it = iter(tr)
    for _ in range(warm):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


def time_tf(name, cls, kw, device, warm=10, n=50):
    text = get_corpus()
    vocab, batches = make_batches(text, 64, 32, 64, device, seed=1000)
    model = TinyGPT(vocab).to(device)
    kw = dict(kw)
    kw["lr"] = kw.pop("lr", 1e-3)
    opt = cls(model.parameters(), **kw)
    for _ in range(warm):
        x, y = batches[0]
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        x, y = batches[0]
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    device = args.device

    out = {}
    for name in NAMES:
        if name not in OPTIMIZERS:
            continue
        cls, kw = OPTIMIZERS[name]
        row = {}
        for tag, fn in (("mnist", time_mnist), ("transformer", time_tf)):
            try:
                row[tag] = fn(name, cls, kw, device)
            except Exception as e:
                row[tag] = None
                print(f"  !! {name} {tag}: {e}")
        out[name] = row
        print("%-14s MNIST %7.2f ms/step   Transformer %8.2f ms/step"
              % (name, row["mnist"] or -1, row["transformer"] or -1), flush=True)

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "logs", "timing_probe.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print("[json]", path)


if __name__ == "__main__":
    main()
