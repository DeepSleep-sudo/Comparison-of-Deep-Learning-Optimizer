"""
================================================================================
train_transformer.py  --  Optimizer comparison on a small Transformer LM
================================================================================
Modern optimizers (AdamW, Lion, Sophia, SOAP, Muon, Adam-mini, ...) are designed
for Transformer training. This script trains a small GPT-style character-level
language model with each optimizer and records loss curves + throughput.

Data
----
Uses a self-contained corpus shipped with the repo (``data/tiny_corpus.txt``).
If the file is absent, a synthetic periodic corpus is generated so the script
always runs offline.

Run:
    python src/experiments/train_transformer.py --device cuda:1 --steps 1500
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

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


# ----------------------------------------------------------------------------
# Tiny GPT
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_head=4, n_layer=3, block=64):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(block, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def get_corpus():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tiny_corpus.txt")
    if not os.path.exists(path):
        # synthetic periodic corpus: learnable structure + long-range periodicity
        rng = np.random.default_rng(0)
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        chunks = []
        for _ in range(4000):
            period = rng.integers(3, 12)
            pat = "".join(rng.choice(list(alphabet), size=period))
            chunks.append(pat * rng.integers(3, 8))
        text = " ".join(chunks)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_batches(text, block, batch_size, n_batches, device, seed=1):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - block - 1, size=batch_size)
        x = torch.stack([data[i:i + block] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
        batches.append((x, y))
    return len(chars), batches


def lr_candidates(default_lr, override=None):
    """Learning-rate candidates for one optimizer.

    Every optimizer gets the same budget of three trials. The centre of the grid
    is the learning rate its own authors recommend (the value stored in the
    registry), so the grid follows each family's natural scale instead of
    imposing one scale on all of them: an SGD run at Adam's 1e-3 is not a test
    of SGD, it is a test of a mis-tuned SGD.
    """
    if override:
        return list(override)
    d = float(default_lr)
    return [d / 3.0, d, d * 3.0]


OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "logs", "transformer_results.json"))


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


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(name, opt_cls, opt_kwargs, lr, steps, vocab, batches, device, seed=0):
    set_seed(seed)
    model = TinyGPT(vocab).to(device)
    kwargs = dict(opt_kwargs)
    kwargs["lr"] = lr
    if name == "AdaBelief":
        kwargs["eps"] = 1e-12
    if name == "Muon":
        # apply Muon to 2-D hidden weights only (embeddings/head/norms -> AdamW)
        from src.optimizers import AdamW
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        opt = torch.optim.AdamW(other_params, lr=lr)
        muon = opt_cls(muon_params, **kwargs)
        opt = _Hybrid(muon, opt)
    elif name == "Lookahead":
        from src.optimizers import Adam
        opt = opt_cls(model.parameters(), Adam, k=5, alpha=0.5, lr=lr)
    else:
        opt = opt_cls(model.parameters(), **kwargs)

    history = []
    t0 = time.time()
    for step in range(steps):
        x, y = batches[step % len(batches)]
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    dt = time.time() - t0
    return history, dict(time=dt, sec_per_step=dt / steps)


class _Hybrid:
    """Helper to run two optimizers over disjoint parameter sets (for Muon)."""
    def __init__(self, *opts):
        self.opts = opts
    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()
    def step(self):
        for o in self.opts:
            o.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--lr_grid", type=float, nargs="+", default=None,
                    help="if given, this grid is used for every optimizer "
                         "(legacy behaviour); default = three candidates centred "
                         "on each optimizer's own registry lr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one run per seed; the reported number is the mean")
    ap.add_argument("--optimizers", nargs="+", default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[train_transformer] device = {device}  seeds={args.seeds}")
    text = get_corpus()

    names = args.optimizers or ["Adam", "AdamW", "RAdam", "Lion", "LAMB", "Adan",
                                "AdamMini", "ScheduleFree", "Sophia", "SOAP", "Muon"]
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
                # data order varies with the seed as well, otherwise the spread
                # only reflects initialisation and understates the variance
                vocab, batches = make_batches(text, args.block, args.batch_size,
                                              64, device, seed=1000 + sd)
                print(f"  running {name} lr={lr:.1e} seed={sd} ...", flush=True)
                try:
                    hist, meta = train_one(name, opt_cls, opt_kwargs, lr, args.steps,
                                           vocab, batches, device, seed=sd)
                except Exception as e:
                    print(f"    !! {name} lr={lr:.1e} seed={sd} failed: {e}")
                    continue
                per_seed[sd] = dict(final=float(np.mean(hist[-50:])), hist=hist, meta=meta)
            if not per_seed:
                continue
            mean_final = float(np.mean([v["final"] for v in per_seed.values()]))
            print(f"    -> {name} lr={lr:.1e}: mean final={mean_final:.4f} "
                  f"over {len(per_seed)} seed(s)", flush=True)
            if best is None or mean_final < best[0]:
                best = (mean_final, lr, per_seed)

        if best is not None:
            mean_final, lr, per_seed = best
            finals = [v["final"] for v in per_seed.values()]
            std_final = float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0
            n = min(len(v["hist"]) for v in per_seed.values())
            mean_hist = np.mean([v["hist"][:n] for v in per_seed.values()], axis=0).tolist()
            results[name] = dict(
                final_loss=mean_final,
                final_loss_std=std_final,
                n_seeds=len(finals),
                lr=lr,
                history=mean_hist,
                meta=dict(sec_per_step=float(np.mean(
                    [v["meta"]["sec_per_step"] for v in per_seed.values()]))),
                per_seed={str(k): dict(final=v["final"],
                                       sec_per_step=v["meta"]["sec_per_step"])
                          for k, v in per_seed.items()},
            )
            print(f"  == {name}: final={mean_final:.4f} +- {std_final:.4f} "
                  f"@ lr={lr:.1e} ({results[name]['meta']['sec_per_step']*1000:.1f} ms/step)",
                  flush=True)
            n_saved = save_merged({name: results[name]})
            print(f"     [json] {n_saved} optimizers on disk", flush=True)

    plot_loss_curves({n: r["history"] for n, r in results.items()},
                     "Tiny Transformer LM training loss",
                     "transformer_loss", ylabel="Cross-entropy", logy=False, smooth=20)

    print(f"[json] saved {OUT_PATH} ({len(results)} updated this run)")
    print("done.")


if __name__ == "__main__":
    main()
