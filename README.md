# Deep-Learning Optimizer Comprison 

From-scratch implementations of fourteen optimizers spanning 1951 (SGD) to 2024
(SOAP, Muon, Schedule-Free), each compared on the same three scenarios under the
same protocol, with every figure reproducible from the shipped JSON metadata.

---

## The fourteen

| Optimizer | Family | Year | Venue | Default `lr` |
|---|---|---|---|---|
| SGD | F1 temporal filtering | 1951 | Ann. Math. Stat. | 1e-2 |
| Momentum | F1 temporal filtering | 1964 | USSR Comput. Math. | 1e-2 |
| Nesterov (NAG) | F1 temporal filtering | 1983 | Soviet Math. Dokl. | 1e-2 |
| Lion | F1 temporal filtering | 2023 | ICML | 1e-4 |
| AdaGrad | F2 diagonal scaling | 2011 | JMLR | 1e-1 |
| RMSProp | F2 diagonal scaling | 2012 | Coursera lecture | 1e-2 |
| Adam | F2 diagonal scaling | 2015 | ICLR | 1e-3 |
| AdamW | F2 diagonal scaling | 2019 | ICLR | 1e-3 |
| Shampoo | F3 matrix preconditioning | 2018 | ICML | 1e-2 |
| Sophia | F3 matrix preconditioning | 2023 | ICLR 2024 | 1e-4 |
| SOAP | F3 matrix preconditioning | 2024 | ICLR 2025 | 3e-3 |
| Muon | F4 spectral | 2024 | blog | 2e-2 |
| LAMB | F5 meta-structure | 2020 | ICLR | 1e-3 |
| Schedule-Free | F5 meta-structure | 2024 | NeurIPS | 2.5e-3 |

The set covers five design families with at least one member each, and both a
classical and a modern representative wherever the family has both. The `lr`
column is the value the method's own authors recommend; it is the centre of the
three-point grid the experiments sweep around.

---

## Results

Three scenarios, fourteen optimizers, one protocol. Values are means over three
seeds (the synthetic scenario is deterministic and has no seed variance).

| Optimizer | Family | Synthetic `log10 f(x)` ↓ | MNIST acc. % ↑ | Transformer CE ↓ |
|---|---|---|---|---|
| **Muon** | F4 | −4.92 | **99.00 ± 0.19** | **0.318 ± 0.003** |
| Shampoo | F3 | −3.73 | 98.97 ± 0.13 | 0.319 ± 0.032 |
| Lion | F1 | **−1.61** | 98.89 ± 0.13 | 0.340 ± 0.012 |
| AdamW | F2 | −3.42 | 98.79 ± 0.06 | 0.499 ± 0.021 |
| Sophia | F3 | −8.95 | 98.78 ± 0.04 | 0.499 ± 0.015 |
| Adam | F2 | **−9.04** | 98.85 ± 0.11 | 0.501 ± 0.043 |
| SOAP | F3 | −3.34 | 98.88 ± 0.16 | 0.534 ± 0.010 |
| Schedule-Free | F5 | −4.63 | 98.96 ± 0.04 | 0.635 ± 0.077 |
| LAMB | F5 | −8.43 | 98.69 ± 0.08 | 0.740 ± 0.017 |
| AdaGrad | F2 | −2.73 | 98.55 ± 0.24 | 1.109 ± 0.088 |
| RMSProp | F2 | −4.26 | 98.70 ± 0.30 | 1.342 ± 0.179 |
| Nesterov | F1 | −3.24 | 98.94 ± 0.11 | 2.175 ± 0.095 |
| Momentum | F1 | −3.73 | 98.83 ± 0.27 | 2.193 ± 0.086 |
| SGD | F1 | −3.65 | 98.67 ± 0.05 | **3.176 ± 0.002** |

Three things these numbers show, in order of how much they matter:

1. **The Transformer is the only scenario that separates the methods widely.**
   Its spread is 2.86 in cross-entropy (SGD 3.176 against Muon 0.318, a factor of
   ten). On MNIST the whole spread is 0.45 points against per-method standard
   deviations of 0.04–0.30: 85 of the 91 pairwise differences sit inside two
   standard deviations, so no ranking should be read into that column.
2. **The ordering does not transfer across scenario types.** Spearman rank
   correlation between the synthetic landscapes and the Transformer is 0.14
   (p = 0.64), and against MNIST 0.02 (p = 0.94). The only pair that correlates
   significantly is MNIST against the Transformer, at 0.57 (p = 0.034) — the two
   *neural* scenarios agree with each other and neither agrees with the analytic
   one. Lion is last on the synthetic landscapes and third on the Transformer;
   LAMB is third and ninth; Adam is first and sixth.
3. **Matrix and spectral methods cost more per step and buy nothing on an
   already well-conditioned task.** Measured with nothing else on the GPU:
   Shampoo 299 ms/step on the Transformer against Adam's 11.8 (25×), SOAP 42.8
   (3.6×), Muon 21.7 (1.8×). On MNIST all fourteen are within 15–22 ms.

Figures: `results/figures/matrix_three_tasks.png` (three panels),
`matrix_rank_heatmap.png` (per-scenario ranks), `bench_*.png`,
`mnist_loss.png`, `transformer_loss.png`.

---

## Layout

```
comp6704-optimizers/
├── src/
│   ├── optimizers/          # the fourteen, one module per family
│   │   ├── classic.py       # SGD, Momentum, Nesterov
│   │   ├── adaptive.py      # AdaGrad, RMSProp, Adam, AdamW
│   │   ├── modern.py        # LAMB, Lion, ScheduleFreeAdamW
│   │   ├── second_order.py  # Shampoo, Sophia, SOAP
│   │   └── muon.py          # Muon
│   ├── experiments/
│   │   ├── test_functions.py     # scenario 1: five analytic landscapes
│   │   ├── train_mnist.py        # scenario 2: small CNN
│   │   └── train_transformer.py  # scenario 3: tiny GPT
│   └── utils/
│       ├── plotting.py      # shared figure style
│       ├── make_matrix.py   # JSON -> matrix table + both figures
│       └── time_probe.py    # clean per-step timing, nothing else on the GPU
├── results/
│   ├── figures/             # 11 figures, PNG (report) + PDF (LaTeX)
│   └── logs/                # the metadata the figures are built from
└── data/
    └── tiny_corpus.txt      # 144k-char corpus for the Transformer scenario
```

---

## Reproduce

```bash
python -m pip install -r requirements.txt

# scenario 1: five analytic landscapes (deterministic, ~5 min on one GPU)
python src/experiments/test_functions.py --device cuda:1

# scenario 2: MNIST, small CNN (14 methods x 3 lr x 3 seeds, ~50 min)
python src/experiments/train_mnist.py --device cuda:1 --epochs 3 --seeds 0 1 2

# scenario 3: tiny Transformer (same budget; Shampoo alone takes ~70 min)
python src/experiments/train_transformer.py --device cuda:1 --steps 1500 --seeds 0 1 2

# clean per-step cost, run with nothing else on the GPU
python src/utils/time_probe.py --device cuda:1
```

All runs use `cuda:1` on a single RTX 3090. MNIST is downloaded automatically by
`torchvision`; the Transformer corpus ships with the repo, so that scenario runs
offline.

## Regenerate the figures (no GPU needed)

The shipped `results/logs/*.json` are the metadata. Every number in the table
above and every panel of both matrix figures is computed from them:

```bash
python src/utils/make_matrix.py            # screen-size figures
python src/utils/make_matrix.py --paper    # column-width variants for LaTeX
```

`make_matrix.py` writes `results/logs/matrix_summary.json`, both figures (PNG +
PDF) and `latex/experiments/tables/table_matrix.tex`. Nothing is transcribed by
hand, so a new figure or a different cut of the data is a one-line change in
that script rather than a re-run of the experiments.

### Metadata schema

| File | Contents |
|---|---|
| `bench_test_functions.json` | `{function: {optimizer: {final, lr}}}` — best final objective and the learning rate that produced it, for each of the five landscapes |
| `mnist_results.json` | `{optimizer: {acc, acc_std, n_seeds, lr, history{loss, acc}, meta{sec_per_step}, per_seed}}` |
| `transformer_results.json` | `{optimizer: {final_loss, final_loss_std, n_seeds, lr, history, meta{sec_per_step}, per_seed}}` |
| `matrix_summary.json` | the assembled 14 × 3 matrix with per-scenario ranks and per-function breakdowns |
| `timing_probe.json` | `{optimizer: {mnist, transformer}}` — ms/step measured alone |

`history` holds the full loss curve (and, for MNIST, the per-epoch accuracy), so
a different smoothing window or a different y-range needs no re-run.

---

## Protocol, and where it stops

- **Equal tuning budget, not one shared grid.** Each optimizer gets three
  learning rates centred on its own recommended value. A shared grid would judge
  SGD at Adam's step size; doing that lowers SGD on MNIST from 98.67% to 96.73%,
  which is a property of the grid rather than of the algorithm.
- **Three seeds** on both neural tasks; the model initialisation *and* the data
  order change with the seed. The synthetic landscapes are deterministic.
- **The synthetic column ranks only F1 and F2.** Those landscapes optimise a
  one-dimensional parameter vector, and both `Shampoo` and `Muon` send a
  parameter with `dim() != 2` to a plain momentum branch. On these landscapes
  Shampoo's loss curve is *bitwise identical* to Momentum's (maximum pointwise
  difference 0.00). F3 and F4 are read from the two neural panels.
- **The learning-rate grid is on the low side.** Ten of the fourteen pick the
  top of their grid on the Transformer, so the reported numbers are lower bounds.
  An out-of-grid check on SGD gives 3.176 / 2.971 / 2.082 at `lr` = 3e-2 / 1e-1 /
  3e-1, against Adam's 0.501.
- **Shampoo recomputes its matrix roots every step** (`update_freq=1`), which is
  the naive variant; the reference implementation defaults to 10. Its 299 ms/step
  is the cost of this implementation, not of the algorithm.
- **MNIST accuracy is the best epoch within the run**, matching the original
  script. With the whole column inside two standard deviations this does not
  change any conclusion, but it is mildly favourable to methods with high
  epoch-to-epoch variance.

### What reproduces exactly

The synthetic scenario is deterministic: re-running it gives the same values
bit-for-bit, and the shipped `bench_test_functions.json` was produced by the code
in this repository. The two neural scenarios reproduce to about `1e-4` rather
than bit-for-bit, because cuDNN picks non-deterministic convolution algorithms;
one Adam MNIST run, for example, lands on 0.9891 against the shipped 0.9890.
Differences of that size do not move any ranking in the table above.

---

## License

MIT — see [LICENSE](LICENSE). Course project, COMP 6704; the optimizer
implementations follow the cited papers.
