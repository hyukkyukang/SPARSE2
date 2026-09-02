# Dynamic-Vocabulary Learned Sparse Retrieval — pilot study

A sparse retriever whose output dimensions are **defined by text** rather than by rows
of a learned matrix. Vocabulary entry *j* is a text *t_j*; its vector is `f(t_j)` for
the same encoder that embeds the input. Adding an entry means encoding its text — no
new parameters, no retraining:

```
h_1..h_n = f(x)                      contextual word-unit states of the input
v_j      = f(t_j)                    entry vector, produced not stored
s_j(x)   = log(1 + ReLU(t·cos(g(h̃_i), ṽ_j) − b))   max-pooled over i
score    = Σ_j s_j(q)·s_j(d)         ordinary inverted-index dot product
```

Nothing in the model is indexed by *j*. This repository runs the pilot protocol
(`notes/protocol.md`) that tests whether entries added **after** training behave like
entries seen during it.

## Headline result

**They do, and most clearly in the case the idea was designed for.** The recovery
ratio ρ measures how much of the oracle's benefit from a vocabulary entry survives
when that entry is inserted post-hoc (Pilot D, C₁ = 1.85M passages, dev-small):

| held-out split | ρ | 95% CI | \|Q_H\| | MRR@10 all → seen-only |
|---|---|---|---|---|
| **rare** — the whole bottom frequency decile; new low-frequency terms | **0.932** | [0.80, 1.07] | 352 | 0.2017 → 0.1961 |
| random-stratified — 20% within each decile | 0.685 | [0.60, 0.77] | 3346 | 0.2010 → 0.1850 |
| cluster — a semantically new region of the entry space | 0.567 | [0.49, 0.65] | 1675 | 0.1944 → 0.1765 |

Every denominator is valid (≥ 0.02 absolute MRR, CI excluding zero), so ρ is
interpretable in all three cases.

Supporting evidence for the premise:

* The trained frozen-encoder model reaches **MRR@10 0.2010** on C₁ — above its own
  full-vocabulary oracle (0.1991) and above BM25 (0.1882), up from 0.1251 untrained.
* **Insertion beats aliasing.** Mapping each held-out entry to its nearest seen entry
  instead of inserting it gives 0.1595 against 0.2010, so the held-out entries are not
  redundant with the trained vocabulary.
* **Calibration holds on the random split**: |gap_r| = 0.067, far inside the log(1.5)
  bound — a held-out entry fires at nearly the same rate as a seen entry of the same
  frequency.
* **The control kills the alternative explanation.** Replacing the vocabulary with
  random unit vectors of the same shape collapses the model to **MRR@10 = 0.0000**.
  The text-defined vocabulary is carrying the signal, not the architecture.

The clear negative: on the **cluster** split, signed gap_r = **+0.47** — held-out
entries systematically *over*-fire when they come from an unseen region of the entry
space. That is the failure mode insertion-time calibration (Pilot E) exists to fix.

## Three protocol expectations were falsified

| | expectation | what happened |
|---|---|---|
| **H3** | whitening raises z-gap ≥2× and cuts nnz/token ~10× | Neither. **Centering** does most of the density reduction, and whitening *lowers* cross-representation self-hit at every layer below 11. It is still worth 0.096 vs 0.056 MRR@10 end to end — peakiness was simply the wrong yardstick. |
| **H5** | bare-string entries are hubbier than contextual prototypes | **Reversed.** Hub skewness 3.6 (R1) vs 17.2 (R2); it is the *prototype* hub list that is dominated by function words. The df–frequency half of H5 does hold (Spearman 0.25 vs 0.57). |
| §A.4 rule | select the layer by cross-representation self-hit@10 | **Anti-correlated with retrieval.** Identity peaks at layer 9, but layer 12 retrieves 30% better (0.1251 vs 0.0963). The rule's *tie-breaker* — related-term MRR, 23× vs 210× random — is what tracks effectiveness. |

Two further findings about training the encoder:

* The **LoRA arm collapses without vocabulary dropout** on every configuration tried
  (two seeds, two warmup schedules). `log(1+ReLU(·))` is a hard gate: an entry that
  falls below threshold for a whole batch receives no gradient and cannot return, and a
  trainable encoder can switch most of the vocabulary off in a few dozen steps. Masking
  30% of entries per step prevents it.
* After 3,000 LoRA steps an entry **re-encodes to cosine 0.46** with its previous
  vector. The entry space moves faster than any practical refresh interval tracks — a
  concrete cost of making the encoder trainable in this design.

## Pipeline validation (run before any pilot)

| system | full-collection MRR@10 | published |
|---|---|---|
| BM25 (k1=0.82, b=0.68) | 0.1874 | 0.1875 |
| SPLADE++ CoCondenser-EnsembleDistil | 0.3827 | 0.383 |
| e5-base-v2 | 0.3542 | ~0.35 |
| bge-base-en-v1.5 | 0.3498 | ~0.35 |

All four reproduce, so the pilots do not inherit a prefix / pooling / normalisation bug.

## Selected configuration

e5-base-v2, **hidden layer 12**, entries as **contextual prototypes** (mean word-unit
state over k=50 occurrences), whitened per space, k_d=128 / k_q=16 / log1p, τ set so
mean nnz ≈ 120 per document. Vocabulary V = 30,000 lowercase alphabetic words with
≥100 occurrences in the prototype pool; the bottom decile's minimum collection
frequency is 498, so no entry has a noisy prototype.

## Layout

```
dvlsr/      library: paths, data, encoders, whitening, sparse rule, GPU inverted index,
            model (residual head + text-defined entry side), metrics
scripts/    numbered stages, each writing a JSON artifact into results/
configs/    polysemy set (sense probe) and run configs
results/    small JSON results, versioned
reports/    FINDINGS.md plus a report per pilot and qualitative dumps
notes/      protocol summary, deviations, running log
run_all.sh  the whole study in order
```

Large artifacts (collection binary, corpus embeddings, banks, prototypes, checkpoints)
live outside the repo under `$DVLSR_DATA` (default `/workspace/SPARSE/dvlsr`).

## Reproducing

```bash
export DVLSR_DATA=/path/to/scratch
./run_all.sh          # sets the OpenBLAS thread cap and the JDK path Pyserini needs
```

Requires one or more 40 GB GPUs (the study ran on 8× A100-40GB), a JDK for Pyserini,
and ~400 GB of scratch. Read `notes/deviations.md` before comparing against the
protocol: eleven departures are documented there, each with the measurement that
forced it — including a manual audit of the sense labels that found a 12% error rate
and corrected it (agreement 0.883 → 0.963).

## Status

Pilots A, B and C are complete. Pilot D reports the four decisive runs above (V1 on
three splits plus the C-rand control); the LoRA and full-fine-tuning arms are trained
and their C₁ encodes were in progress when the study was time-boxed. Pilot E and the
full-corpus confirmation are implemented (`scripts/50_*`, `scripts/51_*`,
`scripts/60_*`) but not yet run — the cluster-split over-firing above is what would
trigger them.
