# Pilot F — per-entry background normalisation inside the score

Added 2026-09-04, after Pilot D's cluster split resisted every training-time
intervention. This file is the protocol for the pilot: the reasoning, the exact
change, the runs, and the decision rule, written before the runs finished.

## Why

Pilot D's one clear negative is that entries from an unseen *region* of the entry
space over-fire, and on the rebuild five separate attempts to fix that during
training all failed:

| cluster split, frozen encoder | rho | signed gap_r |
|---|---|---|
| no dropout | 0.313 | +0.386 |
| entry-wise vocabulary dropout | 0.288 | +0.345 |
| cluster-wise vocabulary dropout | 0.273 | +0.394 |
| cluster dropout + meta-held-out calibration loss | 0.232 | +0.366 |
| linear head (capacity control) | 0.198 | +0.464 |

The meta arm is the informative one: its calibration loss reached ~0.04 on the
regions it reserved, so it *did* learn to equalise their activation, and the truly
held-out regions over-fired exactly as before. It learned a region-specific fix
because a region-specific fix is the only kind the architecture can express.

That is a representational limit, not a training failure. The score is

    z_ij = t · cos(g(h̃_i), ṽ_j) − b

with **one** scale t and **one** threshold b shared by all 30,000 entries, because
§D.1 forbids any parameter indexed by j. But an entry's firing rate is inherently a
per-entry property: in SPLADE each output dimension owns a row whose norm and bias
absorb it. Here the only way to make entry j fire less is to move token states away
from ṽ_j's direction, which is the co-adaptation that by construction does not
transfer to directions the model never saw.

**H3's failure is the same observation from the other side.** Whitening equalises the
entry space *globally* (second moment over all entries) and that is why it never made
profiles peakier. What is missing is a *local* normalisation: each entry's own
relationship to the token distribution. `signed gap_r` — the quantity Pilot D reports
and Pilot E exists to repair — is exactly that mismatch.

## Which statistic to normalise (measured, not assumed)

The first draft of this pilot normalised each entry's background **mean and standard
deviation**. Before spending the runs, `scripts/48_entry_stats.py` measured what
actually separates an over-firing held-out entry, on the trained `V1_cluster` model,
within frequency decile (32,768 bank tokens):

| statistic of entry j over B_H | held − seen gap | Spearman with the realised rate r_j |
|---|---|---|
| mean μ_j | +0.0037 | 0.44 |
| standard deviation σ_j | +0.0003 | **0.08** |
| 99.9th percentile q_j | +0.0106 | **0.63** |
| maximum | +0.0200 | 0.50 |
| activation rate r_j | log ratio **+0.386** | — |

Two things follow, and both contradict the first draft:

1. **σ_j carries essentially no information about how often an entry fires** (Spearman
   0.08) and its seen/held gap is ~0. Dividing by it is noise, not correction.
2. **The upper tail is what matters.** `s_j` is a max over ~60 word units, so the
   activation rate is a property of the tail, not of the centre. The tail is where the
   seen/held gap is largest (+0.011 at q0.999, +0.020 at the maximum, against +0.004
   at the mean) and it is the best single predictor of the rate (0.63).

So the correction is a **shift that matches the upper tail**, not an affine that
matches the moments.

## The change

Per entry, compute a high quantile of its similarity against the frozen document-side
token bank B_H, and shift it onto the median of comparable seen entries, before the
global scale and threshold:

    a_ij  = cos(g(h̃_i), ṽ_j)
    B_j   = q*_{d(j)} − q_j          A_j = 1          (mode `q`, primary)
    z_ij  = t · (A_j · a_ij + B_j) − b
    s_j(x) = log(1 + ReLU(max_i z_ij))

* **q_j** is the 99.9th percentile of entry j's similarity over B_H under the current g.
  Mode `z` keeps the moment-matching form, `A_j = σ*/σ_j`, `B_j = μ* − μ_j A_j`, as the
  contrast the table above predicts should fail.
* **q*_d** is the median of q_j over **seen** entries in frequency decile d (the §D.1
  leakage rule; the rare split, which holds out a whole decile, falls back to the
  median over all seen entries).
* Conditioning the target on the frequency decile is deliberate: a global target would
  force every entry to the same background firing rate and destroy the frequency
  signal that df calibration depends on. A new entry's decile is computable from its
  corpus frequency at insertion.

**This does not introduce a parameter indexed by j.** A_j and B_j are deterministic
functions of ṽ_j and a frozen bank, with no fitting and no gradient — the same status
as the whitening transform W_V, which §0.5 already applies to every entry. Inserting
an entry is still one forward pass, plus one matmul of its vector against the bank.

**Trained with, not repaired by.** The essential difference from Pilot E: the map is
in the score during *training*, so the model never learns to rely on uncorrected
geometry and there is no train/test mismatch. Pilot E applies a similar map to
held-out entries only, after the fact, to a model trained without it.

**Refresh.** g moves during training, so the statistics are recomputed every
`--norm-every` steps (default 250) over `--norm-bank` sampled bank tokens (default
16,384) and once more before the checkpoint is written, so the stored map is the one
insertion will use. The cost is ~0.75 TFLOP per refresh, under a second, ~13 refreshes
per run.

**Initialisation is neutral by construction.** For an entry whose statistics already
equal its decile median the map is the identity, so b's initialisation from Pilot C
stays valid and the arm is a controlled modification of V1 rather than a new operating
point.

## Runs

| run | split | entry transform | oracle |
|---|---|---|---|
| `V1norm_cluster` | cluster | own (seen-only) | `V1oracle_norm_cluster` |
| `V1normz_cluster` (mode `z`) | cluster | own (seen-only) | `V1oracle_normz_cluster` |
| `V1normWh_cluster` | cluster | shared (token-side) | `V1oracle_normWh_cluster` |
| `V1norm` | random | own (seen-only) | `V1oracle_norm` |

The cluster pair is the decisive test. The `z` pair is the mechanistic contrast: if
tail-matching works and moment-matching does not, the diagnostic table above is
confirmed and the story is clean. The random pair says what the normalisation costs
where calibration was never the problem. The `Wh` pair combines it with the token-side
entry transform, which Pilot D found is worth +23% MRR@10 on its own; that combination
is the configuration we would actually recommend if both hold.

**Pilot E is the post-hoc version of the same two corrections** and is already running:
its `Z` variant is moment matching and its `DF` variant matches the realised document
frequency, which is tail matching by another route, both applied to held-out entries
only after training. Read its table first. If `DF` moves gap_r and `Z` does not, that
is the same conclusion from an independent direction and Pilot F is the trained-in
version of `DF`.

Every arm is compared to the matching baseline from the same rebuild
(`V1_cluster`, `V1_cluster_Wh`, `V1`), never to the A100 numbers.

## H16 and the decision rule

* **H16**: on the cluster split, per-entry normalisation raises ρ by at least 0.15
  absolute over its matching baseline and brings |gap_r| under log(1.5) = 0.405,
  without a loss of MRR@10 on all entries larger than 0.01 absolute and without
  raising nnz(d) above the baseline's.
* **Read gap_r first.** The normalisation is *defined* to remove the activation-rate
  mismatch, so a run whose gap_r does not fall has failed mechanically and its ρ is
  uninformative. A run whose gap_r falls to ~0 while ρ does not move says the gap in
  *rate* was never what cost the recovery, which would be a real and publishable
  negative: it would point at the entry weights rather than their firing.
* **Watch nnz(d) beside every number**, as §E.2 requires. A variant that lifts ρ by
  making documents denser has not fixed anything.
* **Decision.** H16 met on the cluster split → per-entry normalisation is a core
  component of the architecture, not an ablation, and the paper reports the frozen
  encoder + light head + normalised score as the recipe. Met only with the shared
  entry transform → both are core and are reported together. gap_r falls but ρ does
  not → the problem is the *weights* of held-out entries, not their rate; the next
  move is a per-entry scale learned as a function of the entry (still not indexed by
  j) rather than of its background rate. Neither → the limit is representational in
  the sense of §D.7's final branch, and the contribution is the extension use case
  (rare terms, ρ 0.93) with a measured boundary.

## Result and what it changed downstream (2026-09-04)

`V1norm_cluster` against its matching baseline, both from this rebuild:

| cluster split | rho | 95% CI | gap to oracle | signed gap_r | nnz(d) | MRR@10 |
|---|---|---|---|---|---|---|
| baseline `V1_cluster` | 0.313 | [0.23, 0.40] | 0.195 | +0.386 | 77 | 0.1922 |
| Pilot E `DF` (repaired after training) | 0.539 | [0.46, 0.62] | 0.057 | +0.154 | 80 | — |
| **Pilot F `q` (trained in)** | **0.636** | [0.57, 0.70] | 0.063 | +0.179 | **58** | 0.1849 |

**H16 met on all four conditions**: +0.32 rho (threshold +0.15), |gap_r| 0.179 (bound
0.405), MRR cost 0.0073 (tolerance 0.010), nnz(d) *down* 77 → 58 rather than up. Four of
H10's five conditions now hold; only rho ≥ 0.8 does not. Trained-in beats repaired-after
by +0.10 rho, which is the specific claim this pilot was built to test.

The gain is in the numerator, not a shrinking denominator: on Q_H the model's own benefit
from the inserted entries went 0.0484 → 0.0886 while the oracle's went 0.1546 → 0.1392.

**The cost is peak effectiveness**, and it falls on the oracle hardest (0.2526 → 0.2218 on
Q_H). Per-entry idiosyncrasy is therefore partly signal, not only artefact. `--norm-alpha`
now exists to shrink the correction toward each entry's own statistics if that trade needs
tuning.

### Pilot G — the three runs this result makes worth doing

1. **`V1norm_cluster_R1`** — bare-string entries fail on this split *by over-firing*
   (gap_r +0.769, rho −0.048) and they retrieve better than prototypes (0.2214 vs 0.1922).
   Normalisation targets exactly that failure. If it rescues them, insertion costs one
   forward pass instead of ~50 occurrences and retrieves better at the same time. This is
   the highest-value run remaining.
2. **`V1norm_rare`** — the rare split is the motivating use case and the one place recovery
   was already high (0.93 on the A100 run). Normalisation must be shown not to damage it.
3. **`V1param`** — the seen entry rows trained as free parameters, everything else
   identical. This isolates what defining the vocabulary by text actually costs, which
   matters more now that we know normalisation itself costs effectiveness.

The encoder-training arms are capped at 3,000 steps (warmup 500, the same 16% ratio D11
requires) rather than 12,500. They cost ~24 GPU-hours at full length, they answer a
question the study has moved past, and the collapse/stability behaviour D11 documents
appears within the first few hundred steps.

## Pilot G1 result — calibration is necessary but not sufficient (2026-09-04)

Tail normalisation applied to **bare-string** entries on the cluster split:

| cluster split | MRR@10 | rho | signed gap_r | gap_w | nnz(d) |
|---|---|---|---|---|---|
| prototypes | 0.1922 | 0.313 | +0.386 | −0.039 | 77 |
| prototypes + tail normalisation | 0.1849 | **0.636** | +0.179 | −0.021 | 58 |
| bare strings | 0.2214 | −0.048 | +0.769 | −0.031 | 218 |
| bare strings + tail normalisation | 0.2128 | 0.289 | +0.210 | +0.043 | 121 |

**The mechanism works on bare strings**: over-firing falls from 2.16x to 1.23x, density
from 218 to 121 non-zeros, and recovery swings +0.34 from actively harmful to positive.

**It does not make them competitive.** With over-firing now *comparable* to the prototype
arm (0.210 vs 0.179) and the weight gap also near zero (+0.043 vs −0.021), bare strings
still recover only 0.289 against the prototypes' 0.636. Both of the quantities §D.6.5
measures — how often a held-out entry fires, and how heavily — are calibrated, and less
than half the benefit still survives.

**So the residual deficit is neither rate nor weight: it is which documents the entry
fires on.** For bare-string entries the limitation is representational in the sense of
§D.7's final branch, while for prototypes it was calibrational. The same diagnostic
framework separates two different failure modes in two different entry representations,
which is a cleaner result than either arm winning outright.

This closes the bare-string question: the representation that retrieves best (0.2128 vs
0.1849 even after normalisation) is still the wrong one for insertion, and no further
bare-string arms are needed. It also implies a ceiling: if part of the prototypes'
remaining 0.636 → 1.0 gap is representational too, calibration alone will not reach the
0.8 pass criterion. A cheap test of that is available — `protoB`, the second disjoint
50-occurrence half already stored by `scripts/11_prototypes.py`, can be averaged with
`protoA` to make k=100 prototypes without any new encoding — and is the natural next
lever if the combined `Wh` arm falls short.
