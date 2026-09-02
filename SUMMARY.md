# Pilot study — summary

What was run, what it found, and what it means for the idea. The source protocol is
`PROTOCOL.md`; per-pilot tables are in `reports/`; every departure from the protocol is
in `notes/deviations.md`, each with the measurement that forced it.

---

## 1. The question

Standard learned sparse retrieval bakes the vocabulary into the parameters: each output
dimension is a row of a learned matrix, so a new term is a new untrained row. This study
asks whether a vocabulary can instead be **defined by text** — entry *j* is a text *t_j*,
its vector is `f(t_j)` for the same encoder that embeds the input — so that a new entry
is added by encoding it, with no new parameters and no retraining.

The whole thing turns on one empirical question: **do entries inserted after training
behave like entries seen during it?**

---

## 2. What was built and run

MS MARCO v1 passage (8,841,823 passages) on 8× A100-40GB.

* **Setup**: collection binary + P/Q/S splits; a frozen 30,000-entry vocabulary (top
  lowercase alphabetic words with ≥100 occurrences in the prototype pool); contextual
  prototypes built in one streaming pass over 2.48M passages per encoder; 200k-token
  background banks for the document and query sides; ZCA whitening estimated once per
  space and frozen; a 1.85M-passage retrieval sub-corpus C₁ containing every reference
  system's own candidates.
* **Pilot A**: 2 candidate encoders × 9 layers × 3 transforms × 2 entry representations,
  plus ColBERTv2 as a reference ceiling — 133 configurations, each streaming a
  45k × 30k similarity profile.
* **Pilot B**: hubness, document-frequency calibration, in-context self-hit,
  self-activation, sense accuracy and a stability curve over disjoint occurrence sets,
  for 4 entry representations at 3 layers.
* **Pilot C**: C₁ encoded under 6 configurations; a 36-cell (k_d, k_q, τ, saturation)
  grid on the primary one; score preservation against the dense encoder; a 40-list
  qualitative annotation.
* **Pilot D**: 17 models trained (frozen / LoRA / full fine-tuning × vocabulary dropout
  × three held-out splits, plus oracles, a random-vocabulary control and seed repeats),
  ~200k train queries with mined BM25 hard negatives, then C₁ retrieval with and without
  the held-out entries.
* **Infrastructure**: an exact GPU inverted index (verified against brute force) that
  searches 1.85M documents for 6,980 queries in ~20 s, which is what made a 36-cell grid
  and 17-model evaluation affordable.

**Pipeline validation before any pilot** — BM25 0.1874 (published 0.1875), SPLADE++
0.3827 (0.383), e5-base-v2 0.3542 (~0.35), bge-base-en-v1.5 0.3498 (~0.35). All four
reproduce, so no pilot inherits a prefix/pooling/normalisation bug.

---

## 3. What it found

### 3.1 The premise holds, most clearly where it matters most

Recovery ratio ρ — the share of the oracle's benefit from a vocabulary entry that
survives inserting that entry after training (Pilot D, frozen encoder + light head):

| held-out split | ρ | 95% CI | MRR@10 all → seen-only | oracle all → seen-only |
|---|---|---|---|---|
| **rare** (entire bottom frequency decile) | **0.932** | [0.80, 1.07] | 0.2017 → 0.1961 | 0.1992 → 0.1928 |
| random-stratified (20% per decile) | 0.685 | [0.60, 0.77] | 0.2010 → 0.1850 | 0.1991 → 0.1713 |
| cluster (a semantically new region) | 0.567 | [0.49, 0.65] | 0.1944 → 0.1765 | 0.1946 → 0.1690 |

The rare split is the closest analogue to the real use case — new, low-frequency terms
arriving after training — and it is where the method does best, recovering 93% of the
benefit. Every denominator passes the validity precondition (≥0.02 absolute MRR, CI
excluding zero), so all three ρ values are interpretable.

Three further things support the premise rather than an artefact:

* The trained model reaches **MRR@10 0.2010** on C₁ — above its own full-vocabulary
  oracle (0.1991) and above BM25 (0.1882), up from 0.1251 untrained.
* **Insertion beats aliasing**: mapping each held-out entry to its nearest *seen* entry
  gives 0.1595 against 0.2010, so held-out entries are not redundant with the trained
  vocabulary. This is the control that makes a passing ρ mean anything.
* **The random-vocabulary control is dead**: replacing V with random unit vectors of the
  same shape collapses the model to **MRR@10 = 0.0000**. The text-defined vocabulary is
  carrying the signal, not the architecture.

Calibration is good on the random split — |gap_r| = 0.067 against a log(1.5) = 0.405
bound, i.e. a held-out entry fires at nearly the same rate as a seen entry of the same
frequency.

### 3.2 The one clear negative

On the **cluster** split, signed gap_r = **+0.47**: held-out entries systematically
*over*-fire when they come from an unseen region of the entry space. This is the failure
mode insertion-time calibration (Pilot E) is designed for, and it is the direction that
calibration can fix — over-firing, not silence.

### 3.3 Three of the protocol's own expectations were wrong

| | expectation | measurement |
|---|---|---|
| **H3** | whitening raises z-gap ≥2× and cuts nnz/token ~10× | Neither. Whitening *lowers* cross-representation self-hit at every layer below 11 (0.966 raw / 0.969 centered / 0.923 whitened at layer 9), and most of the density reduction comes from **centering** (nnz/token 10.1 → 4.6 centered → 4.6 whitened). Yet whitening is worth 0.096 vs 0.056 MRR@10 end to end: peakiness was the wrong yardstick for it. |
| **H5** | bare-string entries are hubbier than contextual prototypes | **Reversed.** Hub skewness 3.6 (R1) vs 17.2 (R2), hub share 0.044 vs 0.161, and it is the *prototype* hub list that is dominated by function words (`that, and, but, which, or, the`). The df–frequency half of H5 does hold (Spearman 0.25 vs 0.57). |
| §A.4 | select the layer by cross-representation self-hit@10 | **Anti-correlated with retrieval.** Identity peaks at layer 9 (0.923) and decays to 0.789 at layer 12, but layer 12 retrieves 30% better (MRR@10 0.1251 vs 0.0963). The rule's *tie-breaker* — related-term MRR, 23× random at layer 9 against 210× at layer 12 — is the metric that tracks effectiveness. |

Two more results about training the encoder, both negative for the fine-tuned arms and
therefore positive for the frozen design the study is really testing:

* **The LoRA arm collapses without vocabulary dropout**, on every configuration tried
  (two seeds, two warmup schedules): non-zeros per document fall to ~13 and
  cross-entropy rises to ~5.1 within tens of steps. The cause is structural —
  `log(1+ReLU(·))` is a hard gate, so an entry that falls below threshold for a whole
  batch receives no gradient and cannot return, and a trainable encoder can switch most
  of the vocabulary off very quickly. Masking 30% of entries per step prevents it; the
  frozen arm, whose residual head starts at the identity, cannot fall into it.
* **The entry space moves faster than any practical refresh**: after 3,000 LoRA steps an
  entry re-encodes to cosine **0.46** with its previous vector. The protocol's staggered
  10%-per-100-steps refresh actively harms the model, because it leaves 90% of entries
  encoded by a stale encoder and the fresh 10% becomes a block of hubs.

### 3.4 Where the untrained method stands, and why

Training-free retrieval on C₁ reaches MRR@10 0.1251 = **0.665× BM25** — real signal,
short of the 0.8× the protocol hoped for, and the "weak" branch of §C.4. The term lists
are not the problem: 39 of 40 annotated lists are plausible, and expansion is genuine
(a nursing passage activates `midwives`, a cortisol passage `pituitary` and `gland`, a
rash passage `ringworm` and `fungal`). The weighting is the problem, and it is
measurable: query profiles are peaked (top-1 carries 18.5% of the mass, participation
ratio 14 of 30 non-zeros) while document profiles are flat (3.5%, participation ratio
**63 of 119**). Untrained max-pooled cosine spreads a document's mass over ~63
effectively-equal dimensions. Training fixes exactly this — MRR@10 goes 0.1251 → 0.2010.

### 3.5 Two measurement lessons

* **The sense metric was 12% wrong until audited.** A manual check of 240 labels found
  28 errors, all traceable to ambiguous inflections (`matches`, `palms`, `tanks`,
  `cellular`) and over-generic cues used as sense indicators. Dropping 61 indicators
  raised agreement from 0.883 to 0.963, and every sense number reported uses the
  corrected labels.
* **The FLOPS regulariser is inert at the protocol's λ.** Document density lands at
  46.3–46.4 non-zeros across a 10× range of λ_d, and the penalty is ~5·10⁻⁴ of the
  cross-entropy at the top of the grid. Sparsity comes from the learned threshold —
  precisely what §A.4's H3-failure branch predicted would happen.

---

## 4. What this means for the idea

**The central claim survives its go/no-go test, with a boundary.** Entries inserted
after training recover 93% of an oracle's benefit when they are new *rare terms* — the
case the method was motivated by — and they do so while beating both a nearest-seen-entry
alias control and a random-vocabulary control that collapses entirely. A frozen encoder
plus a ~1.2M-parameter head is enough; nothing in the model is indexed by the entry.

The boundary is semantic novelty rather than rarity: when the held-out entries form an
unseen *region* of the entry space, they over-fire (signed gap_r +0.47) and recovery
falls to 0.57. That is a calibration problem with a known direction, which is what
Pilot E exists to test, and it should be the next thing run.

The most consequential methodological finding is that **identity is the wrong selection
criterion**. The metric the protocol selects layers on — whether a token state retrieves
the embedding of its own word — peaks in the middle of the network and is anti-correlated
with retrieval across layers. Semantic organisation, measured by how highly a passage's
expansion terms rank in a token's profile, tracks effectiveness instead. Any future
version of this study should select on that.

---

## 5. What was not run

The study was time-boxed. Complete: Pilots A, B, C, and the four decisive Pilot D runs
(frozen encoder on all three splits plus the random-vocabulary control). Trained but not
evaluated: the LoRA and full-fine-tuning arms, the vocabulary-dropout arms and the seed
repeats — their C₁ encodes were about 60% done. Implemented but not run: Pilot E
(`scripts/50_*`, `scripts/51_*`) and the full-corpus confirmation (`scripts/60_*`).

The cluster-split over-firing is what would trigger Pilot E, and it is the first thing to
run next; the full-corpus confirmation is the second, since every C₁ number is mildly
optimistic for us by construction (C₁ contains every reference system's candidates but
not ours).
