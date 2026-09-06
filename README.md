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

**Start here:** [`SUMMARY.md`](SUMMARY.md) — what was run, what it found, what it means.
[`PROTOCOL.md`](PROTOCOL.md) is the source protocol, reproduced as written.
[`notes/pilotF.md`](notes/pilotF.md) is the protocol for the follow-up pilot that the
first round's results made necessary.

## Headline result

Recovery ratio ρ is the share of an oracle's benefit from a vocabulary entry that
survives inserting it after training. **The premise holds where the idea was motivated,
degrades with semantic novelty, and is rescued there by a training-free correction.**

| held-out split | what it simulates | ρ, base recipe | ρ, + tail normalisation |
|---|---|---|---|
| **rare** — whole bottom frequency decile | new low-frequency terms | **0.942** | 0.852 |
| **random** — 20% within each decile | scattered new terms | **0.774 / 0.801** (2 seeds) | 0.785 |
| **cluster** — a semantically new region | a new domain | 0.313 / 0.313 (2 seeds) | **0.636 / 0.646** (2 seeds) |
| **phrase** — 2,000 multi-word entries | entities, product names | −0.783 | 0.012 |

Every denominator passes the validity precondition, and every arm beats the
nearest-seen-entry alias control.

### The trade-off the study is about

The control that isolates what defining the vocabulary by text actually costs: identical
head, data and schedule, with the **seen entry rows trained as free parameters** — what
SPLADE does — while a held-out entry is still inserted as its text-defined vector.

| random split | MRR@10 | R@100 | ρ | nnz(d) |
|---|---|---|---|---|
| text-defined vocabulary | 0.2063 | 0.765 | **0.774** | 72 |
| trained entry rows | **0.2750** | 0.865 | 0.158 | 251 |

Training the vocabulary buys **+33% MRR@10** and **forfeits insertion**. That is the
trade, measured on both sides with everything else held fixed.

### What a learned vocabulary actually buys

The entry side was frozen while the token side had a learned map — an asymmetry with no
principled justification. Adding the mirror image, **one shared residual map applied to
every entry vector** (not indexed by *j*, so an inserted entry is transformed exactly like
a trained one):

| random split | MRR@10 | ρ | signed gap_r | nnz(d) |
|---|---|---|---|---|
| baseline | 0.2063 | **0.774** | −0.03 | 72 |
| + shared entry-side map | **0.2780** | 0.376 | +0.01 | 564 |
| trained rows (per-entry ceiling) | 0.2750 | 0.158 | −0.16 | 251 |

**A single shared function matches the per-entry ceiling.** So essentially *all* of a
learned vocabulary's advantage is a systematic transformation of the text-defined space;
per-entry memorisation contributes nothing measurable.

**And it costs half the extensibility with the activation gap at zero.** Inserted entries
fire at exactly the right rate and still contribute far less, so this is not
miscalibration and the tail correction does not repair it. The map is *fit on seen
entries* and reshapes the space around them. That is a **third failure mechanism**,
distinct from over-firing and from wrong-document firing.

The consequence is a choice: frozen entry side + normalisation for extensibility (ρ 0.64
on novel regions, 0.94 on rare terms, 0.185 MRR@10), or the entry-side map for
effectiveness (0.265–0.278 MRR@10, insertion actively harmful). Nothing currently does
both, and the next lever is to hold regions out of the *map's own* fitting.

### The failure mode, and the fix

On the cluster split held-out entries **over-fire by ~1.5×**. Five training-time
interventions all failed — entry dropout, cluster dropout, a meta-held-out calibration
loss, a linear head, a seen-independent entry transform — and dropout is harmful on all
three splits. The architecture has **one** scale and **one** threshold for 30,000 entries,
so a per-entry firing rate cannot be expressed at all.

The fix is a per-entry correction inside the score, computed from background statistics
against a frozen token bank, so insertion stays training-free and nothing is indexed by
*j*. A diagnostic (`scripts/48_entry_stats.py`) says which statistic to correct: an
entry's background **standard deviation** correlates 0.08 with how often it fires, its
**99.9th percentile** correlates 0.63, because the score max-pools and firing is a tail
property.

| cluster split | ρ | signed gap_r | nnz(d) |
|---|---|---|---|
| baseline | 0.313 | +0.386 | 77 |
| moment matching, after training | 0.494 | +0.229 | 71 |
| tail matching, after training | 0.539 | +0.154 | 80 |
| moment matching, trained in | 0.582 | +0.335 | 66 |
| **tail matching, trained in** | **0.636 / 0.646** | **+0.179** | **58** |

Tail beats moments and trained-in beats patched-afterwards, on both axes independently.
Documents get *sparser*, so this is not the degenerate fix where ρ rises because
everything fires more.

**The correction is conditional.** It is worth +0.32 ρ on the cluster split, ~0 on the
random split, and −0.09 on the rare split, because it removes per-entry variation that is
sometimes miscalibration and sometimes signal. Whether an entry needs it is computable at
insertion time from its own statistics.

## Real vocabulary shift

A model trained only on MS MARCO, given entries for terminology a different corpus uses,
evaluated on that corpus. Entries selected from corpus text by frequency alone, never from
queries or labels.

| corpus | this corpus's terms | random vectors | wrong-domain terms |
|---|---|---|---|
| nfcorpus, 3.6k passages | **+0.027** (sig) | 0.000 | +0.000 |
| scifact, 5.2k passages | **+0.054** (sig) | 0.000 | +0.005 |
| trec-covid, 171k passages | **−0.257** (sig) | 0.000 | +0.020 |

Random vectors change retrieval by exactly zero everywhere, so the gain is not capacity;
another corpus's terminology is non-significant everywhere, so it is not "any real words".
Two corpora gain and one is badly harmed, and R@100 moves the same way as MRR in all three
(0.747 → 0.832 on scifact; 0.074 → 0.027 on trec-covid).

**The harm is topical dominance.** The top inserted trec-covid entries are `covid`,
`coronavirus`, `wuhan`, and every query in that benchmark is about COVID: terms in most
documents *and* most queries add no discrimination. Calibration does not rescue it, and
the activation gap is ~+5 on all three corpora, so over-firing does not predict the
outcome. **Adding vocabulary helps when the terms discriminate within the target corpus
and hurts when they are corpus-defining** — a property computable at insertion time from
document frequency, with no labels.

## Two negatives that close off explanations

* **Prototype quality is not the limit.** Doubling the occurrence sample from 50 to 100
  (free — the second disjoint half is already stored) leaves recovery unchanged, 0.302 vs
  0.313. The prototype vectors move by a cosine of only 0.9975, so 50 occurrences already
  converge; sampling noise does not explain the residual gap.
* **The headline is not an artefact of the evaluation set.** Q_H is defined from each
  arm's *own* oracle, so compared arms see slightly different query sets. Re-scoring every
  cluster arm on one identical set of 1,732 queries moves ρ by at most 0.026, and the
  tail-normalisation result by 0.012. Orderings unchanged.

## Two representational boundaries

* **Bare-string entries retrieve better and generalise worse.** They beat prototypes end
  to end (0.2434 vs 0.2063) and have **negative** recovery on the cluster split. Under
  normalisation their firing rate *and* their weights reach parity with prototypes, and
  they still recover only 0.289 against 0.636 — so their residual deficit is *which
  documents they fire on*, not how often. Calibration is necessary, not sufficient.
* **Multi-word entries do not work.** A phrase prototype averaged from its constituent
  word states sits close to every document containing either word and is a hub by
  construction (background tail 0.149 vs the word median 0.109). Inserting 2,000 bigrams
  makes retrieval *worse*; normalisation makes it merely neutral.

## Three protocol expectations were falsified

| | expectation | what happened |
|---|---|---|
| **H3** | whitening raises z-gap ≥2× and cuts nnz/token ~10× | Neither. **Centering** does most of the density reduction; whitening *lowers* cross-representation self-hit below layer 11, yet is worth 0.096 vs 0.056 MRR@10 end to end. Peakiness was the wrong yardstick. |
| **H5** | bare-string entries are hubbier than contextual prototypes | **Reversed** at layer 9 (hub skew 3.6 vs 17.2); the *prototype* hub list is dominated by function words. The df–frequency half does hold. |
| §A.4 rule | select the layer by cross-representation self-hit@10 | **Anti-correlated with retrieval.** Identity peaks at layer 9; layer 12 retrieves 30% better. The rule's *tie-breaker* is what tracks effectiveness. |

Plus: the **LoRA arm collapses without vocabulary dropout** (`log(1+ReLU(·))` is a hard
gate, so an entry below threshold for a whole batch never returns), and after 3,000 LoRA
steps an entry **re-encodes to cosine 0.46** with its previous vector.

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
state over k=50 occurrences), whitened per space, k_d=128 / k_q=16 / log1p, τ set so mean
nnz ≈ 120 per document, plus **tail normalisation** where the entries to be inserted are
semantically novel. Vocabulary V = 30,000 lowercase alphabetic words with ≥100 occurrences
in the prototype pool.

Honest positioning: our best text-defined configurations reach 0.19–0.25 MRR@10 on C₁
against BM25 0.189, dense 0.356 and SPLADE++ 0.382. This is a method for **extending** a
vocabulary after training, with a measured boundary — not a replacement for learned sparse
retrieval.

## Layout

```
dvlsr/      library: paths, data, encoders, whitening, sparse rule, GPU inverted index,
            model (residual head + text-defined entry side, per-entry normalisation),
            precision (bf16 on Ampere+, fp16 elsewhere), metrics
scripts/    numbered stages, each writing a JSON artifact into results/
configs/    polysemy set (sense probe) and run configs
results/    small JSON results from the A100 run, versioned
reports/    FINDINGS.md plus a report per pilot and qualitative dumps
notes/      condensed protocol, pilotF protocol, deviations, running log
PROTOCOL.md the source protocol, verbatim
SUMMARY.md  the executive account of the study
run_all.sh          the original study in order
run_followups.sh    the rebuild + follow-up pilots, resumable by markers
```

Large artifacts (collection binary, corpus embeddings, banks, prototypes, checkpoints)
live outside the repo under `$DVLSR_DATA`.

## Reproducing

```bash
export DVLSR_DATA=/path/to/scratch
./run_all.sh          # the original study; needs a JDK for Pyserini and ~400 GB scratch
./run_followups.sh    # the rebuild and follow-up pilots; resumable, stage-selectable
```

`run_followups.sh` takes `STAGES="refs art pilotC pilotD_cluster pilotE ..."` to run a
subset, keeps per-step markers so an interrupted run resumes, and routes its output to
`$DVLSR_RESULTS_DIR` / `$DVLSR_REPORTS_DIR` so it never overwrites the original record.
Trainers and encoders wait for free GPU memory rather than crashing when several stages
share a card.

Read `notes/deviations.md` before comparing against the protocol: thirteen departures are
documented there, each with the measurement that forced it — including a manual audit of
the sense labels that found a 12% error rate, and D12, which records that the follow-up
pilots ran on different hardware with the §0 setup rebuilt from scratch.

## Status

Pilots A–E complete. Pilots F (per-entry normalisation), G (bare strings under
normalisation, normalisation on rare, the parameterized-vocabulary control), H (seed
repeats), I (shared entry-side map) and J (k=100 prototypes) complete. **62 models trained
and 45 evaluated on the rebuild**, on top of the original A100 run.

Not run: the **full-corpus confirmation** (`scripts/60_*`), so every number is on C₁ and
mildly optimistic for us. The encoder-training arms on the rebuild are capped at 3,000
steps rather than 12,500.

The experiment the results now define: train the entry-side map with **regions held out of
its own fitting**, so it cannot reshape the space around exactly the entries it saw. Also
unrun: **partial normalisation** (`--norm-alpha`), phrase prototypes built from states that
see the phrase *as a unit*, and an evaluation on **real vocabulary shift** — a different
corpus with its own terminology — rather than synthetic held-out splits of one collection.
