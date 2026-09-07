# Pilot study — summary

What was run, what it found, and what it means for the idea. The source protocol is
`PROTOCOL.md`; per-pilot tables are in `reports/`; every departure from the protocol is
in `notes/deviations.md`, each with the measurement that forced it.

The study has two parts. **Pilots A–E** ran on 8× A100-40GB and are recorded in
`results/` and `reports/`. **Pilots F–H and the follow-ups** were added afterwards on a
different machine (5× TITAN RTX 24 GB, `notes/deviations.md` D12), rebuilt the §0 setup
from the raw collection with the same seeds, and are recorded in `results_gpu10/` and
`reports_gpu10/FOLLOWUPS.md`. The rebuild reproduces the originals closely — BM25 0.1874
against 0.1874, SPLADE++ 0.3827 against 0.3827, rare-split recovery 0.942 against 0.932 —
so the two parts can be read together. **Numbers are never mixed inside a comparison**:
every follow-up arm is compared to a baseline re-run in the same rebuild.

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

## 2. The answer, in one table

Recovery ratio ρ is the share of an oracle's benefit from a vocabulary entry that
survives inserting that entry after training. 1.0 is perfect; 0 means insertion bought
nothing; negative means it did harm.

| held-out split | what it simulates | ρ, base recipe | ρ, with tail normalisation |
|---|---|---|---|
| **rare** (whole bottom frequency decile) | new low-frequency terms | **0.942** | 0.852 |
| **random-stratified** (20% per decile) | scattered new terms | **0.774** / 0.801 (2 seeds) | 0.785 |
| **cluster** (whole regions of entry space) | a semantically new domain | 0.313 | **0.636** / 0.646 (2 seeds) |
| **phrase** (2,000 multi-word entries) | entities, product names | −0.783 | 0.012 |

Read down the first column: the premise holds where the method was motivated, degrades
with semantic novelty, and fails for multi-word entries. Read across: a training-free
per-entry correction, computed at insertion time, roughly doubles recovery exactly where
the base recipe is weakest and costs a little where it is already strong.

---

## 3. What was built and run

MS MARCO v1 passage (8,841,823 passages). **68 trained models** across the two parts, plus
three BEIR corpora for the domain-shift pilots.

* **Setup**: collection binary + P/Q/S splits; a frozen 30,000-entry vocabulary; contextual
  prototypes over 2.48M passages per encoder; 200k-token background banks; ZCA whitening
  frozen per space; a retrieval sub-corpus C₁ (1.85M on the A100s, 1.47M on the rebuild).
* **Pilot A**: 2 encoders × 9 layers × 3 transforms × 2 entry representations + ColBERTv2
  — 133 configurations.
* **Pilot B**: hubness, df calibration, self-hit, self-activation, sense, stability.
* **Pilot C**: C₁ under 6 configurations, a 36-cell operating-point grid, score
  preservation, a 40-list qualitative annotation.
* **Pilot D**: the go/no-go — frozen / LoRA / full fine-tuning × vocabulary dropout ×
  three splits, with oracles and two controls.
* **Pilot E**: insertion-time calibration, four variants, training-free.
* **Pilot F** (`notes/pilotF.md`): per-entry normalisation moved *inside* the score.
* **Pilot G**: the three runs Pilot F's result made worth doing — bare strings under
  normalisation, normalisation on the rare split, and a parameterized-vocabulary control.
* **Pilot H**: seed repeats for the arms the final decision rests on.
* **Pilot I–K**: a shared learned map on the entry side, k=100 prototypes, and two attempts
  to make the entry map generalise (region holdout, displacement penalty).
* **Pilot L**: real vocabulary shift — scifact, nfcorpus, trec-covid — with three term-
  selection rules, a capacity control, a wrong-domain control, and BM25 / SPLADE++ /
  SPLADE-v3 / dense references on the same corpora.
* **Pilot M** (`notes/backbones.md`): Pilot L repeated on two more backbones,
  Octen-Embedding-0.6B and jina-embeddings-v5-text-small (Qwen3-0.6B decoders), each with
  its own probe-selected layer, prototypes, whitening and threshold — 54 domain
  evaluations in total, plus a score-decomposition diagnostic (`96`) and a label-free
  entry filter (`97`).
* **Infrastructure**: an exact GPU inverted index (verified against brute force) that
  searches C₁ for 6,980 queries in ~20 s.

**Pipeline validation before any pilot** — BM25 0.1874 (published 0.1875), SPLADE++
0.3827 (0.383), e5-base-v2 0.3542 (~0.35), bge-base-en-v1.5 0.3498 (~0.35).

---

## 4. What it found

### 4.1 The premise holds, and it is seed-stable

On the random split the frozen encoder plus a ~1.2M-parameter head recovers **0.774 and
0.801** across two seeds (spread 0.027, MRR spread 0.0005). On the rare split, the closest
analogue to the real use case, it recovers **0.942**, independently reproducing the A100
run's 0.932 on different hardware with a different random split.

Three things rule out artefacts:

* **Insertion beats aliasing.** Mapping each held-out entry to its nearest *seen* entry
  gives 0.164 against 0.206 on the random split, so held-out entries are not redundant.
* **The random-vocabulary control is dead.** Replacing V with random unit vectors of the
  same shape collapses the model to **MRR@10 = 0.0000**.
* **Calibration holds** where the split does not create a systematic gap: |gap_r| = 0.03
  on the random split, far inside the log(1.5) = 0.405 bound.

### 4.2 The cost of the core constraint, measured

The control the original protocol never specified: identical head, data and schedule, but
the **seen entry rows trained as free parameters** — what SPLADE does — while a held-out
entry is still inserted as its text-defined vector.

| random split | MRR@10 | R@100 | ρ | nnz(d) |
|---|---|---|---|---|
| text-defined vocabulary | 0.2063 | 0.765 | **0.774** | 72 |
| trained entry rows | **0.2750** | 0.865 | 0.158 | 251 |

Training the vocabulary buys **+33% MRR@10** and **destroys insertion**: recovery falls
to 0.158 and the activation gap flips sign (−0.156), so inserted entries now *under*-fire.
Once the trained rows drift into a space of their own, a text-defined row is a foreigner.

**This is the paper's central trade-off**, and it is now measured on both sides with
everything else held fixed.

### 4.3 The failure mode is calibrational, and training cannot fix it

On the cluster split, held-out entries **over-fire by ~1.5×** and recovery is 0.313.
Five training-time interventions were tried and *all five failed*:

| cluster split | ρ | signed gap_r | nnz(d) |
|---|---|---|---|
| no dropout | 0.313 / 0.313 (2 seeds) | +0.386 | 77 |
| entry-wise vocabulary dropout | 0.288 | +0.345 | 117 |
| cluster-wise vocabulary dropout | 0.273 | +0.394 | 133 |
| cluster dropout + meta-held-out calibration loss | 0.232 | +0.366 | 170 |
| linear head (capacity control) | 0.198 | +0.464 | 87 |

Dropout is harmful on **all three splits** — 0.774→0.658 on random, 0.942→0.817 on rare —
while inflating index size by half. The meta arm is the diagnostic one: it drove its
calibration loss to ~0.04 on the regions it *reserved*, and the genuinely held-out regions
over-fired exactly as before. It learned a region-specific fix because a region-specific
fix is the only kind the architecture can express: the score has **one** scale and **one**
threshold shared by 30,000 entries, since §D.1 forbids any parameter indexed by *j*.

### 4.4 The fix is in the scoring function, and it is the tail

`scripts/48_entry_stats.py` measured what actually separates an over-firing held-out
entry, within frequency decile, on the trained model:

| statistic of an entry against the token bank | held − seen | Spearman with firing rate |
|---|---|---|
| mean | +0.0037 | 0.44 |
| standard deviation | +0.0003 | **0.08** |
| 99.9th percentile | +0.0106 | **0.63** |
| maximum | +0.0200 | 0.50 |

The score max-pools over ~60 word units, so firing is a **tail** property. The standard
deviation carries almost no information about it. Pilot F therefore shifts each entry so
its background tail matches the median of *seen* entries in the same frequency decile.
Both statistics are deterministic functions of the entry vector and a frozen bank, so
**insertion stays training-free and nothing is indexed by *j***.

| cluster split | ρ | signed gap_r | nnz(d) |
|---|---|---|---|
| baseline | 0.313 | +0.386 | 77 |
| moment matching, applied after training (Pilot E `Z`) | 0.494 | +0.229 | 71 |
| tail matching, applied after training (Pilot E `DF`) | 0.539 | +0.154 | 80 |
| moment matching, trained in | 0.582 | +0.335 | 66 |
| **tail matching, trained in** | **0.636 / 0.646** (2 seeds) | **+0.179** | **58** |

Two orderings hold on both axes independently: **tail beats moments**, and **trained in
beats patched afterwards**. Only tail matching actually halves the activation gap, so its
mechanism and its effect agree. Documents get *sparser* (77 → 58), so this is not the
degenerate fix where recovery rises because everything fires more.

Four of H10's five conditions now pass on the cluster split against one for the baseline;
only ρ ≥ 0.8 does not.

### 4.5 What a learned vocabulary actually buys, decomposed

The parameterized control says training the entry rows is worth +33% MRR@10 with
L2-normalised rows, so the whole benefit is in the entry *directions*. §D.1 gives the
token side a learned map and leaves the entry side frozen — an asymmetry with no
principled justification. **Pilot I** adds the mirror image: one shared residual map
applied to every entry vector, identity at initialisation, not indexed by *j*, so a newly
inserted entry is transformed exactly like a trained one.

| | MRR@10 | ρ | signed gap_r | nnz(d) |
|---|---|---|---|---|
| random: baseline | 0.2063 | **0.774** | −0.03 | 72 |
| random: + shared entry-side map | **0.2780** | 0.376 | +0.01 | 564 |
| random: trained rows (per-entry ceiling) | 0.2750 | 0.158 | −0.16 | 251 |
| cluster: baseline | 0.1922 | 0.313 | +0.386 | 77 |
| cluster: + shared entry-side map | 0.2452 | −0.565 | +1.469 | 679 |
| cluster: + map + tail normalisation | **0.2651** | −0.309 | +1.230 | 698 |

**A single shared function matches the per-entry ceiling** (0.2780 vs 0.2750) and slightly
exceeds it. So essentially *all* of a learned vocabulary's advantage is a **systematic
transformation** of the text-defined space; per-entry memorisation contributes nothing
measurable. That is a positive result about what is learnable in principle, and the
strongest finding of the follow-up.

**And it costs half the extensibility even on the easy split** (ρ 0.774 → 0.376) — with
the activation gap at **zero**. Inserted entries fire at exactly the right rate and still
contribute far less, so this is *not* miscalibration, and adding the tail correction does
not repair it (ρ −0.565 → −0.309, gap_r barely moves). The map is *fit on seen entries*
and reshapes the space around them; an untransformed insertion no longer belongs in it.
This is the co-adaptation failure §D.7 anticipated for the encoder, appearing on the entry
side, and it is a **third mechanism**, distinct from over-firing and from wrong-document
firing.

**The consequence is a choice, not a recipe.** For extensibility: frozen entry side + tail
normalisation, ρ 0.64 on novel regions and 0.94 on rare terms, at 0.185 MRR@10. For
effectiveness: the entry-side map, 0.265–0.278 MRR@10, with insertion actively harmful.
No configuration currently does both.

### 4.6 Two negatives that close off explanations

**Prototype quality is not the limit.** Pilot J doubles the occurrence sample from 50 to
100 by averaging the two disjoint halves `scripts/11_prototypes.py` already stores — no
new encoding. Recovery is unchanged (0.302 vs 0.313), as are effectiveness and density.
The prototype vectors move by a cosine of only 0.9975, so 50 occurrences already converge.
Sampling noise does not explain the residual gap.

**The headline is not an artefact of the evaluation set.** Q_H is defined from each arm's
*own* oracle (§D.6.3), so compared arms are scored on slightly different query sets (1,732
vs 1,854 on the cluster split). Re-scoring every cluster arm on the baseline's identical
1,732 queries moves ρ by at most 0.026, and the tail-normalisation result by 0.012
(0.636 → 0.624). Orderings are unchanged. Every arm scores slightly *lower* on the shared
set, which quantifies the protocol's built-in self-selection at under three points.

### 4.7 The correction is conditional, not universal

| split | baseline ρ | normalised ρ | change | MRR cost |
|---|---|---|---|---|
| cluster | 0.313 | 0.636 | **+0.32** | −0.007 |
| random | 0.774 | 0.785 | +0.01 | −0.017 |
| rare | 0.942 | 0.852 | **−0.09** | −0.021 |

The benefit tracks how badly calibrated the split was to begin with. Normalisation removes
per-entry variation: where that variation is miscalibration it helps a lot, where there is
none it is neutral, and where it carries signal it costs. **So it should be applied per
entry, not globally** — and whether an entry needs it is computable at insertion time from
its own background statistics, with no labels. `--norm-alpha` exists for the partial case.

### 4.8 Two representational boundaries

**Bare-string entries retrieve better and generalise worse.** They beat prototypes end to
end (0.2434 vs 0.2063 random; 0.2214 vs 0.1922 cluster) and have **negative** recovery on
the cluster split (−0.048), over-firing by 2.2×. Normalisation fixes their firing rate and
their weights to parity with prototypes (gap_r 0.210 vs 0.179, gap_w +0.043 vs −0.021) —
and they still recover only **0.289** against 0.636. Both quantities §D.6.5 measures are
calibrated and less than half the benefit survives, so the residual deficit is *which
documents an entry fires on*: representational, not calibrational. The same framework
cleanly separates two different failure modes in two different entry representations.

**Multi-word entries do not work.** Inserting 2,000 Title-case bigrams (`united states`,
`social security`, `supreme court`) into the 30k word vocabulary makes retrieval *worse*
than leaving them out (0.1880 vs 0.2046), with an activation gap of **+1.70** — 5.5× too
much firing. The cause is structural: a phrase prototype is the mean of its two constituent
word states, so it sits close to every document containing *either* word and is a hub by
construction; its background tail is 0.149 against the word median of 0.109. Normalisation
takes the gap to +0.69 and recovery to 0.012 — no longer harmful, still not useful. On the
subset where phrases matter the *retrained* oracle does gain 0.042, so the phrases are
useful entries; it is post-hoc insertion that fails to capture that.

### 4.9 What the untrained method looks like, and what training is for

Training-free retrieval at layer 12 on C₁ reaches MRR@10 **0.1298 = 0.69× BM25** with
prototypes, **0.1416** with bare strings. Term lists are excellent (39 of 40 annotated
lists plausible; a nursing passage activates `midwives`, a cortisol passage `pituitary`);
the *weighting* is what is missing, and training fixes exactly that (0.1298 → 0.2063).

### 4.10 Three of the protocol's own expectations were wrong

| | expectation | what happened |
|---|---|---|
| **H3** | whitening raises z-gap ≥2× and cuts nnz/token ~10× | Neither. **Centering** does most of the density reduction, and whitening *lowers* cross-representation self-hit below layer 11. It is still worth 0.096 vs 0.056 MRR@10 end to end: peakiness was the wrong yardstick. |
| **H5** | bare-string entries are hubbier than contextual prototypes | **Reversed** at layer 9 (hub skew 3.6 vs 17.2); the *prototype* hub list is the one dominated by function words. The df–frequency half of H5 does hold. |
| §A.4 rule | select the layer by cross-representation self-hit@10 | **Anti-correlated with retrieval.** Identity peaks at layer 9, layer 12 retrieves 30% better. The rule's *tie-breaker*, related-term MRR, is what tracks effectiveness. |

Two further results about training the encoder: the **LoRA arm collapses without
vocabulary dropout** on every configuration tried, because `log(1+ReLU(·))` is a hard gate
and an entry below threshold for a whole batch receives no gradient; and after 3,000 LoRA
steps an entry **re-encodes to cosine 0.46** with its previous vector, so the entry space
moves faster than any practical refresh interval tracks.

### 4.11 Real vocabulary shift, on three backbones

Every split above is a synthetic ablation of one collection. **Pilots L and M** run the
real thing: a model trained only on MS MARCO, given entries for terminology a *different*
corpus uses and the model has never had a dimension for, then evaluated on that corpus.
Entries are chosen from corpus text by frequency alone — never from queries or relevance
labels. The baseline is the same model with its trained vocabulary only, as its own encode.
Three backbones (`notes/backbones.md`): e5-base-v2 (BERT, 12 layers, 768-d, mean pooling),
Octen-Embedding-0.6B and jina-embeddings-v5-text-small (both Qwen3-0.6B decoders, 28
layers, 1024-d, last-token pooling), each at its own probe-selected layer and threshold.

| corpus | e5-base-v2 | Octen-0.6B | jina-v5-small |
|---|---|---|---|
| nfcorpus (3.6k passages) | 0.4899 → 0.5166 **+0.027\*** | 0.4577 → 0.4875 **+0.030\*** | 0.4960 → 0.5248 **+0.029\*** |
| scifact (5.2k passages) | 0.4119 → 0.4629 **+0.051\*** | 0.4828 → 0.5272 **+0.044\*** | 0.4179 → 0.4086 −0.009 |
| trec-covid (171k passages) | 0.6942 → 0.3746 **−0.320\*** | 0.5927 → 0.7790 **+0.186\*** | 0.5965 → 0.8517 **+0.255\*** |

`*` = paired bootstrap CI over queries excludes zero. Reference systems on the same
corpora, queries, qrels and metrics:

| corpus | BM25 | SPLADE++ | SPLADE-v3 | e5 dense | Octen dense | jina dense |
|---|---|---|---|---|---|---|
| nfcorpus | 0.5086 | 0.5611 | 0.5817 | — | 0.5781 | 0.6008 |
| scifact | 0.6290 | 0.6484 | 0.6574 | — | 0.6703 | 0.7077 |
| trec-covid | 0.7676 | 0.8883 | 0.9153 | 0.9133 | 0.9300 | 0.8967 |

**Seven of nine cells gain significantly, and the controls are clean everywhere.**
2,000–3,000 random unit vectors change retrieval by *exactly* zero in all nine cells: they
never clear the firing threshold, so extra dimensions are inert without meaning and the
gain is not capacity. Another corpus's terminology — real words, real prototypes, wrong
domain — is non-significant in eight of nine. The specificity result therefore replicates
on three independently trained encoders.

**Pilot M overturned a single-backbone conclusion.** With e5 alone (Pilot L) this study
concluded that inserting corpus-defining terminology into a large single-topic corpus is
inherently harmful, and that the failure was topical dominance: `covid`, `coronavirus` and
`cov` appear in most documents *and* most queries, so they add no discrimination. Two other
backbones gain 0.186 and 0.255 on exactly that corpus with exactly those terms, both from a
*lower* baseline than e5's. The terms were never the problem; e5's treatment of them was.

**What distinguishes the failures is hub formation.** Insertion multiplies e5's document
density by 4.6–12x on every corpus (0.23–0.35 extra non-zeros per inserted entry); the two
0.6B decoders stay at 1.1–2.4x (0.012–0.066). `scripts/96_domain_diagnose.py` decomposes
the retrieval score and finds inserted entries supplying **96% of the score** of e5's
top-ranked trec-covid documents, against 70–75% for jina, drowning the trained vocabulary
that produced e5's 0.694 baseline. For jina/scifact the failure has a different shape:
inserted entries supply 49.8% of the score on *irrelevant* top-10 documents against 36.4%
on relevant ones, so their mass is actively mis-directed.

**No measured statistic predicts the outcome across all nine cells.** Query-side firing
rate separates the six cells it was derived from perfectly (gains 0.0006–0.0048, losses
0.0158 and 0.0704) but is a property the two regimes share on trec-covid, where `covid`
fires on 96% of queries for *every* backbone and is destructive for one and valuable for
two. Density multiplication fails too: e5 gains +0.051 on scifact with the highest
multiplication of any cell (10.6x). **Why one encoder turns an inserted vocabulary into
hubs and another does not is the open question Pilot M leaves**, and it is a question about
the encoder's geometry, not about the vocabulary or the corpus.

**A label-free filter repairs the failures but is not yet a rule.**
`scripts/97_qf_filter.py` drops inserted entries whose query-side firing rate exceeds a
threshold, computed from a query sample with no relevance labels:

| cell | before | after dropping high-query-firing entries |
|---|---|---|
| e5 / trec-covid | −0.320 | **+0.075** (613 of 2,974 dropped) |
| jina / scifact | −0.009 | **+0.081\*** (148 of 2,005) |
| e5 / scifact | +0.051 | **+0.061\*** (27 of 2,003) |
| jina / trec-covid | +0.255 | +0.069 (31 of 2,974) |

Three cells repaired, one badly damaged: removing 1% of jina's trec-covid entries costs
0.186 of its 0.255 gain, because there the high-query-firing entries (`covid`,
`coronavirus`, `cov`) are precisely the valuable ones. A criterion that uses both the
query and document sides is the obvious next step, and is not yet validated.

**Calibration never helps outside the synthetic splits.** The Pilot F tail correction was
applied at insertion in all nine cells: six clearly worse, three within noise, none
improved. It was developed and validated on held-out splits *within* MS MARCO, where
inserted entries genuinely over-fire relative to trained ones, and there it roughly doubled
recovery (§4.4). On real vocabulary shift the mismatch it corrects is either absent or not
what limits performance, and rescaling a non-discriminative entry only amplifies it — which
is why its worst cell (−0.088 on jina/scifact) is the one where insertion was already
failing.

**Layer choice is backbone-specific and matters more than any tuning knob.** A probe over
{6, 8, 10, 12, 16, 20, 24, 28} (D14) gives octen a U-shaped curve peaking at its *final*
layer (0.2672) and jina a plateau at layers 8–12 falling away after 16 (0.2981 at 12). A
fixed "use the last layer" rule would cost jina ~30% relative MRR; "use an early layer"
would cost octen ~20%. Octen is fine-tuned as an embedding model, so its late layers are
shaped for the pooled retrieval vector; jina is a base LM under a task adapter, whose late
layers still serve next-token prediction.

---

## 5. What this means for the idea

**The claim survives, with a measured boundary and a measured price.**

Entries inserted after training recover 94% of an oracle's benefit when they are new rare
terms, 77–80% when scattered, and — with the insertion-time correction this study added —
64% when they form a semantically novel region, up from 31%. A frozen encoder plus a
1.2M-parameter head is enough, and nothing in the model is indexed by the entry.

The price is now quantified rather than assumed: **training the vocabulary rows instead
buys 33% more MRR@10 and forfeits insertion entirely** (ρ 0.774 → 0.158). That is the
trade the paper is about.

The boundary is **semantic novelty and unit size**. Novel regions are a *calibration*
problem, and the fix is a per-entry background correction that stays training-free.
Bare-string entries and multi-word entries are *representational* problems that
calibration does not solve.

**The study separates three distinct mechanisms**, which is the contribution most likely
to outlast the specific numbers. An entry can fire **too often** (calibration fixes it),
fire the right amount **on the wrong documents** (calibration does not), or find that the
space has been **reshaped around the entries the model saw** (a learned entry side causes
it). The pair (activation gap, recovery ratio) tells them apart: a large gap with low
recovery is the first; a gap near zero with low recovery is one of the other two.

The most consequential methodological finding is unchanged from the first pass:
**identity is the wrong selection criterion**. Semantic organisation tracks effectiveness;
whether a token state retrieves the embedding of its own word does not.

**Real domain shift is where the claim is strongest, and it now replicates on three
backbones** (§4.11). Inserting a genuinely foreign vocabulary gains significantly in seven
of nine (corpus, backbone) cells, with random-vector controls at *exactly* zero in all nine
and wrong-domain controls non-significant in eight. The one systematic failure, e5 on
trec-covid, is a property of that encoder rather than of the corpus or the terms: the other
two backbones gain 0.186 and 0.255 on the same corpus with the same terms, from lower
baselines. **The open question Pilot M leaves is why one encoder turns an inserted
vocabulary into hubs and another does not** — no statistic we measured (query-side firing,
document-side firing, density multiplication, relevant/irrelevant score split) orders all
nine cells correctly, and a filter built on the most promising of them repairs three cells
and damages a fourth.

**Honest positioning.** On MS MARCO's C₁ our best text-defined configurations reach
0.19–0.25 MRR@10 against BM25 0.189, dense 0.356 and SPLADE++ 0.382. On the domain corpora
the sparse projections sit below BM25 and well below their own dense backbones — except
jina on trec-covid, where insertion lifts 0.5965 → 0.8517 and clears BM25's 0.7676. This is
a method for *extending* a vocabulary after training, with a measured boundary, not a
replacement for learned sparse retrieval.

---

## 6. What was not run

The **full-corpus confirmation** (`scripts/60_*`) was not run; every number here is on C₁,
which contains every reference system's candidates but not ours and is therefore mildly
optimistic for us. The encoder-training arms on the rebuild are **capped at 3,000 steps**
rather than 12,500 (`notes/deviations.md`); they reproduce the stability picture but are
not a full-length comparison.

The obvious next experiment, which §4.5 defines rather than leaves open: train the
entry-side map with **regions held out of its own fitting**, the way the ranking loss
already holds them out, so the map cannot reshape the space around exactly the entries it
saw. That is the only known lever on the third mechanism, and the effectiveness ceiling it
would preserve is already measured.

Also unrun: **partial normalisation** (`--norm-alpha`, for the conditional case of §4.7);
phrase prototypes built from states that see the phrase *as a unit* rather than by
averaging its parts (§4.8); and an evaluation on **real vocabulary shift** — a different
corpus with its own terminology — rather than synthetic held-out splits of one collection.
That last is the weakest point in the current evidence: every split here is an ablation of
MS MARCO, while the motivating story is new entities and evolving jargon.
