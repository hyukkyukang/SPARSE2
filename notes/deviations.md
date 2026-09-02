# Deviations from the protocol, and why

Every item here is a place where the written protocol could not be followed
literally, or where a choice it left open had to be fixed. Nothing else deviates.

## D1 — Sense labelling without an LLM annotator (§A.1 polysemy set)
The protocol asks for occurrences "sense-labeled by an LLM with 10% manual
verification". No annotation API is available in this environment, and using the
same encoder under test to label would make the sense metric circular.

**What we do instead.** Each sense carries two disjoint word lists:
* `anchors` — the 5 entries the *scoring* uses (as in the protocol);
* `indicators` — a separate list used only to *label* an occurrence.

An occurrence of word w in a passage is labelled sense A if the passage contains
at least one sense-A indicator and no sense-B indicator (and vice versa);
ambiguous passages are dropped. Because the labels never touch the anchors, the
metric is not scored against the evidence that produced it. Manual verification
of a 10% sample is reported in `results/14_sense_verification.json`.

**Residual bias.** Labels still come from lexical context, and a contextual token
state attends to that same context. The number is therefore an upper bound on
sense separation from context *alone*; it is comparable across encoders, layers
and representations, which is all Pilot A and B use it for.

## D2 — C1 includes the top-100 of *both* dense candidates (§0.2)
C1 is specified with "the dense candidate encoder's top-100". The candidate is
not chosen until Pilot A finishes, so C1 contains both e5-base-v2's and
bge-base-en-v1.5's top-100. This only enlarges C1 and makes it *more* even-handed
towards the references, so the direction of the residual bias described in §0.2
is unchanged.

## D3 — Occurrence sampling unit (§0.3)
Prototype occurrences are sampled as distinct (passage, entry) pairs rather than
as raw token occurrences; a passage that contains the word twice contributes once,
and the entry's state from that passage is the mean over its matching word units.
Sampling is exact bottom-k with uniform random keys, so the sample is uniform
without replacement over those pairs.

## D4 — Retrieval engine for our own method
BM25 and SPLADE++ use Pyserini's prebuilt Lucene indexes (our BM25 reproduces the
published 0.187 MRR@10 exactly). Our own sparse retrieval uses `scipy.sparse` /
torch matmul over C1 as §0.7 specifies, not Lucene.

## D5 — Encoding length
Passages are truncated at 192 wordpieces and queries at 64. MS MARCO passages are
short; this affects a negligible tail and is applied identically to every system
we compare.

## D6 — Pilot A's sense constraint could not be applied as written (§A.4)
The rule is "argmax whitened cross-representation self-hit@10, with sense accuracy
≥ 0.8 required". Exactly one configuration clears 0.8 — bge layer 12, at 0.814 —
and its self-hit is 0.507 against e5 layer 9's 0.923. Applying the constraint
verbatim would hand the whole study to an encoder that is worse by 0.42 absolute
on the metric the rule is built around, on the strength of a 0.014 margin in a
metric whose own label agreement is 0.963.

**What we do instead.** Select on the primary metric (e5, layer 9), record what the
literal rule would have chosen (`literal_rule_pick` in
`results/15_pilotA_decision.json`), and let Pilot C settle it with end-to-end
retrieval: bge layer 12 is evaluated on C1 alongside the selected configuration.

## D7 — Pilot C evaluates comparators the protocol fixes earlier
Pilot A found that identity peaks at layer 9-10 while semantic organisation and
sense peak at layer 12, and that centering beats whitening on the selection metric
(H3's failure). Pilot B found R1 and R2 trading places depending on the metric.
Rather than let those tensions be settled by a threshold, Pilot C encodes C1 under
six configurations — (e5,L9,R2,whitened) with the full 36-cell grid, plus
(e5,L9,R1), (e5,L9,R3), (e5,L12,R2), (e5,L9,R2,centered) and (bge,L12,R2) at the
best cell — and reports which retrieves better. Each extra configuration costs
about three minutes of encoding, so this is cheap evidence in place of a coin flip.

## D8 — Pilot D's entry-side statistics are estimated on seen entries only
§D.1 requires μ_V, τ and b to be computed over seen entries. We also estimate the
entry-side whitening W_V on seen entries only, which is stricter than the letter of
§0.5 (which freezes one transform per representation) and removes a leakage path
the protocol's checklist implies but does not name.

## D9 — V2's staggered entry refresh
§D.3 refreshes a random 10% of entries every 100 steps by re-encoding their
contexts. We re-encode k=50 contexts per refreshed entry, as at initialisation, so
refreshed and unrefreshed entries stay comparable; the refresh is what dominates
V2's wall-clock, which is reported alongside its effectiveness.

## D10 — V2 refreshes the whole vocabulary, not a staggered 10%
§D.3 refreshes a random 10% of entries every 100 steps, on the reasoning that "a
full refresh lets the entry matrix drift stale between updates and produces a loss
discontinuity at each one". **Measured, the staggered version is the one that
breaks.** With 10% refreshed at step 250, mean non-zeros per document jumped from
39 to 519 in the next fifty steps and in-batch accuracy fell from 0.44 to 0.25; the
same thing happened in both V2 arms.

The cause is not the interval but the *heterogeneity*: after a partial refresh, 10%
of the entry matrix has been produced by the current LoRA encoder and 90% by the
encoder as it was at initialisation. Token states come from the current encoder, so
the freshly-encoded tenth is systematically better aligned with every token than the
stale nine-tenths — the refreshed group becomes a block of hubs, and the threshold
that was calibrated against a homogeneous matrix admits hundreds of them.

**What we do instead.** Refresh **all seen entries at once, every 3,000 steps**
(4 refreshes over a two-epoch run), so the entry matrix is always internally
consistent, and log the mean cosine between each entry's old and new row so the
drift the protocol worried about is measured rather than assumed. Both knobs remain
flags (`--refresh-frac`, `--refresh-every`); the partial-refresh failure is
reproducible by setting them back.

## D11 — warmup for the encoder-training arms
§D.2 gives one warmup (500 steps) for every arm, but the arms do not have the same
number of steps: V1 trains 128 queries per step (3,125 steps, so 500 warmup is 16%
of training) while V2 and V3 train 32 (12,500 steps, so 500 warmup is 4%).

At 4%, **the LoRA arm collapses**. On both seeds tried, mean non-zeros per document
fell to ~13 within 50 steps of warmup ending and cross-entropy rose from ~1.8 to
~5.1, with in-batch accuracy at 0.03. The failure is irrecoverable by construction:
`s_j = log(1 + ReLU(z_j))` is a hard gate, so an entry that switches off for the
whole batch receives no gradient and cannot come back. A trainable encoder can
switch most of the vocabulary off in a few steps; a frozen one with a
zero-initialised residual head cannot.

**What we do instead.** Warmup for the 32-query arms is set to **2,000 steps**, the
same 16% of training that V1 gets. Both LoRA arms are trained this way so the VD
comparison inside the arm is not confounded. The collapsed runs are kept as
`logs/V2_seed1_collapsed.log` and `logs/V2_seed2_collapsed.log`; the instability
itself is reported, because it says something real about the architecture: the
representation's hard gate makes encoder training fragile in a way the frozen arm
is not.
