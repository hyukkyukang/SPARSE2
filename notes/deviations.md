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

## D12 — the follow-up experiments run on a different machine
The pilots ran on 8× A100-40GB with every large artifact under
`/workspace/SPARSE/dvlsr`. The follow-ups (semantic vocabulary dropout, Pilot E,
distillation, bare-string entries at layer 12, the phrase insertion test) run on
`dslab-gpu10`: 5× TITAN RTX 24 GB, where none of those artifacts exist. Three things
change, none of which touches the variable under test in any comparison:

* **Everything is rebuilt from the raw collection with the same seeds** (collection
  binary, splits, V, prototypes, banks, whitening, negatives). Numbers are therefore
  comparable *within* the rebuild, and every follow-up arm is compared to baselines
  re-run in the same rebuild, never to the A100 numbers.
* **C₁ is built without the dense candidates' top-100** (§0.2 item iv). The
  full-collection dense encode is the one setup step that does not fit the time
  budget on these cards; only the C₁ rows are encoded densely, which is all the dense
  *reference* on C₁ needs. C₁ is therefore smaller and slightly easier for every
  system, and it no longer contains the dense encoder's own hard negatives — the
  dense reference on this C₁ is mildly flattered, in the same direction the protocol
  already notes for our method.
* **fp16 autocast with dynamic loss scaling** replaces bf16 (Turing has no native
  bf16); `dvlsr/precision.py` picks the dtype from the GPU. Encodes use the same
  sharding scripts with fewer workers per card.

Only e5 is rebuilt (bge and ColBERT were Pilot A comparators and are not needed).

## D13 — twelve vocabulary entries have degenerate prototypes and are never held out
The rebuild logs what the pilot's prototype build also computed but did not act on:
seven entries have **zero** contributing occurrences in one or both prototype halves
and five more have fewer than ten in a half (`pok, saut, caf, espa, jos, fianc, beyonc,
andr, clich, speci, jalape, nestl`). All are fragments of accented words — `[A-Za-z0-9]+` cuts
"beyoncé" at the accent, while the encoder's pre-tokenizer keeps the whole word and
strips the accent, so the sampled occurrences never match a word unit. A zero
prototype whitens to the same fixed vector for every such entry, which is exactly
the block of hubs that dominated Pilot B's R2 hub list at layer 10.

**What we do instead.** These twelve entries are flagged as stopwords in `vocab.npz`
(plus a `degenerate` mask). Stopwords are never held out by any §D.4 split, so a
degenerate entry can never be *inserted* and masquerade as an over-firing held-out
entry; they stay in V, so |V| and every seen-side statistic are unchanged. The list
is in `results_gpu10/01_vocab_degenerate.json`. The pilot's A100 numbers were
computed with these entries eligible for holding out.

**Correction (2026-09-04).** Flagging those twelve entries as stopwords had a side effect
on the rare split, which holds out an entire frequency decile: the decile's only remaining
"seen" entry was `nestl`, a degenerate one, whose activation rate is ~1e-5. The
within-decile activation comparison of §D.6.5 then reported signed gap_r = +3.37 for
`V1norm_rare`, an artefact of a single near-dead entry. `activation_stats` now excludes
degenerate entries and requires 20 usable seen entries in a decile before that decile
contributes, so the rare split reports the gap as unmeasurable — which is what the A100
run did, and what the split's construction implies.

## D14 — Pilot M backbones choose their layer by a retrieval probe, not by §A.4

**What the protocol says.** §A.4 selects the hidden layer by cross-representation
self-hit@10, with related-term MRR as a tie-breaker.

**What we did.** For the two Qwen3-0.6B backbones of `notes/backbones.md` the layer is
chosen by `scripts/95_layer_probe.py`: training-free retrieval (the §C rule) at five
candidate layers {12, 16, 20, 24, 28} on a probe corpus of every dev-small positive plus
100k random C1 passages, picking the best MRR@10; tau_d/tau_q come from the §0.6 samples
at that layer.

**Why.** For e5 the §A.4 rule picked layer 9 and Pilot C showed layer 12 retrieves 30%
better; identity and retrieval are anti-correlated across layers (§A findings). A 28-layer
causal decoder has no "last BERT layer" analogue and its final layer is shaped for
next-token prediction, so neither the rule nor a guess is safe. The probe applies the
criterion that actually chose e5's layer at ~7% of C1's encoding cost.

## D15 — word-unit boundaries are decided per piece, and surfaces are stripped

**What changed.** `Encoder._word_units` kept a unit iff its first character lay at or
after the prompt (`c0 >= plen`). Byte-level BPE offsets include the space preceding a
word, so under that rule the first text word after a prompt was silently dropped
(`"Document: The"` tokenises to `ĠThe` spanning (9,13) with plen = 10). Qwen's
pre-tokenizer also glues a leading punctuation mark to the next word, so Octen's
`"…\nQuery:what"` yields one word whose first piece `:` is prompt. Now a *piece* is text
iff it ends after the prompt, a unit is text iff it has such a piece, the unit's state
averages only those pieces, and its surface is taken from the prompt boundary and
stripped.

Byte-level BPE backbones (`trim_punct=True`) additionally trim leading/trailing
non-alphanumerics from a unit's surface and drop punctuation-only pieces from its mean,
because Qwen's pre-tokenizer glues one leading non-letter onto a word (`brown-fox` →
`brown`, `-fox`; `sars-cov-2` → `sars`, `-cov`, `-`, `2`), which would otherwise never
match a vocabulary word. This part is *not* applied to e5: BERT keeps symbol characters
(°, €, ™) inside words and `str.isalnum` rejects them, so it would alter 0.1% of e5 units.

**Proof the prompt rule changes nothing for e5.** WordPiece offsets exclude spaces and no
piece straddles the boundary, so the rule is equivalent. Verified: 500 P passages and 200 dev
queries through the old and new code give identical unit lists, rows and bit-identical
states (28,234 document units, 1,048 query units).

## D16 — single-layer encoding truncates the frozen stack

`Encoder.truncate_to(L)` drops every transformer layer above the one in use and replaces
the stack's final norm by the identity (HF applies that norm to the *last* entry of
`hidden_states`, and all artifacts were built from the un-normed layer-L state of the full
model). `hidden_states[L]` depends on layers 1..L only, so this is exact: verified
`max |diff| = 0` against the full stack for both backbones at layer 20. Used only where no
pooled embedding is needed (frozen training, corpus encoding, domain evaluation); the
prototype/bank builders and the dense reference keep the full stack. Saves (28−L)/28 of
the forward pass.

## D17 — Octen inputs follow the model card's reference code, not sentence-transformers 6

The Octen model card gives two usages. Its HuggingFace snippet is the Qwen3-Embedding
format: an `Instruct: …\nQuery:` prefix before queries, bare documents, the `<|endoftext|>`
the tokenizer appends as the pooled last token. Our `Encoder` reproduces that snippet to
cosine 1.0000 on 32 passages and 16 queries (left-padded reference vs our right padding).
sentence-transformers 6.0 wraps every input in the repo's chat template
(`<|im_start|>user\n…<|im_end|>\n`) and pools the trailing newline instead of the end
token — a library behaviour that differs from the authors' code (cosine 0.80–0.86 against
it), so it is not what we match. The card's sentence-transformers document prompt `" "` is
stripped by that library before tokenisation, i.e. it is a no-op there; we use `""` so the
first document token is `The`, not `ĠThe`, as in the reference snippet.

jina-embeddings-v5's retrieval LoRA is merged into the weights at load: pooled outputs
identical (cosine 1.0000), unit states 0.9999 in fp16, throughput 102 → 183 passages/s.

## D18 — two posting stores and a per-query gate, added at evaluation time

The protocol scores every document from one per-document store of at most 1,024 entries.
Under e5 the inserted entries fill that store (51.6% of trec-covid documents, 35.0% of
scifact documents reach the cap) and evict trained entries, which no query-side rule can
restore. `scripts/92_domain_eval.py --two-store` scores instead from two stores, trained
(the baseline encode) and inserted (the inserted part of the extended encode), as an index
that keeps them separately would; entry ids are disjoint so the score is the exact sum.
The domain table (`reports_gpu10/DOMAIN.md`) keeps the one-store numbers; the gate report
(`reports_gpu10/QUERY_GATE.md`) shows both.

The per-query gate (`--qgate K --qgate-mode all|dfdrop|topk`) is a query-time rule the
protocol did not have. It obeys the study's constraint that nothing about the test queries
is known in advance: it uses the count of inserted entries firing on the query being
answered and, for `dfdrop`, the fraction of indexed documents each inserted entry fires on,
fixed at indexing time. K=20 and the 5% document-firing threshold were set before any run;
K=10/30/50 were run afterwards for sensitivity. The query-sample filters of
`scripts/97_qf_filter.py` do not obey the constraint and are reported as an upper bound.

## D19 — text-defined word units for the main-experiment backbones

The protocol's word unit is the tokenizer's pre-tokenizer word (§0.4), which BERT and Qwen
split at punctuation but SentencePiece models (XLM-R in arctic-embed, Gemma) split only at
whitespace: under them `covid-19` or `brown-fox` is one unit whose surface matches no
vocabulary entry. `dvlsr/encoders.py` `unit_mode="alnum"` defines a unit instead as a
maximal run of Unicode alphanumeric characters in the text after the prompt; a piece
contributes to every unit its character span overlaps. On Qwen this reproduces the D15
rule on shared units (state cosine 0.9985 over 5,074 units) and differs only where Qwen's
pre-tokenizer splits letters from digits (`cd33`, `s100a9` stay whole). e5's frozen pilot
artifacts keep the legacy rule; the three main-experiment backbones use the new one.

## D20 — arctic-embed-m-v2.0 under transformers 5

Its remote code (Alibaba GTE) builds position ids and rotary caches as non-persistent
buffers in `__init__`; transformers 5 constructs models on the meta device, so they load as
uninitialised memory and the first forward indexes with garbage. `rematerialize_gte_buffers`
recomputes them from the module's own attributes after loading, and the memory-efficient
attention and unpadding flags (which assert xformers) are switched off in the config. With
both, our CLS-pooled embeddings match the model's sentence-transformers pipeline (given the
same fix) to cosine 1.0000 in fp32 and fp16.
