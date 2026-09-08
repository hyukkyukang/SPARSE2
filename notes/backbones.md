# Pilot M — the domain-shift experiment on two more backbones

**Goal.** Reproduce the whole domain-shift table (`reports_gpu10/DOMAIN.md`) with the
sparse projection built on two different frozen encoders, so the finding is not an
artefact of e5-base-v2:

| key | model | base | layers | hidden | pooling | prompts |
|---|---|---|---|---|---|---|
| `e5` (done) | intfloat/e5-base-v2 | BERT | 12 | 768 | mean | `query: ` / `passage: ` |
| `octen` | Octen/Octen-Embedding-0.6B | Qwen3-Embedding-0.6B | 28 | 1024 | last token | instruction / `" "` |
| `jina5s` | jinaai/jina-embeddings-v5-text-small | Qwen3-0.6B-Base + LoRA (task adapters) | 28 | 1024 | last token | `Query: ` / `Document: ` |

Both new models are **decoder-only** (causal attention), use **byte-level BPE** with the
space attached to the following token, ship in bf16 (Turing has no bf16; both verified
finite at every layer in fp16), and are ~5.5x the parameters of e5-base.

## What "the same experiment" requires per backbone

```
10_probes_banks  --encoder X          probes + token banks B_H, B_Q at candidate layers
10_probes_banks  --encoder X --aux    tau-fitting samples (5k S passages, dev queries)
11_prototypes    --encoder X          R2 prototypes of the SAME 30k vocabulary (k=50, from P)
12_r1            --encoder X          bare-string embeddings (13_whitening expects them)
13_whitening     --encoder X          ZCA per space per candidate layer
95_layer_probe   --encoder X   NEW    pick ONE layer by training-free retrieval; fit tau
42_train  --name V1oracle_X --encoder X --layer L --tau t --oracle   (frozen enc + head)
91_domain_vocab  --encoder X          domain prototypes (3 corpora x 3 selection rules)
92_domain_eval   --name-model V1oracle_X   18 cells per backbone, corrected baseline
94_domain_baselines --encoder X       the backbone's own dense retrieval as a reference row
93_domain_report                      one table per corpus, a block per backbone
```

Everything upstream of `10_` (collection, splits, vocabulary, C1, negatives, corpora) is
encoder-independent and already exists.

## Single layer, chosen the way that actually worked

The user wants the single-layer method. e5 uses layer 12 (its last), chosen by Pilot C's
end-to-end retrieval after the §A.4 identity rule picked the wrong layer. A 28-layer
decoder has no obvious analogue of "layer 12", and the last layer of a causal LM is
shaped for next-token prediction, so guessing is unsafe. `95_layer_probe.py` stores five
candidates `{12, 16, 20, 24, 28}` in the prototypes/banks and picks by training-free
MRR@10 on a ~107k-passage probe corpus (every dev-small positive + 100k random C1
passages) — the same criterion that chose e5's layer, at a fraction of C1's cost.
Deviation **D14** in `notes/deviations.md`.

## Code changes

### `dvlsr/paths.py`
* `ENCODERS["octen"]`, `ENCODERS["jina5s"]` with `pooling="last"`, `dim=1024`,
  `layers=[12,16,20,24,28]`, `trust_remote_code`, `adapter="retrieval"` (jina),
  `fp16_weights=True`.
* `layers_for(enc)` reads `ENCODERS[enc].get("layers", LAYERS)`.

### `dvlsr/encoders.py`
* `Encoder.__init__`: `trust_remote_code`, adapter selection (`set_adapter(["retrieval"])`).
* `_pool`: `"last"` = hidden state at `attention_mask.sum(1) - 1` (right padding; both
  tokenizers pad right; Octen appends `<|endoftext|>` so its last token is the EOS the
  model pools, Jina pools the last text token — both match the models' own `encode`).
* **Word-unit boundary fix.** BPE offsets include the leading space, so the first word
  after a prompt starts one character *inside* the prompt and the old rule
  `c0 >= plen` dropped it (`Document: The` → `ĠThe` spans (9,13) with plen=10). New rule:
  keep a unit iff it *ends* after the prompt (`c1 > plen`), and strip whitespace from the
  surface string before vocabulary lookup. For WordPiece (e5) this is provably identical
  (offsets exclude spaces; prompt words end at or before plen) — verified by a regression
  test that the e5 word units of 500 passages are unchanged.
* `truncate_to(L)`: drop decoder layers above the one in use. A causal model's layer-L
  state does not depend on layers > L, so for single-layer encoding this is exact and
  saves (28−L)/28 of the compute. Verified `allclose` against the full model. Only used
  when no pooled embedding is needed (training with no teacher, encoding, domain eval).

### Dimension and layer-index assumptions (bugs for any non-e5 encoder)
* `Head(768, 768)` / `EntryHead(768, 768)` / `zeros(nV, 768)` in `42_train.py`,
  `43_encode_eval.py`, `82_phrase_ckpt.py` → take the dimension from the entry matrix.
* `paths.LAYERS.index(layer)` in `42_train.py`, `30_encode_c1.py`, `41_vocab_splits.py`,
  `82_phrase_ckpt.py`, `92_domain_eval.py` (x2) → `paths.layers_for(enc).index(layer)`.
* `42_train.py` loads the frozen encoder in fp32 and relies on autocast; with
  `output_hidden_states=True` a 0.6B model's 29 fp32 layer outputs for a 512-passage
  chunk is ~12 GB. Frozen (V1) arms load `fp16_weights` encoders in fp16 and use
  `--enc-chunk 128`; truncation removes the unused layers anyway.
* `92_domain_eval.py` `encode()` runs the encoder in fp32 with no autocast → wrap in
  `precision.autocast()`.
* `91_domain_vocab.py` writes the encoder-independent `vocab_/vsplits_` files with an
  encoder-*specific* "has a prototype" mask baked into `domain`/`is_stop`. Second encoder
  would overwrite. Fix: write those files only if absent (assert identical words), and
  have `92` define the inserted set as `is_domain & (cntA_of_this_encoder > 0)`.
* `94_domain_baselines.py`: dense key becomes `dense_{enc}` (with `dense` kept as the e5
  alias); `93_domain_report.py` renders one reference row per backbone and one block
  (baseline + 6 variants) per backbone under each corpus.

### New: `scripts/95_layer_probe.py`
For each candidate layer: tau from the tauS sample (target 120 nnz, as §0.6), encode the
probe corpus and dev queries with the training-free rule (whitened cosine, max over word
units, `ReLU(p − tau)`, log1p, top-128/16), exact chunked scoring, MRR@10 / R@100.
Writes `results/95_layer_probe_{enc}.json` with the chosen layer and its tau — the two
numbers `42_train` needs.

## Measured before launch (TITAN RTX, fp16, 192-token passages, batch 64)

| | octen | jina5s (LoRA merged) |
|---|---|---|
| full 28-layer stack | 180 passages/s | 183 passages/s |
| truncated to layer 20 | 239 passages/s | ~240 passages/s |
| fp16 vs fp32, pooled and every candidate layer | cosine ≥ 0.9999 | cosine ≥ 0.9999 |
| vs the model card's reference pipeline | 1.0000 (README code, D17) | 0.9997 (sentence-transformers) |
| truncated vs full at layer 20 | max abs diff 0 | max abs diff 0 |
| e5 word units under the new boundary rule | bit-identical (28,234 + 1,048 units) | |

## Compute plan (4 usable cards; card 1 belongs to another user)

| stage | per backbone | notes |
|---|---|---|
| banks, aux, R1, whitening | ~10 min | one card |
| prototypes (8 shards, 2/card, bs 64) | ~20–30 min | ~2M passages at ~250/s/shard |
| layer probe (5 layers) | ~15 min | one card per layer in parallel |
| **training V1oracle** (3,124 steps) | **benchmark first** | e5: 3.6 s/step; expect 4–6x → cap at 3,124 steps or 1,500 if >10 s/step |
| domain vocab + 18 evals + baselines | ~1.5 h | trec-covid is the long pole |

Each backbone owns a pair of cards (`run_backbone.sh <enc> <main> <aux>`; jina5s 0/3,
octen 4/2) and never waits on the other: prototypes build on both cards, training runs
on the main card while the domain vocabularies and the dense reference build on the aux
card, and the final evaluations split trec-covid (main) from the small corpora (aux).

## Review findings applied before launch (independent code review, 2026-09-07)

* `91_domain_vocab.py` still hard-coded 768 → `enc.dim` (would have crashed every
  backbone after training).
* `92_domain_eval.py` took the inserted set from the split file's `domain`, which older
  files wrote as e5's own has-a-prototype mask → now `is_domain & (this encoder's cntA > 0)`.
* Qwen's pre-tokenizer glues a leading non-letter onto the next word, so hyphenated words
  never matched the vocabulary and some trained entries would have been garbage rows →
  piece-level punctuation handling, gated to the BPE backbones (D15).
* The calibrated (`_norm`) row also recalibrated the 30k trained entries; the correction
  now applies to the inserted rows only, and the e5 rows are redone under that definition.
* Second review: the committed driver concatenated `$REPO/` onto already-absolute
  results/log paths → both now come from `dvlsr.paths`; the e5 rows had mixed two baseline
  encodes (before/after mixed precision was added to the domain evaluator) → every e5
  cell re-run under one cache keyed on checkpoint mtime *and* autocast dtype; the
  reference-row writer is atomic.
* Two drivers writing `94_domain_baselines_<corpus>.json` → merge-before-save plus a
  per-corpus lock; baseline cache keyed on the checkpoint's mtime and written atomically;
  `HF_HUB_OFFLINE=1` so the hub is not a failure mode mid-run.

## Validation before launch
1. e5 word-unit regression (500 passages, identical units and states).
2. Truncated-vs-full hidden state at the chosen layer, both models, `allclose`.
3. `encode_pooled` for both models reproduces the models' own `encode` on 8 texts
   (cosine > 0.999) — proves prompts, pooling and adapters are wired as the authors intend.
4. A 3-step training smoke test per backbone (loss finite, checkpoint loads in `43`).
5. Independent code review (bugs + optimisation) before the long runs start.

## Report
`reports_gpu10/DOMAIN.md` gains, under every corpus, one italic reference row per dense
backbone and one block per sparse backbone. `notes/deviations.md` D14 records the layer
choice, D15 the word-unit boundary fix and its e5 regression proof.

## Running on another server

The pipeline is self-contained: code in this repository, large artifacts under
`$DVLSR_DATA`. Nothing encoder-specific from this machine is needed.

1. **Environment.** Python 3.10+, `torch` 2.11 (CUDA), `transformers` 5.16, `peft` 0.20,
   `numpy`, `scipy`; `sentence-transformers` only for the validation script. Copy
   `env.example.sh` to `env.sh` and set `DVLSR_DATA`. Models download on first use
   (Octen 1.2 GB, jina-v5 1.2 GB + adapters, e5, and for the reference rows SPLADE++ and
   the gated naver/splade-v3, which needs `HF_TOKEN`); set `HF_HUB_OFFLINE=1` afterwards.
2. **Data.** `rsync -a --files-from=notes/backbones_manifest.txt <this machine>:$DVLSR_DATA/ $DVLSR_DATA/`
   — about 7 GB: the MS MARCO collection binary and splits, the 30k vocabulary, the probe
   plan, training negatives, and the three domain corpora (or rebuild those with
   `scripts/90_domain_data.py`, which downloads BEIR).
3. **Run.** One instance per backbone on disjoint card pairs, e.g. on an 8-GPU box
   `./run_backbone.sh octen 0 1` and `./run_backbone.sh jina5s 2 3`. Each is resumable by
   the markers under `$DVLSR_DATA/markers/bb_<enc>/`; per-step logs land in
   `$DVLSR_LOGS_DIR`. `BB_PROTO_SHARDS=8` spreads the prototype build over more cards if
   `<main>,<aux>` is widened to a list.
4. **Results.** `results_gpu10/95_layer_probe_<enc>.json` (chosen layer + tau),
   `results_gpu10/92_domain_<corpus>_V1oracle_<enc>*.json` (18 per backbone),
   `results_gpu10/94_domain_baselines_<corpus>.json` (gains a `dense_<enc>` key),
   `reports_gpu10/DOMAIN.md` (regenerate with `scripts/93_domain_report.py`). Commit the
   JSONs and the report; checkpoints stay under `$DVLSR_DATA/ckpt/V1oracle_<enc>`.

Expected durations on an A100-class card (≈2.5x the TITAN RTX numbers above): banks and
threshold samples ~5 min, prototypes ~30 min on two cards, layer probe ~8 min, training
~2 h, domain vocabularies ~20 min (in parallel with training), evaluations ~40 min split
over two cards — roughly 3.5 h per backbone, both in parallel.

The e5 rows of the table are final in this repository and need no recomputation.

## Layer choice: the two backbones disagree, and the reason is instructive

`scripts/95_layer_probe.py` on a 106,921-passage probe corpus (every dev-small positive +
100k random C1 passages), training-free retrieval, MRR@10 per candidate layer:

| layer | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|
| octen | 0.2203 | 0.2233 | 0.2145 | 0.2184 | 0.2041 | 0.1947 | 0.2570 | **0.2672** |
| jina5s | 0.2905 | 0.3044 | 0.2914 | **0.2981** | 0.2864 | 0.2552 | 0.2332 | 0.2063 |

Octen is U-shaped and peaks at its last layer; jina5s decreases monotonically and peaks at
the shallowest candidate. Both winners sit at an edge of the sweep, and both curves dip in
the middle. The plausible reason is what each model was tuned for: Octen is fine-tuned as
an embedding model, so its late layers are shaped for the pooled retrieval vector, while
jina5s is a base LM with a retrieval LoRA, whose late layers still serve next-token
prediction and whose term-level semantics sit early. This is the §A finding — identity and
retrieval prefer different depths — reappearing *across models* rather than across layers
of one model, and it is why a fixed "use the last layer" rule would have cost jina5s ~30%
relative MRR.

Because jina5s peaked at the edge, the sweep was extended to layers {6, 8, 10} under the
separate encoder keys `octen_lo`/`jina5s_lo` (identical models and prompts, different
artifact names, so the extension runs beside a live pipeline). Result for jina5s:

| layer | 6 | 8 | 10 | **12** | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|
| MRR@10 | 0.2905 | 0.3044 | 0.2914 | **0.2981** | 0.2864 | 0.2552 | 0.2332 | 0.2063 |

The curve is a broad plateau over layers 8–12 and falls away after 16. Layer 8 leads layer
12 by 0.006, inside the noise of a single probe and below the 0.01 margin set for
retraining, so layer 12 stands and the jina5s results below are not a layer-selection
artefact. Octen's peak is at its maximum depth and cannot be extended.

## Term selection and entry filters, all variants on six cells

Change in MRR@10 against the backbone's own baseline; `*` = paired bootstrap CI excludes
zero; entries inserted in parentheses. Lexical rules re-rank the candidate list; filters
start from the frequency-ranked list and drop entries by how the MODEL fires them.

| variant | e5 / nfcorpus | jina / nfcorpus | e5 / scifact | jina / scifact | e5 / trec-covid | jina / trec-covid |
|---|---|---|---|---|---|---|
| frequency ranking | +0.027* (957) | +0.029* (961) | +0.051* (2003) | -0.009 (2005) | -0.320* (2974) | +0.255* (2974) |
| tf-idf | +0.027* (957) | +0.029* (961) | +0.051* (2003) | -0.011 (2005) | -0.238* (2976) | +0.240* (2976) |
| df ceiling 10% | +0.027* (957) | +0.029* (961) | +0.051* (2003) | -0.009 (2005) | -0.278* (2969) | +0.217* (2969) |
| pure IDF | +0.027* (957) | +0.029* (961) | +0.052* (2003) | -0.010 (2005) | -0.117 (2774) | +0.109* (2797) |
| query-firing filter (labels-free, needs a query sample) | +0.027* (957) | +0.029* (961) | +0.061* (1976) | +0.081* (1857) | +0.074 (2361) | +0.069 (2943) |
| model document-firing filter, fixed 10% | +0.012* (346) | +0.029* (894) | +0.018* (571) | +0.071* (1589) | +0.079 (1308) | +0.042 (2792) |
| model document-firing filter, 99th pct of trained entries | +0.013* (262) | +0.027* (864) | +0.014* (332) | +0.057* (1410) | +0.021 (475) | +0.083* (2560) |

**Lexical selection is inert on the small corpora** (fewer candidates clear the occurrence
floor than the cap, so every rule takes the same set) and mirror-imaged on trec-covid:
rarer-is-better recovers 0.20 of e5's 0.32 loss and costs jina 0.15 of its 0.26 gain. The
hub entries are lexically *rare* — `predisposes` occurs in 0.04% of trec-covid documents and
fires on 84% of them under e5 (rank correlation of text frequency with model firing: +0.05)
— so IDF weighting selects them rather than removing them.

**Both failures are repairable, by any model-based filter**: jina/scifact −0.009 → +0.057 to
+0.081; e5/trec-covid −0.320 → +0.021 to +0.079. **No filter is safe on every cell.** Each
costs jina/trec-covid two thirds or more of its +0.255, because the entries it removes there
(`covid`, `coronavirus`, `cov`: ~50% document firing, ~96% query firing under *both* backbones)
carry the gain for jina and the loss for e5 with identical statistics. The document-firing
filters also over-prune e5 on the small corpora (keeping 260–570 of 2,003), because under e5
*every* inserted entry fires broadly, the helpful ones included (`microdissection`: 63%).

The per-entry approach therefore has a ceiling that this table locates: the same entry with
the same firing statistics is essential under one backbone and destructive under another,
and what differs is the population it arrives with. Whether an inserted vocabulary drowns a
backbone is a property of the whole set on that backbone, and the label-free quantity that
reflects it — the share of retrieval score the inserted entries take on a corpus sample — is
measurable at insertion time but was not tested here.

## Can the prototypes' geometry select entries per backbone? (scripts/98_geometry_analysis.py)

Ground truth per inserted entry from the score decomposition (`96`): the entry is *harmful*
if the share of its score mass landing on relevant documents is below the corpus's base rate
of relevant pairs (a lift < 1). Candidates computable from the new prototypes and the frozen
artifacts alone, in the backbone's own space: raw prototype norm, cosine to the trained
centroid (raw and whitened), occurrence count, cosine to the nearest trained entry, and the
mean / 99.9th-percentile similarity against the document bank and the query bank (Pilot F's
statistics, both sides, absolute and relative to the trained entries). Full table in
`reports_gpu10/GEOMETRY.txt`.

**Per-entry geometry does not predict per-entry harm.** Over nine cells and eleven
statistics, none is consistent in sign: the raw norm scores AUC 0.74 on jina/nfcorpus and
0.25 on e5/nfcorpus; the whitened centroid cosine 0.73 on e5/nfcorpus and 0.42 on
jina/nfcorpus. The bank statistics sit within 0.05 of chance on six of nine cells and reach
0.63–0.76 only on the two decoder trec-covid cells, which have 5 and 9 harmful entries. The
firing statistics that need the corpus are no better as per-entry predictors (document
firing: 0.29–0.75).

**Per-entry harm is the wrong target.** Query-side firing rate has AUC *below* 0.5 in all
nine cells (0.17–0.46): the entries that fire on many queries are individually *more*
concentrated on relevant documents than chance. Yet removing exactly those entries turns
jina/scifact from −0.009 to +0.081 and e5/trec-covid from −0.320 to +0.075. Each is mildly
informative alone; together they add a large near-uniform component that crowds out the
trained signal. No label assigned to single entries can express a crowding effect, which is
the ceiling every per-entry filter in the table above ran into.

**Aggregate geometry is a within-backbone warning signal.** Mean pairwise cosine of the
inserted set (an equal-sized random subset of trained entries is ~0.000 everywhere):

| backbone | nfcorpus | scifact | trec-covid | failing cell |
|---|---|---|---|---|
| e5 | 0.257 | 0.274 | **0.318** | trec-covid (−0.320) |
| jina5s | 0.139 | 0.151 | **0.174** | scifact (−0.009) |
| octen | 0.043 | 0.059 | 0.047 | none |

In both backbones that have a failure, the failing cell is that backbone's most internally
correlated inserted vocabulary. The scale is backbone-specific — e5's *lowest* (0.257) is
above jina's highest and five times octen's — and the ordering among the gains is not
monotonic, so this is a diagnostic to compare a new vocabulary against what the same
backbone has tolerated before, not a threshold and not a predictor of magnitude.

**The gated query-firing filter is the only procedure that reaches the best-known result
on every cell.** Filter (drop entries with query-firing > 0.05) only when the inserted set's
mean query-firing exceeds 0.01. On the six cells it was derived from it selects the
best-known variant on five and is within 0.01 on the sixth. Held out on octen's three cells
(mean query-firing 0.0005, 0.0036, 0.0036 → do not filter) it preserves all three gains,
+0.030, +0.044, +0.186, where the ungated filter would have cut the last to +0.040.

## Query-time gate: the deployment-compliant rule (`scripts/92_domain_eval.py --two-store --qgate`)

Full tables: `reports_gpu10/QUERY_GATE.md` (`scripts/99_query_gate_report.py`).

**The constraint.** Test queries are unknown until each one arrives. A rule may use the
single query being answered and anything fixed at indexing time — the corpus, the model,
statistics of the index — and never statistics over a set of queries. The query-firing
filter and its gate in the previous sections use a *sample of queries*, which on these
benchmarks is the test set; they are an upper bound on what query-side information can do,
not a method. Everything in this section obeys the constraint.

**The rule.** At indexing time keep two posting stores, trained and inserted
(`--two-store`), so an inserted entry can never displace a trained one. At query time count
the inserted entries that fire on the query. If more than K fire, act on the inserted side
of *that query only*, one of three ways: answer from the trained store alone (`all`); drop
the inserted entries that fire on more than 5% of the indexed documents, an index statistic
computed with no query (`dfdrop`); keep the K strongest (`topk`). K=20 and the 5% threshold
were fixed before any run; K=10/30/50 were run afterwards for sensitivity, so K=50 is post
hoc. Octen's cells were never used in choosing either value.

**Why two stores.** The evaluation store keeps 1,024 entries per document. Under e5 the
inserted entries fill it — 51.6% of trec-covid documents and 35.0% of scifact documents hit
the cap — and evict trained entries: 17.8 of a document's 61.2 trained entries on
trec-covid, 5.9 of 72.3 on scifact. No query-side rule can bring an evicted entry back, so
on one store the gate that drops every inserted entry from 49 of e5's 50 covid queries still
leaves −0.066; on two stores the same gate gives exactly 0.000. Two stores with no gate at
all already move e5/trec-covid from −0.320 to −0.273 and e5/scifact from +0.051 to +0.055.
The decoders never reach the cap on any corpus, so for them one store and two stores are
identical to four decimals.

**Results** (change in MRR@10 against the backbone's baseline; `*` CI excludes zero;
`(Nq)` queries on which the gate acted; upper bound = query-sample gate of the previous
section):

| corpus | backbone | insert all, one store | two stores, no gate | **two stores + df-drop, K=20** | two stores + fall back, K=50 | two stores + top-k, K=20 | upper bound |
|---|---|---|---|---|---|---|---|
| nfcorpus | e5 | +0.027* | +0.027* | **+0.027*** (1q) | +0.027* (0q) | +0.027* (1q) | +0.027* |
| nfcorpus | Octen | +0.030* | +0.030* | **+0.030*** (0q) | +0.030* (0q) | +0.030* (0q) | +0.030* |
| nfcorpus | Jina | +0.029* | +0.029* | **+0.029*** (3q) | +0.029* (0q) | +0.029* (3q) | +0.029* |
| scifact | e5 | +0.051* | +0.055* | **+0.037*** (43q) | +0.048* (12q) | +0.057* (43q) | +0.051* |
| scifact | Octen | +0.044* | +0.044* | **+0.041*** (23q) | +0.044* (0q) | +0.046* (23q) | +0.044* |
| scifact | Jina | −0.009 | −0.009 | **+0.045*** (126q) | +0.014 (80q) | +0.021 (126q) | +0.081* |
| trec-covid | e5 | −0.320* | −0.273* | **+0.020** (49q) | 0.000 (49q) | +0.035 (49q) | +0.074 |
| trec-covid | Octen | +0.186* | +0.186* | **+0.196*** (4q) | +0.186* (0q) | +0.186* (4q) | +0.186* |
| trec-covid | Jina | +0.255* | +0.255* | **+0.255*** (6q) | +0.255* (0q) | +0.255* (6q) | +0.255* |

**What it establishes.**

1. **The gate sees the explosion-type failure with a wide margin.** On trec-covid e5 fires a
   median of 238 inserted entries per query; the decoders' maxima on the same corpus are 33
   and 48, and their scifact medians are 2–11. Any K between 50 and 200 gates every one of
   e5's covid queries and none of the decoders', so K is not a tuned number. K=10 is too
   low: it gates 16–34 of the decoders' covid queries and costs them 0.03–0.10.
2. **Fall-back (`all`) is a guarantee, not a repair.** A gated query is answered from the
   untouched trained store, so it can never score below the baseline; e5/trec-covid is
   exactly 0.000 at every K. Its cost is bounded by the gated fraction (K=20: 0.022 on
   e5/scifact, 0.020 on jina/trec-covid; K=50: at most 0.007 anywhere). It does not repair
   jina/scifact (+0.014, not significant).
3. **Df-drop at K=20 is the rule to report.** Both failures repaired — e5/trec-covid
   −0.320 → +0.020, jina/scifact −0.009 → +0.045* — every gaining cell within 0.018 of its
   ungated two-store number, and octen/trec-covid, held out, up from +0.186 to +0.196. It is
   the only compliant rule that makes jina/scifact significant, because that failure is not
   an explosion (median 11 firing) but broad entries firing on many documents, which the
   index statistic identifies without a query.
4. **Top-k is fragile in the wrong direction.** K=20 gives the best e5/scifact number
   (+0.057) and +0.035 on e5/trec-covid, but K=50 leaves e5/trec-covid at −0.065 and
   jina/scifact at −0.005: keeping more of the inserted entries reintroduces the crowding.
5. **The price of the constraint** is visible on the two failing cells only: the
   query-sample gate reaches +0.081 on jina/scifact and +0.074 on e5/trec-covid where the
   compliant rule reaches +0.045 and +0.020. On the other seven cells they are within 0.018.
6. **Caveats.** The data contain one explosion-type failure (e5/trec-covid), so the margin
   in point 1 is one observation. The two-store design changes the ungated e5 numbers
   (point "why two stores"); the domain table reports the one-store numbers, this section
   both. The per-document cap is a property of the evaluation store, but the eviction it
   causes is a real effect of inserting into any budgeted index.

## Main-experiment backbones: wiring (2026-09-08)

Decision: jina-embeddings-v5-text-small, embeddinggemma-300m, snowflake-arctic-embed-l-v2.0
(`notes/backbone_survey.md`). Every backbone is checked the way Pilot M checked its two
(D17): pooled embeddings against the model's own sentence-transformers pipeline, finiteness
of the unit states at every candidate layer, and throughput at 192 tokens, batch 64.

| backbone | key | stack; pooling; prompts | layers probed | pooled cos vs own pipeline | precision | passages/s |
|---|---|---|---|---|---|---|
| arctic-embed-l-v2.0 | `arctic` | XLM-R large 24 × 1024; CLS; `query: ` / none | 8, 12, 16, 20, 24 | 1.0000 (fp32 and fp16) | fp16; states finite, max coordinate 26 | 259 (5 layers) |
| embeddinggemma-300m | `gemma` | Gemma 3 bidirectional 24 × 768; mean incl. prompt, then Dense 768→3072→768; `task: search result \| query: ` / `title: none \| text: ` | 6, 9, 12, 15, 18, 21, 24 | 1.0000 (fp32) | fp32 only: fp16 states are NaN at layer 6 already; residual coordinates reach 37,000 at layer 21, so states are stored scaled by 1/8 (`state_scale`; cosine- and whitening-invariant); outer autocast disabled around its forward | 142 (7 layers, fp32) |
| jina-v5-small | `jina5s_alnum` | as Pilot M | 12, 16, 20, 24, 28 | 0.9997 (Pilot M) | fp16 | 183 (full stack) |
| arctic-embed-m-v2.0 (wired, not selected) | `arctic_m` | GTE 12 × 768; CLS; `query: ` / none | 4–12 | 1.0000 | fp16 | 417 |

All three use the text-defined unit rule (D19). Gemma's bidirectionality was checked
directly: the layer-12 state of `bank` in "the bank of the river …" against "the bank of
the money …" has cosine 0.87, so later tokens reach earlier states. Card-reported nDCG@10
on our three corpora, the end-to-end check for each backbone's dense row once
`94_domain_baselines` runs: arctic-l SciFact 70.6, NFCorpus 35.3, TREC-COVID 83.9;
arctic-m 71.8, 35.9, 80.3. EmbeddingGemma's card publishes no per-dataset numbers.
