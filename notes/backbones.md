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
