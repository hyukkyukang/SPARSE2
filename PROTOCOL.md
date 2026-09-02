# Pilot Study Protocols — Dynamic-Vocabulary Learned Sparse Retrieval

> This is the study's source document, reproduced as written before any work began.
> It is kept verbatim so that every result can be checked against what was actually
> specified. Where the study departed from it, the departure and the measurement that
> forced it are recorded in `notes/deviations.md`; `notes/protocol.md` is a condensed
> operative summary of the hypotheses and decision rules.

## Background

### Research topic

Learned sparse retrieval (LSR) with a **dynamic vocabulary**: a sparse retriever whose output dimensions can be added after training, without retraining or introducing new parameters.

### The problem

In standard LSR (SPLADE and its descendants), a text is encoded into a sparse vector over the model's wordpiece vocabulary, and each dimension's weight comes from an output projection matrix — in practice the MLM head's tied embedding matrix. The vocabulary is therefore **baked into the parameters**. Adding a dimension for a new term, entity, product, or concept means adding a row to a learned matrix, which is untrained and meaningless until the model is retrained. This is a poor fit for corpora whose terminology moves: new entities, new product names, new domain jargon, evolving user intent.

### Proposed approach

Define the vocabulary as an arbitrary **set of texts**, and obtain each dimension's vector by *encoding that text* with the same dense encoder used on the input side, rather than by learning a parameter per dimension.

- Vocabulary V = {t₁ … t_|V|}; entry vectors v_j = f(t_j) for a dense encoder f.
- For an input x, take contextual token states h₁ … h_n = f(x), score each against every entry, and max-pool over tokens: s_j(x) = pool_i sim(h_i, v_j).
- Apply a nonlinearity and a threshold so s(x) is non-negative and sparse; retrieve with an inverted index and a dot-product score, exactly as in existing LSR.
- Train with a ranking loss (contrastive, with hard negatives) plus a sparsity regularizer (FLOPS), with the constraint that **no parameter is indexed by j**.

Because the output "matrix" is produced by the encoder rather than stored as weights, a new vocabulary entry is added by encoding its text. If document token states or dense embeddings are cached, a new entry's posting list is one matrix–vector product over the corpus, with no re-encoding of documents.

In SPLADE terms: replace the fixed wordpiece embedding matrix of the MLM head with a text-generated projection matrix. In another framing, the model performs sparse coding of contextual token states over a dictionary whose atoms are defined by text.

### Goal

Show that **entries added after training behave like entries seen during it** — comparable activation rates, weights, posting-list lengths, and retrieval contribution — so that the vocabulary can be extended at any time, and that this holds for arbitrary text units rather than a fixed inventory.

### Central hypotheses under test

1. Contextual token states of a dense retriever, though trained only through a pooled output, retain enough term-level semantics to be scored against text-defined vocabulary entries.
2. A frozen encoder plus a light head suffices; the head generalizes across the entry space because it applies one smooth map to every token state, whereas fine-tuning against a fixed entry matrix lets the model co-adapt to specific entries and fail on new ones.
3. The resulting representation is sparse and effective enough to index and retrieve with.
4. Unseen entries are calibrated — they neither fire on everything (hubness, runaway posting lists) nor stay silent.

Hypothesis 4 is the main technical risk, and hypothesis 2 is the main design question; the pilots below are ordered so that the cheap tests of 1 and 3 gate the expensive test of 2 and 4.

### Positioning against prior work

- **DyVo** (Nguyen et al., EMNLP 2024) extends a SPLADE-style wordpiece vocabulary with Wikipedia *entities*, scoring output states against pre-existing entity embeddings retrieved per text by a separate candidate-selection step. Closest prior work and the primary baseline to position against. Differences: entities only, a separate retrieval stage, wordpiece vocabulary retained, and entity embeddings sourced independently of the input encoder.
- **SparseEmbed** (Kong et al., SIGIR 2023) attaches a contextual embedding to each activated wordpiece term — dense vectors *enrich matching within* fixed dimensions, where here dense vectors *define* the dimensions.
- **SPLATE** (Formal et al., SIGIR 2024) trains a small SPLADE-style adapter on frozen ColBERTv2 token embeddings for candidate generation. Architecturally the closest analogue to the frozen-encoder + light-head design, and useful evidence that the capacity is sufficient, but its output vocabulary is the fixed wordpiece set with a learned per-dimension projection.

None of these defines the vocabulary by encoding text with the same encoder used for the input, and none accepts an arbitrary new entry post-training.

### How the pilots serve the goal

| Pilot | Question | Gates |
|---|---|---|
| A | Do token states carry term-level semantics against text-defined entries, and can one linear transform make the profile peaky? | Encoder and layer; whether a frozen encoder is viable at all |
| B | Should an entry be a bare-string embedding or a contextual prototype? | What "adding an entry" costs at deployment; calibration risk |
| C | Does the untrained projection already retrieve? How lossy is it? | Whether to train at all as parameterized; the sparsity operating point |
| D | Do held-out entries behave like seen ones, under which training regime? | **Go/no-go for the whole premise**; frozen vs. LoRA vs. fine-tuned |
| E | If not, can insertion-time calibration fix it without training? | Whether calibration is a component or the premise is wrong |

Pilots run in the order A → B → C → D (→ E only if D shows a systematic gap). Each fixes a choice the next inherits. Section 0 defines everything shared; the pilots reference it rather than restating it.

---

## 0. Shared setup

### 0.1 Encoders

| Role | Model | Pooling | Notes |
|---|---|---|---|
| Primary candidate | `intfloat/e5-base-v2` | mean | `query: ` / `passage: ` prefixes required. `passage: ` for corpus text and vocabulary contexts, `query: ` for queries. d = 768. |
| Contrast candidate | `BAAI/bge-base-en-v1.5` | CLS | No prefix on passages; queries use `Represent this sentence for searching relevant passages: `. L2 normalization applies to the *pooled* output only — hidden states are used as-is. d = 768. |
| Reference ceiling | `colbert-ir/colbertv2.0` | token-level | Its 128-d projected, L2-normalized token embeddings, with `[Q]`/`[D]` marker tokens as in the original. Pilot A only, final layer only; its whitening is estimated in its own 128-d space. |
| LSR reference | `naver/splade-cocondenser-ensembledistil` | — | Pilots A and C (overlap), C (effectiveness reference). |

Pilot A selects one of the two candidates and a layer; B, C, D use only the winner.

`output_hidden_states=True` returns 13 tensors for a base model: index 0 is the embedding layer, index 12 the final encoder layer. The sweep uses **indices 4–12**.

### 0.2 Data

- **Prototype pool P**: the MS MARCO v1 passage collection (8.8M) minus the passages in Q and minus C₁'s qrels passages. Vocabulary entries are built from occurrences here. Drawing from the full collection, rather than from a small sample, is what makes tail entries usable.
- **Probe / bank pool Q (40k passages)**: uniform sample, disjoint from P. Probes, the background token bank, and all whitening statistics come from here. The P/Q disjointness is what prevents a probe's own passage from contributing to its own entry.
- **Statistics sample S (100k passages)**: uniform sample used for document-frequency and activation statistics in B and D. May overlap P (it is only used for corpus-level counts, never for self-matching).
- **Background token bank B_H (200k token states)**: word units (0.4) sampled uniformly from Q, excluding special tokens and pure punctuation. 200k × 768 × fp16 ≈ 307 MB per (encoder, layer).
- **Query-side background bank B_Q (200k token states)**: word units from 50k MS MARCO train queries encoded with the query convention. Needed because the query prefix shifts the distribution (0.5).
- **Retrieval sub-corpus C₁ (~1.5M passages)**, used by C, D, E: union of (i) all qrels passages for MS MARCO dev-small (6,980 queries), (ii) BM25 top-100 per dev query, (iii) SPLADE++ top-100 per dev query, (iv) **the dense candidate encoder's top-100 per dev query**, (v) 500k random passages. Deduplicate.
  - Including every reference system's own candidates is required for fairness: a system whose hard negatives are absent from the sub-corpus is flattered. Our own method's negatives are still not represented (they cannot be, before the method exists), so C₁ remains mildly optimistic for us. Treat all C₁ numbers as *relative* and note the direction of the residual bias in the paper.
  - **Full-corpus confirmation**: after Pilot D selects a configuration, encode the full 8.8M collection once (~2–4 GPU-hours) and re-run that configuration and the references. Every headline number in the paper comes from this run, not from C₁.
- **Queries**: dev-small (6,980) for evaluation; 200k sampled from MS MARCO train for Pilot D.

### 0.3 Vocabulary V (fixed for all pilots)

Candidate entries: lowercase alphabetic words, ranked by collection frequency. **Membership requires ≥ 100 occurrences in P.** Take the top 30,000 that qualify; if fewer than 30k qualify, shrink V and report the realized size. Keep stopwords in (they are the hub test case in B) but flag a ~150-word stopword list so every metric can be reported with and without them. Store collection frequency `freq_j` for stratification.

**Run first, before anything else**: the occurrence-count distribution per frequency decile. If the bottom decile is thin, cut V rather than accept noisy prototypes — prototype noise correlated with frequency would confound B and D.

Vocabulary size and type (phrases, entities) are main-paper ablations, not pilot variables.

### 0.4 Token states and word units

- Merge wordpieces into words via the tokenizer's offset mapping; a word unit's state is the **mean of its piece states**. Drop [CLS], [SEP], pad, and pure-punctuation units.
- Word units make "own word" and "occurrence" well defined and match the vocabulary's granularity. Operating on raw wordpieces is a later ablation.

### 0.5 Whitening (the one linear transform allowed before training)

Whitening is estimated **per space**, not shared across spaces. For a set of vectors X: μ_X = mean, Σ_X = covariance, shrink Σ_X ← Σ_X + εI with ε = 0.01 · tr(Σ_X)/d, W_X = Σ_X^(−1/2) (ZCA). (ε is the shrinkage constant; λ later denotes FLOPS weights.)

- **Token side (document)**: W_H, μ_H from B_H.
- **Token side (query)**: W_Q, μ_Q from B_Q. Before adopting a separate transform, measure ‖μ_Q − μ_H‖ / ‖μ_H‖ and the Frobenius difference of the two covariances; if both are under 5%, use the document-side transform for both and record that as a finding.
- **Entry side**: one transform per representation, estimated from that representation's own matrix — W_V^R1 from the 30k bare-string embeddings, W_V^R2 from the 30k prototypes. For R2, W_V^R2 ≈ W_H (prototypes are averaged token states) but estimate it separately anyway so R1 and R2 are treated identically.
- Similarity: a(h, v) = cos(W_H(h − μ_H), W_V(v − μ_V)). "Raw" = cos(h, v); "centered" = cos(h − μ_H, v − μ_V).
- **Secondary condition for R1**: also report R1 under the shared token-side transform. The shared-space property is part of the claim, so the gap between R1-own-transform and R1-shared-transform is itself a result.

Estimate each transform once and freeze it. Reusing frozen statistics across pilots is what makes their results comparable.

### 0.6 Training-free sparse encoding rule (B, C; initialization of D)

For text x with word-unit states h_1..h_n:

1. a_ij = cos(h̃_i, ṽ_j) for all entries j
2. p_j(x) = max_i a_ij (max-pool over units)
3. s_j(x) = ReLU(p_j(x) − τ)
4. Truncate to the top-k_d (documents) or top-k_q (queries) by s_j. Optional saturation s_j ← log(1 + s_j) — monotone, so it commutes with the max and can be applied after pooling.
5. score(q, d) = Σ_j s_j(q) · s_j(d)

**Setting τ.** Not a per-token percentile. Binary-search τ on 5,000 passages of S so that mean nnz per passage before truncation is ≈ 120, and separately τ_q on 5,000 dev queries for ≈ 30 nnz. Report the equivalent per-token percentile as a diagnostic (expect roughly p99.99, not p99). Grid variants in C are ±1 step around these operating points. This is also what D's threshold initializes to, so D does not begin in a state the sparsity regularizer has to dig out of.

### 0.7 Tooling and bookkeeping

- HF `transformers` (+ `peft`), Pyserini (BM25, prebuilt SPLADE index, qrels), `ir_measures` (MRR@10, R@100, R@1000), `scipy.sparse` for scoring, `torch` matmul for kNN.
- **Statistical testing**: every effectiveness comparison gets a paired bootstrap CI over queries (10,000 resamples, 95%). Report the CI, not just the point estimate; on Q_H, which may be only a few hundred queries, differences of a few MRR points will often be indistinguishable from zero.
- Fix seeds; log every config as JSON; version the P/Q split, banks, whitening statistics, τ, and V as artifacts.
- Compute: one 40–80 GB GPU. A–C ≈ 4 GPU-hours; D ≈ 30; E < 1; full-corpus confirmation ≈ 4.
- **Similarity profiles are streamed, never stored.** A 50k × 30k profile is 6 GB in fp32; metrics accumulate on the fly in chunks.

---

## Pilot A — Token-state geometry against a text-defined vocabulary

**Purpose.** Verify that token states of a pooled-output retriever carry term-level semantics usable against a *text-defined* vocabulary, that one linear transform makes similarity profiles peaky enough for sparsity, and choose encoder + layer.

### A.1 Setting

- Encoders: both candidates, layers 4–12; ColBERTv2 (final layer) as ceiling.
- **Entry representations, both built here** (A needs both because the primary metric is cross-representation):
  - R2 prototypes: v_j = mean word-unit state over 50 occurrences sampled from P (all if fewer), per (encoder, layer).
  - R1 bare strings: pooled embedding of the word alone. Encoder-final-layer object only, so it is layer-independent by construction — which is precisely why it makes a good fixed target for the layer sweep.
- **Probe set**: 50,000 word-unit occurrences from Q — 40k content words spread uniformly over frequency deciles of V, 10k stopwords. Split into a 5k **tuning slice** (used for any convention choice, e.g. E5's prefix for R1) and a 45k **reporting slice**. Never select on the reporting slice.
- **Related-term set**: for each of 5,000 probe passages, the top-10 SPLADE expansion terms of that passage that are in V and are not the probe word itself.
- **Polysemy set**: 30 words × 2 senses × 40 occurrences from P, sense-labeled by an LLM with 10% manual verification; 5 anchor entries per sense. **Verify every anchor is in V before running** — low-frequency anchors (e.g. `cupertino`) will often not qualify under 0.3 and must be substituted. Starting list: bank (loan/deposit/mortgage/account/credit | shore/creek/stream/flood/water), python (reptile/venom/snake/species/habitat | programming/script/function/library/software), apple (pie/orchard/juice/tree/fruit | iphone/mac/computer/software/device), mouse, java, jaguar, cell, crane, bass, pitcher, seal, spring, ruler, plant, bat, tank, port, match, current, novel, mine, bark, palm, pupil, race, rock, scale, star, trunk, wave.

### A.2 Procedure

1. **Build prototypes in one streaming pass.** For each passage in the sampled portion of P, encode, extract word units, add each unit's state to a running per-(entry, layer) sum and count. Store only the resulting 30k × 768 prototype matrices — never the individual states. Caching all of P's states across 9 layers would be ≈ 45 GB and is never needed.
2. Encode the probe passages once, retaining probe unit states at all layers (45k+5k × 9 × 768 × fp16 ≈ 6 GB), and the bank B_H.
3. Compute R1 embeddings; on E5, choose the prefix on the **tuning slice** only.
4. Estimate whitening per space (0.5) and τ per (encoder, layer) (0.6).
5. Stream the metrics below over probes × entries in chunks, under raw / centered / whitened, for every layer.
6. Repeat 1–5 for ColBERTv2 (final layer, 128-d, own whitening).
7. Sense probe and SPLADE overlap at each candidate's best layer, plus at layer 12 for reference.
8. **SPLADE overlap**: for 1,000 random passages of Q, Jaccard of our passage-level top-20 with SPLADE's top-20. Map SPLADE wordpieces to words by string match, compute Jaccard **only over the mutually representable subset**, and report coverage (the fraction of SPLADE's top-20 that maps into V) alongside — otherwise the number mostly measures vocabulary mismatch.

### A.3 Measurements

**Primary (these select the layer):**

- **Cross-representation self-hit@10**: probe token state vs. **R1** entries — is the probe's own word in the top-10? This is the real question: does a contextual state align with the embedding of the *text* naming it? Report content/stopword separately, per layer, per transform.
- **Related-term rank**: mean reciprocal rank of the passage's SPLADE-expansion terms in the probe's profile (against R1 and against R2), versus a random-entry baseline. Measures whether the profile is semantically organized beyond identity.
- **Sense accuracy**: per polysemous occurrence, margin = mean similarity to the correct sense's anchors − mean to the wrong sense's; accuracy = fraction with margin > 0.
- **Peakiness** (whitened vs. raw): z-gap = (max_j a_j − median_j a_j) / std_j(a_j); participation ratio PR = (Σ_j r_j)²/Σ_j r_j² with r_j = ReLU(a_j − τ); nnz per token at the 0.6 operating point.

**Secondary (floor checks, not selectors):**

- **Prototype self-hit@10** (probe vs. R2): expected ≈ 0.9+ at every layer. A value below ~0.7 means something is broken in the pipeline; it cannot discriminate between working layers.
- **Matched-vs-random AUC**, SPLADE top-20 Jaccard + coverage.

Deliverable: table (encoder × layer × transform) of cross-rep self-hit@10, related-term MRR, z-gap, PR, nnz/token; ColBERTv2 as the bottom row; sense accuracy and SPLADE overlap at the selected layers.

### A.4 Expected results and decision rule

- **H1**: at the best layer of the mean-pooled model, whitened cross-representation self-hit@10 ≥ 0.4 and related-term MRR at least 3× the random baseline. (Deliberately below the prototype-self-hit expectation — cross-representation is the harder, informative test.)
- **H2**: the best layer is *not* 12 for the CLS-pooled model (last-layer states are specialized for pooling); E5's best layer lies in 8–11 and its degradation at 12 is milder.
- **H3**: whitening raises z-gap ≥ 2× and cuts nnz/token by an order of magnitude relative to raw.
- **H4**: sense accuracy ≥ 0.8; ColBERTv2 above both candidates on every metric but not by a wide margin.
- **Selection**: (encoder, layer) = argmax whitened cross-representation self-hit@10, with sense accuracy ≥ 0.8 required and related-term MRR as tie-breaker. **Carry the top-2 layers into Pilot B** and confirm the choice under B's winning representation — otherwise the layer is chosen under R2 and the representation under that layer, which is circular.
- **If cross-rep self-hit < 0.15 at every layer of both candidates but prototype self-hit is high**: token states are word-identifiable but not aligned with the pooled space. R1 is dead; the method must use contextual prototypes, and the paper should say so explicitly.
- **If both self-hit variants are low**: the frozen-encoder plan is not viable. Drop V1 from Pilot D, start at LoRA, and give the head a nonlinearity.
- **If self-hit is fine but whitening barely changes peakiness**: keep the encoder, but expect sparsity to be produced by the learned threshold rather than by geometry (raise λ in D).

---

## Pilot B — Entry representation: bare string vs. contextual prototype

**Purpose.** Decide what a vocabulary entry *is*, and therefore what adding one costs at deployment: encode a string, or collect k occurrences.

### B.1 Setting

Encoder and top-2 layers from A. Representations of the same V:

- **R1 — bare string**: pooled embedding of the word alone (prefix fixed in A). The literal form of the original idea; insertion cost = 1 forward pass.
- **R2 — contextual prototype**: mean word-unit state over k occurrences from P, k = 50.
- **R3 — multi-prototype** (run only if R2 sense accuracy in A was < 0.8): 5 k-means centroids over up to 100 occurrences; similarity = max over centroids.

Stability curve: build R2 twice from **disjoint** occurrence sets S1, S2 ⊂ P for k ∈ {1, 3, 5, 10, 20, 50}. The ≥ 100-occurrence floor in 0.3 is what makes k = 50 disjoint pairs possible for every entry.

Whitening: each representation under its own W_V; R1 additionally under the shared token-side transform (0.5).

### B.2 Procedure

1. Build R1, R2, R2′, R3, and the k-curve variants. Whiten.
2. Top-10 entries of every token in B_H, per representation (200k × 30k, chunked).
3. Encode S with rule 0.6 (τ re-fit per representation to the same nnz target — otherwise hubness differences are confounded with density differences); record df_j.
4. Encode each entry's own word as an input text; record its full profile.
5. Compute metrics, stratified by frequency decile and stopword/content.

### B.3 Measurements

- **Hubness**: N_10(j) = number of bank tokens whose top-10 contains j. Report skewness of N_10 and hub share (fraction of all top-10 slots taken by the top 1% of entries). **List the 20 largest hubs verbatim** — the qualitative list is usually the most informative output of this pilot.
- **df calibration**: Spearman(df_j, freq_j); runaway rate = fraction of entries with df_j > 20% of documents; df distribution per decile.
- **In-context self-hit@10** (the fair comparison): A's cross-representation metric computed against each representation.
- **Self-activation rank**: rank of j in the profile of its own word as input. Median and P(rank = 1). *Structurally optimistic for R1 on a mean-pooled encoder* — a one-word input's pooled vector nearly is its token state — so it is reported as a diagnostic for the insertion check in E, never as evidence for R1 over R2.
- **Stability curve** (R2): per k, mean Jaccard between the top-50 nearest bank tokens of the S1- and S2-prototypes, and whitened cosine between them. Raw cosine is uninformative in an anisotropic space.
- **Sense accuracy** per representation, reusing A's polysemy set.

Deliverable: table (representation × {hub skew, hub share, Spearman df–freq, runaway %, in-context self-hit, self-activation rank, sense accuracy}), the stability curve, the hub lists, and the R1-own-transform vs. R1-shared-transform comparison.

### B.4 Expected results and decision rule

- **H5**: R1 has hubness skewness several times R2's, a hub list dominated by function words and generic terms, weaker df–freq correlation, and lower in-context self-hit.
- **H6**: R2's stability Jaccard at k = 10 reaches ≥ 0.8 of its k = 50 value, so insertion needs ~10 occurrences.
- **H7**: R1 under its own transform is meaningfully better than R1 under the shared transform — which would mean the "one shared space" story needs a per-space transform, a small but real caveat for the method.
- **Selection**: lowest hubness among representations that are not worse on in-context self-hit. Set insertion cost k* = smallest k reaching 0.8 × the k = 50 Jaccard.
- **If R1 ≈ R2**: use R1 — simpler method, no occurrences needed at insertion — and report it as a finding.
- **If runaway rate > 5% for both**: Pilot E becomes mandatory rather than conditional.
- Re-confirm A's layer choice under the winning representation before moving on.

---

## Pilot C — Training-free end-to-end retrieval

**Purpose.** One end-to-end number before any training: does the projected, truncated representation carry ranking signal, and how lossy is projection + sparsification relative to the dense model? Fixes the sparsity operating point for D. This is D's head at initialization, so the only extra cost is retrieval.

### C.1 Setting

- Encoder/layer from A, representation from B, whitening and τ from 0.5–0.6.
- Corpus C₁; queries dev-small.
- Grid: k_d ∈ {64, 128, 256}, k_q ∈ {16, 32, 64}, τ at the nnz-120 operating point and one step stricter (nnz ≈ 60), saturation ∈ {none, log1p}. 36 cells, all re-truncations of one stored set of profiles.
- References on C₁: BM25 (Lucene index of C₁), the dense encoder itself, SPLADE++.

### C.2 Procedure

1. Encode C₁ once; per passage store the top-256 (entry, weight) pairs at each τ (≈ 1.5 GB each). Encode dev queries, top-64. Store pooled dense embeddings for both.
2. Build and evaluate the three references on C₁.
3. Per grid cell, materialize CSR matrices, score in query batches, take top-1000.
4. **Score preservation**: per query, over the dense top-100 within C₁, Spearman between dense and our scores; Jaccard of dense top-100 with our top-100.
5. **Natural sparsity**: mean nnz before truncation, per passage and per query, per τ.
6. **Qualitative**: dump top-10 entries with weights for 20 passages and 20 queries; annotate each list plausible / not (two annotators if available, reporting agreement; otherwise one, reported as such).
7. SPLADE overlap on 1,000 C₁ passages, on the truncated representation actually used for retrieval, with coverage reported as in A.2.

### C.3 Measurements

MRR@10, R@100, R@1000 per run with bootstrap CIs; Spearman and Jaccard vs. dense; natural sparsity; plausibility rate; SPLADE overlap. Plot MRR@10 vs. k_d, one curve per k_q.

### C.4 Expected results and decision rule

- **H8**: the best untrained run reaches ≥ 0.8 × BM25's MRR@10 on C₁ and falls between BM25 and dense.
- **H9**: Spearman vs. dense ≈ 0.5–0.7 at k_d = 128, declining smoothly as k shrinks; MRR@10 saturates by k_d ≈ 128–256, k_q ≈ 32; plausibility ≥ 80%.
- **Pass**: proceed to D with the nnz targets at the knee of the curve (expect ≈ 120 doc / ≈ 30 query). The best (τ, saturation) becomes D's initialization.
- **Weak (0.4–0.8 × BM25)**: proceed, but include the LoRA arm from the start.
- **Fail (< 0.4 × BM25 and Spearman < 0.3)**: the frozen space does not support this projection as parameterized. Redesign the head (nonlinear, higher capacity) and rerun C before spending D's compute.

---

## Pilot D — Held-out vocabulary generalization under different training regimes

**Purpose.** The go/no-go test of the central claim: entries added after training behave like entries seen during it. Also settles frozen vs. LoRA vs. full fine-tuning and whether vocabulary dropout is required.

### D.1 Model

- Token side: h̃ = W_H(h − μ_H) [frozen], then a residual MLP g(h̃) = h̃ + Linear₂(GELU(Linear₁(h̃))), 768→768→768, Linear₂ zero-initialized so g = identity at step 0 (≈ 1.2M params).
- Entry side: ṽ_j = W_V(v_j − μ_V), L2-normalized, frozen |V| × 768 matrix.
- Logit: z_ij = t · cos(g(h̃_i), ṽ_j) − b, learnable scalars t (init 20) and b (init t · τ from C). Note the symbol: z is the logit, a is the whitened similarity of 0.5.
- Representation: s_j(x) = log(1 + ReLU(max_i z_ij)). Score = Σ_j s_j(q)·s_j(d).
- **Nothing is indexed by j.** Hard constraint for every variant.
- **μ_V, τ, and b's initialization are computed over seen entries only.** Computing them over all 30k leaks held-out information into training.

### D.2 Training data and loss

- 200k train queries; positive from qrels; 7 BM25 hard negatives per query per epoch.
- Batch: **128 queries × 8 passages for V1** (no encoder backward pass, so the memory is available and in-batch negatives are far better); 32 × 8 for V2 and V3. Record the difference — effectiveness across arms is not directly comparable at different batch sizes; use gradient caching to equalize if the comparison turns out to be close.
- Loss: InfoNCE over {positive, 7 hard negatives, other in-batch passages} + λ_q·FLOPS(q) + λ_d·FLOPS(d), FLOPS(x) = Σ_j (batch-mean s_j(x))², quadratic λ ramp over the first 20% of steps.
- **Under vocabulary dropout, rescale λ by 1/(1−p)** so the effective sparsity pressure is unchanged (FLOPS is a sum over entries; masking 30% shrinks it by ~30%).
- λ tuning on V1 only: λ_d ∈ {1e-4, 3e-4, 1e-3}, λ_q = 3λ_d; keep the pair landing nearest C's nnz targets. Reuse for all variants and **always report nnz beside effectiveness**.
- 2 epochs, AdamW, lr 2e-4 (head + scalars), 500 warmup, linear decay, wd 0.01, bf16.
- **Seeds**: 2 for every arm the final decision depends on (at minimum V1, V1+VD, and the best-performing alternative). Report seed spread next to ρ; a threshold decision on a single seed is not a decision.

### D.3 Variants and controls

| Id | Encoder | Entry side during training | Head |
|---|---|---|---|
| V1 | frozen | fixed | trained |
| V2 | LoRA (r=16, α=32, q/k/v/o), lr 1e-4 | **refresh a random 10% of entries every 100 steps** by re-encoding their contexts | trained |
| V3 | full fine-tuning, lr 2e-5 | fixed at initialization — the "freeze the vocabulary" option | trained |

V2 refreshes entries in a staggered way rather than re-encoding all of them on a long interval: a full refresh lets the entry matrix drift stale between updates and produces a loss discontinuity at each one. Full re-encoding of the prototype contexts costs 60–120 s; the staggered version amortizes it.

**Vocabulary dropout (VD)**: per step, mask a uniformly random 30% of *seen* entries on both sides (same mask); loss and FLOPS over unmasked entries only, with λ rescaled.

**Oracles**: same architecture, trained on the full vocabulary. Run one per architecture that is a live candidate — V1-oracle and V3-oracle at minimum, V2-oracle only if V2 becomes the recipe. ρ must always compare a model to an oracle of *its own* architecture.

**Controls** (both cheap, both close off an alternative explanation of a positive result):

- **C-rand**: replace V with random unit vectors of the same shape and train V1. If effectiveness is comparable, the text-defined vocabulary is not what carries the signal, and the method's premise is decorative.
- **C-alias**: at test time, map each held-out entry to its nearest *seen* entry instead of inserting it. If this matches genuine insertion, held-out entries are redundant with V and the demonstration is empty — a passing ρ must beat C-alias to mean anything.

### D.4 Vocabulary splits

- **Random-stratified**: within each frequency decile, hold out 20% at random (≈ 6k held / 24k seen). Stopwords never held out.
- **Cluster**: k-means (k = 100) over whitened entry vectors; hold out random clusters until 15–25% of entries are held out. Simulates a semantically new region.
- **Rare**: hold out the entire bottom frequency decile. Closest analogue to the real use case — new, rare terms arriving after training — and the hardest. Report separately; never merge into the random split.

Held-out entries are removed from V during training, including from the FLOPS sum, and re-inserted at test time. **They are removed only as output dimensions; their surface forms still appear in the input text and are still encoded normally.** For V1/V3 insertion is the original row; for V2 it is a re-encoding with the trained LoRA encoder — still parameter-free.

### D.5 Runs

Random split: V1, V1+VD, V2, V2+VD, V3, V3+VD, V1-oracle, V3-oracle, C-rand (9 configs; 2 seeds on V1, V1+VD, and the best alternative). Cluster split: V1, V1+VD, V3, V1-oracle. Rare split: V1, V1+VD, V1-oracle. C-alias is evaluation-only. ≈ 30 GPU-hours.

### D.6 Evaluation

1. Retrieval on C₁ with the full vocabulary: MRR@10, R@100, nnz(q), nnz(d). The trained threshold defines sparsity, but apply a **safety cap of top-1024** and report the fraction of documents that hit it; an uncapped run can be slow and is not comparable to C.
2. Retrieval with held-out entries zeroed ("seen only"), for the model and its oracle. Zeroing only — no renormalization of the remaining weights.
3. **Q_H, primary definition**: dev queries where held-out entries carry ≥ 20% of the *oracle's* query–positive score. This targets cases where the new vocabulary actually matters, including expansion matches.
   **Q_H-lex, secondary**: queries where a held-out word appears in both query and a relevant passage — the easier, lexical-overlap subset, reported separately. Do not use this as the primary definition: it selects exactly the cases a lexical matcher already handles.
   Report |Q_H| for every split.
4. **Recovery ratio** on Q_H: ρ = [MRR_model(all) − MRR_model(seen-only)] / [MRR_oracle(all) − MRR_oracle(seen-only)].
   **Validity precondition**: compute the denominator first. It must be ≥ 0.02 absolute MRR with a bootstrap CI excluding zero. If it fails, the split does not exercise the held-out vocabulary — enlarge the held-out fraction or change the split — and the decision falls back to activation statistics (item 5). Report ρ with a bootstrap CI; never report it as a bare point estimate.
   Also report the absolute gap to the oracle on Q_H and on all dev queries (sanity: the model must not be broken elsewhere), and the same quantities for C-alias.
5. **Activation statistics** on S, per entry: activation rate r_j, mean nonzero weight w̄_j, posting-list length, self-activation rank. Compare held-out vs. seen **within frequency decile**: signed gap_r = median over deciles of log(median r_held / median r_seen) — the sign distinguishes over- from under-firing and determines whether E can help — plus |gap_r|, gap_w, and the within-decile Wasserstein distance between the two activation-rate distributions.

Deliverable: table (variant × split) of MRR@10 all / seen-only with CIs, ρ with CI, gap to oracle on Q_H, nnz, signed gap_r, gap_w; plot of held-out vs. seen activation-rate distributions per variant; C-rand and C-alias rows.

### D.7 Expected results and decision rule

- **H10 (pass criterion for V1 or V1+VD)**: ρ ≥ 0.8 with a CI excluding 0.5; gap to oracle on Q_H ≤ 10% relative; |gap_r| ≤ log(1.5); and ρ beats C-alias by a margin whose CI excludes zero.
- **H11**: V3 has the best seen-only MRR but ρ ≤ 0.5 with a systematic signed gap — trained against a fixed matrix, the encoder has no reason to respect the entry space's geometry.
- **H12**: VD closes much of V3's gap and is roughly neutral for V1. V2 sits between; V2+VD is the best overall if V1 lacks capacity.
- **H13**: difficulty orders random < cluster < rare for every variant, with V1 degrading least.
- **H14 (control)**: C-rand is clearly worse than V1 — if it isn't, stop and reconsider the premise before writing anything.
- **Decisions**: V1 passes → base recipe is frozen encoder + light head. Only V1+VD passes → VD is a core component, not an ablation. V1 fails but V2+VD passes → LoRA with staggered entry refresh is the recipe. All fail with a *systematic signed* gap → Pilot E. All fail with no systematic gap (held-out entries just noisy) → the problem is representational, not calibrational: move to inference-time sparse coding over V and symmetric training, and reframe the contribution.

---

## Pilot E (conditional) — Insertion-time calibration

Run only if D shows a systematic held-out gap (|signed gap_r| > log(1.5), or ρ < 0.8 with a consistent direction of error).

### E.1 Setting

Best trained model from D. For each entry, compute logits z over B_H to get background statistics μ_j, σ_j; targets μ*, σ* are the medians over *seen* entries in the same frequency decile. A new entry's decile is computable at insertion from its corpus df, so every variant below is deployable, not just diagnostic.

Variants, applied to held-out entries only, all training-free and seconds to compute:

- **Raw**: nothing.
- **Z**: z′_ij = (z_ij − μ_j)/σ_j · σ* + μ*, applied to logits **before** the threshold, so the entry's background activation distribution matches comparable seen entries.
- **DF**: per-entry threshold b_j by binary search on S so df_j matches the median df of seen entries in the same decile.
- **SA**: scale c_j so the entry's self-activation on its own word matches the seen median for its decile.

### E.2 Procedure and measurements

Re-run D.6 items 1–5 per variant. Report ρ with CI, gap to oracle on Q_H, signed gap_r, gap_w, and Δnnz(d) — a variant that "fixes" ρ by making documents denser has not fixed anything.

### E.3 Expected results and decision rule

- **H15**: Z alone closes ≥ 80% of the gap when the failure mode is over-firing (hubs); DF is safer when posting-list growth is the binding constraint; SA helps under-firing.
- Adopt the simplest variant reaching ρ ≥ 0.8. If none reaches 0.6, the gap is representational — see D.7's final decision.

---

## Appendix: pitfalls and checks

- **Sanity before A**: reproduce the dense encoder's own MRR@10 on C₁ and check it against published numbers. If it is off, the pipeline (prefix, pooling, normalization) is wrong and every pilot inherits the bug.
- **Occurrence counts before V is frozen** (0.3). Prototype noise correlated with frequency would confound every stratified comparison downstream.
- **Prefixes**: E5 passages/contexts/prototypes use `passage: `, queries `query: `; BGE queries take its instruction string. Fix R1's convention on the tuning slice, never the reporting slice.
- **Never re-estimate whitening, banks, or τ mid-pilot.** Differences must come from the variable under test.
- **Whitening is per space.** Do not whiten pooled embeddings with token-state statistics except as the explicitly labeled secondary condition.
- **Max-pool excludes special tokens and punctuation**, or [CLS]/[SEP] states dominate as hubs.
- **Leakage checklist**: prototypes from P only; probes/banks/whitening from Q only; held-out entries absent from V, FLOPS, μ_V, τ, and b during training; convention choices made on tuning slices.
- **Frequency confounds**: every held-out vs. seen comparison is within decile; never compare pooled distributions.
- **Density confounds**: report nnz beside every effectiveness number, and re-fit τ to a common nnz target whenever comparing representations.
- **C₁ caveat**: it contains every reference system's candidates but not our method's, so it is mildly optimistic for us. State this, and confirm the final configuration on the full 8.8M collection.
- **Statistics**: paired bootstrap CIs on every comparison; ρ's denominator validity-checked before ρ is reported; seed spread reported for any arm a decision depends on.

### Timeline

- **Weeks 1–2**: Pilot A. The layer sweep, prototype construction over the full collection, and three transforms make this two weeks, not one. Pilot B follows immediately on the same artifacts (top-2 layers only).
- **Week 3**: Pilot C, including reference runs on C₁ and D's λ sweep.
- **Weeks 4–6**: Pilot D — random split first, then cluster and rare once V1 and V3 exist; controls are cheap and run alongside. Pilot E if triggered.
- **Week 7**: full-corpus confirmation of the selected configuration.
