# Running log

Chronological record of what was run, what it cost, and what turned up. Numbers
are copied from `results/*.json`; nothing here is a summary of expectations.

## Setup

* MS MARCO v1 passage collection (8,841,823 passages) as a mmap'd blob + offsets.
* Splits (§0.2): Q = 40,000 (probe/bank pool), S = 100,000 (statistics),
  P = 8,794,420 (collection minus Q minus the 7,433 dev-small qrels passages).
* Vocabulary (§0.3): 1,220,347 alphabetic word types in P; 67,676 clear the
  100-occurrence floor; V = the top 30,000. **The §0.3 pre-check passes with room
  to spare** — the bottom decile's minimum collection frequency is 498, so no entry
  has a noisy prototype and V was not cut. Every entry has a full 100-occurrence
  sample, which is what makes k=50 *disjoint* prototype pairs possible in Pilot B.
* Compute: 8xA100-40GB. Full-collection dense encode ~13 min/encoder; prototypes
  over 2.48M passages ~4 min/encoder; C1 encode ~3 min.

## Pipeline validation (§Appendix, before anything else)

| system | ours | published |
|---|---|---|
| BM25 (k1=0.82, b=0.68) | 0.1874 | 0.1875 |
| SPLADE++ CoCondenser-EnsembleDistil | 0.3827 | 0.383 |
| e5-base-v2 | 0.3542 | ~0.35 |
| bge-base-en-v1.5 | 0.3498 | ~0.35 |

All four reproduce, so prefix / pooling / normalisation conventions are right.

## §0.5 — the query side needs its own transform

Measured before adopting one: the query-side token distribution differs from the
document side by ‖μ_Q − μ_H‖/‖μ_H‖ = 0.16–0.54 and a relative covariance Frobenius
difference of 0.80–1.10 across layers, far above the 5% threshold at which the
protocol would have allowed a shared transform. Separate W_Q, μ_Q are used.

## Pilot A

Selected **e5-base-v2, hidden layer 9**; top-2 layers {9, 10} carried into B.
R1's prefix convention was fixed on the 5k tuning slice: **no prefix** for e5
(0.887 vs 0.826 doc / 0.842 query), the query instruction for bge.

* **H1 supported, decisively.** Whitened cross-representation self-hit@10 = 0.923
  at layer 9 (threshold 0.4); related-term MRR is 23x the random baseline
  (threshold 3x). A contextual token state does align with the embedding of the
  bare text that names it.
* **H2 supported.** Best layer is 9 (e5) and 10 (bge); neither is 12.
* **H3 not supported.** Whitening does *not* raise z-gap 2x (it lowers it at every
  layer below 12) and does not cut nnz/token by an order of magnitude (2.2x at
  layer 9). Most of the density reduction comes from **centering** alone. Whitening
  also *lowers* cross-representation self-hit below layer 11 (0.966 raw / 0.969
  centered / 0.923 whitened at layer 9). Sparsity will have to come from the
  learned threshold, not the geometry — §A.4's third fallback branch.
* **H4 partly supported.** Sense accuracy reaches 0.8 only at bge layer 12 (0.814),
  whose cross-representation self-hit is 0.507 — half of e5's. Applying the sense
  constraint verbatim would therefore select a far worse configuration, so the
  conflict is recorded and the selection is made on the primary metric.
  ColBERTv2 leads on identity (0.991) and SPLADE overlap (0.390) but *trails* e5's
  layer 12 on related-term MRR (74x vs 210x) and sense — so "above both candidates
  on every metric" does not hold.
* **A tension worth keeping.** Identity peaks at layer 9-10; semantic organisation
  (related-term MRR: 23x -> 210x) and sense both peak at layer 12. Pilot C
  therefore evaluates layer 9 *and* layer 12 end to end.

## Pilot B

Selected **R2 (contextual prototype)**; insertion cost k\* = 20 occurrences.

* **H5 reversed on hubness.** R1 (bare string) is *far less* hubby than R2:
  N_10 skewness 3.6 vs 17.2, hub share 0.044 vs 0.161. It is R2's hub list that is
  dominated by function words (`that, and, but, which, or, the`). The df-frequency
  half of H5 *is* confirmed: Spearman 0.249 (R1) vs 0.566 (R2).
* Runaway rate < 0.3% for every representation, so Pilot E is not mandatory on
  B's evidence.
* **H6 narrowly missed.** Stability Jaccard: k=10 reaches 0.788x the k=50 value,
  just under the 0.8 the rule wants; k=20 reaches 0.89x. k\* = 20.
* **H7 supported, strongly.** R1 under the shared token-side transform collapses:
  self-hit 0.923 -> 0.837, SPLADE Jaccard 0.286 -> 0.002, and its profile is so flat
  that the passage-level threshold leaves ~0 entries per token. The "one shared
  space" story does need a per-space transform.
* R3 (5 centroids per entry) is competitive at layer 9 but degenerate at layer 10
  (its threshold cannot reach the common density target), so it is not carried.

## Pilot C

Configuration selected by A and B: **e5, layer 9, R2 (contextual prototype),
whitened**, evaluated on C1 = 1,852,472 passages with dev-small (6,980 queries).
The 36-cell grid is 36 re-truncations of one stored profile set, as §C.1 requires.

References on C1: BM25 **0.1882**, e5-base-v2 dense **0.3542**, SPLADE++ **0.3817**.

* Best cell **k_d = 128, k_q = 16, nnz≈120, log1p**: MRR@10 **0.0963**,
  R@100 0.490, R@1000 0.751.
* **H8 not met.** 0.0963 / 0.1882 = **0.51x BM25**, inside the "weak" band
  (0.4–0.8x), so §C.4 says proceed to Pilot D *with the LoRA arm from the start* —
  which the run list already includes.
* **H9 not met on score preservation.** Spearman against the dense encoder over its
  own top-100 within C1 is **0.211** (expected 0.5–0.7) and Jaccard@100 is 0.137.
  The sparse projection is not a lossy copy of the dense score; it is a different,
  weaker signal.
* **H9 met on plausibility.** 39 of 40 annotated lists are plausible (single
  annotator, reported as such). Expansion is real: a nursing passage activates
  `midwives`, a cortisol passage `pituitary` and `gland`, a rash passage `ringworm`
  and `fungal`, a SWIFT-code passage `aba`.
* SPLADE++ top-20 Jaccard on the truncated representation: 0.358 at coverage 0.85.
* **Why the number is low, quantified.** Query profiles are peaked (top-1 carries
  18.5% of the mass; participation ratio 14 of 30 non-zeros) but document profiles
  are nearly flat (top-1 3.5%; participation ratio **63 of 119**). Untrained
  max-pooled cosine gives a document ~63 effectively-equal dimensions, so
  discriminative terms are not up-weighted. This is precisely what Pilot D's
  learned threshold and scale exist to fix, and it is consistent with the term
  lists being good while the ranking is not.

### Pilot C, continued — the layer the protocol picked is not the layer that retrieves

Because Pilot A left a tension (identity peaks at layer 9, semantics at layer 12)
and because H3 failed, Pilot C encoded C1 under six configurations instead of one.
Best cell per configuration (MRR@10 on C1):

| configuration | MRR@10 | note |
|---|---|---|
| e5 L12 R2 whitened | **0.1251** | best; full 36-cell grid |
| e5 L9 R2 whitened | 0.0963 | the configuration §A.4 selected |
| e5 L9 R3 whitened | 0.0962 | multi-prototype buys nothing |
| bge L12 R2 whitened | 0.1083 | what the literal sense constraint would have picked |
| e5 L9 R1 whitened | 0.0623 | bare-string entries |
| e5 L9 R2 centered | 0.0563 | whitening *is* worth it end to end |

Four things follow.

1. **Layer 12 beats layer 9 by 30%.** Cross-representation self-hit — the metric
   §A.4 selects on — is *anti*-correlated with retrieval across layers here, while
   related-term MRR (23x at L9, 210x at L12) tracks it. Identity is not the property
   that matters; semantic organisation is.
2. **Whitening is vindicated where H3 said it should not be.** It does not make
   profiles peakier (H3 fails), but it is worth 0.096 vs 0.056 in MRR@10 against
   centering alone. Peakiness was the wrong thing to measure it by.
3. **e5 beats bge end to end** (0.1251 vs 0.1083), so the sense-threshold conflict
   in §A.4 resolves in favour of the configuration the primary metric chose.
4. **R2 ~ R3 > R1**, confirming Pilot B's choice without relying on its
   structurally-favoured in-context self-hit.

At layer 12 sense accuracy is 0.850 (R1, R2) and 0.862 (R3) — so the §A.4
constraint that could not be satisfied at layer 9 *is* satisfied at the layer
retrieval selects. Stability is worse there (k* = 50 rather than 20): later-layer
prototypes need more occurrences.

Query-side transform, measured end to end: own transform 0.1250 vs the shared
document transform 0.1237, paired-bootstrap difference +0.0013 with CI
[-0.0013, +0.0038]. The large distribution shift §0.5 measures (mean shift 0.16-0.54)
does **not** translate into retrieval impact.

**Pilot C verdict: weak** (0.665x BM25), so Pilot D runs the LoRA arm from the start.

### Pilot D — setup findings before the results

* The **lambda sweep is degenerate**: nnz_d lands at 46.3-46.4 for every lambda_d in
  {1e-4, 3e-4, 1e-3}, and the FLOPS term is 5e-4 of the cross-entropy at the top of
  the grid. Density is set by the *learned threshold*, not by the regulariser, and it
  settles well below C's 120-per-document operating point. This is §A.4's H3-failure
  branch playing out exactly as written.
* **A bug worth recording**: V2's staggered refresh initially averaged every
  occurrence of a refreshed word found in the gathered passages, rather than the
  entry's own sampled contexts. Frequent entries drifted towards the corpus mean and
  became hubs — one refresh took nnz_d from 53 to 685 and CE from 2.8 to 4.1. Fixed
  by restricting accumulation to the sampled (passage, entry) pairs; the corrected
  refresh reproduces the original prototypes to cosine 0.999998 when the encoder has
  not yet moved.
