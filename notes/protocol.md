# Pilot protocol — operative summary

The full protocol document is the source of record; this file condenses the parts
the code implements and is checked against: the fixed setup, the hypotheses, and
the decision rules. Section numbers match the source.

## §0 Shared setup
* **Encoders.** e5-base-v2 (mean pooled, `query: `/`passage: `), bge-base-en-v1.5
  (CLS, query instruction), ColBERTv2 (128-d token embeddings, final layer, Pilot A
  ceiling), SPLADE++ CoCondenser-EnsembleDistil (LSR reference). Hidden-layer sweep
  4–12.
* **Data.** P = prototype pool (collection minus Q minus dev qrels), Q = 40k probe/bank
  pool, S = 100k statistics sample, C1 ≈ 1.5M retrieval sub-corpus, B_H / B_Q = 200k
  background token states each.
* **V.** Lowercase alphabetic words with ≥100 occurrences in P; top 30,000; stopwords
  kept but flagged; collection frequency stored for stratification.
* **Word units.** Wordpieces merged to pre-tokenizer words (mean of piece states);
  specials, padding, prompt prefixes and pure punctuation dropped.
* **Whitening (§0.5).** ZCA per space, shrinkage ε = 0.01·tr(Σ)/d, estimated once and
  frozen. Token side from B_H (docs) and B_Q (queries); entry side per representation.
* **Rule (§0.6).** a_ij = cos(h̃_i, ṽ_j); p_j = max_i a_ij; s_j = ReLU(p_j − τ); top-k;
  optional log1p; score = Σ_j s_j(q)s_j(d). τ set so mean nnz ≈ 120 (docs) / 30 (queries).
* **Statistics (§0.7).** Paired bootstrap over queries, 10,000 resamples, 95% CI, on
  every effectiveness comparison. Profiles streamed, never stored.

## Hypotheses and decision rules

| id | claim | threshold |
|---|---|---|
| H1 | token states carry term-level semantics vs text-defined entries | whitened cross-rep self-hit@10 ≥ 0.4; related-term MRR ≥ 3× random |
| H2 | best layer is not 12 for the CLS-pooled model; e5's best is 8–11 | — |
| H3 | whitening raises z-gap ≥2× and cuts nnz/token ~10× | — |
| H4 | sense accuracy ≥ 0.8; ColBERTv2 above both candidates but not by much | — |
| H5 | R1 is hubbier than R2, weaker df–freq correlation, lower in-context self-hit | — |
| H6 | R2 stability at k=10 ≥ 0.8 × its k=50 value | k* = smallest such k |
| H7 | R1 under its own transform beats R1 under the shared transform | — |
| H8 | best untrained run ≥ 0.8 × BM25 MRR@10 on C1, between BM25 and dense | — |
| H9 | Spearman vs dense ≈ 0.5–0.7 at k_d=128; plausibility ≥ 80% | — |
| H10 | **pass criterion**: ρ ≥ 0.8 (CI excludes 0.5), gap to oracle on Q_H ≤ 10% rel., \|gap_r\| ≤ log 1.5, ρ beats C-alias | go/no-go |
| H11 | V3 has best seen-only MRR but ρ ≤ 0.5 with a systematic signed gap | — |
| H12 | VD closes much of V3's gap, neutral for V1 | — |
| H13 | difficulty orders random < cluster < rare | — |
| H14 | **control**: C-rand clearly worse than V1 | stop if not |
| H15 | Z-calibration closes ≥80% of an over-firing gap | Pilot E |

**A**: select (encoder, layer) = argmax whitened cross-rep self-hit@10, sense ≥ 0.8
required, related-term MRR as tie-break; carry the top-2 layers into B.
**B**: lowest hubness among representations not worse on in-context self-hit; k* from
the stability curve. If R1 ≈ R2, prefer R1 and report it.
**C**: pass ≥0.8×BM25 → D with the knee operating point; 0.4–0.8× → D with the LoRA arm
from the start; <0.4× and Spearman <0.3 → redesign the head and rerun C.
**D**: V1 passes → frozen encoder + light head. Only V1+VD → VD is a core component.
V1 fails, V2+VD passes → LoRA with staggered refresh. All fail with a systematic signed
gap → Pilot E. All fail with no systematic gap → representational, reframe.
**E**: adopt the simplest variant reaching ρ ≥ 0.8; none reaching 0.6 → representational.
