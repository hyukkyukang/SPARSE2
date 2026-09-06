# Dynamic-vocabulary learned sparse retrieval — pilot findings

A sparse retriever whose output dimensions are **defined by text** rather than
by rows of a learned matrix. Entry *j* is a text *t_j*; its vector is `f(t_j)`
for the same encoder that embeds the input, so a new entry is added by encoding
it — no new parameters, no retraining. These pilots test whether that premise
survives contact with data.

Everything below is from `results/*.json`. Per-pilot tables are in
`reports/pilot{A,B,C,D,E}.md`; departures from the written protocol are in
`notes/deviations.md`; the chronological record is `notes/log.md`.

## 0. The pipeline reproduces published numbers

| system | full collection MRR@10 | published |
|---|---|---|
| BM25 (k1=0.82, b=0.68) | 0.1874 | 0.1875 |
| SPLADE++ CoCondenser-ED | 0.3827 | 0.383 |
| bge-base | 0.3498 | ~0.35 |
| e5-base | 0.3542 | ~0.35 |

Vocabulary V: **30,000** entries from 1,220,347 alphabetic types in P; 67,676 cleared the 100-occurrence floor, and the bottom decile's minimum collection frequency is 498 — so no entry has a noisy prototype and V was not cut.

## The short version

1. **Token states do carry term-level semantics against text-defined entries.** Whitened cross-representation self-hit@10 is 0.923 (threshold 0.4) and related-term MRR is 23x the random baseline (threshold 3x). H1 holds with room to spare.
2. **Untrained, the projection retrieves at 0.67x BM25** on C1 — real signal, well short of the 0.8x the protocol hoped for. Term lists are excellent (see the plausibility annotation); the weighting is what is missing.
3. **Three of the protocol's expectations were falsified**, each in an informative direction: whitening does not sharpen profiles (H3), bare-string entries are *less* hubby than prototypes rather than more (H5), and the metric §A.4 selects layers on is anti-correlated with retrieval across layers.
4. **Pilot D verdict:** V1 passes on ['rare'] and falls short elsewhere; recipe = *frozen encoder + light head*. 1 of 4 runs meet H10. No systematic signed gap; if no arm passes the problem is representational, not calibrational (§D.7 final branch).

## Hypotheses

| hypothesis | claim | verdict | evidence |
|---|---|---|---|
| **H1** | token states carry term-level semantics vs text-defined entries | **supported** | self-hit@10 0.923; related-term MRR 23x random |
| **H2** | best layer is not 12 for the CLS-pooled model; e5's is 8–11 | **supported** | best layers {"bge": 10, "e5": 9} |
| **H3** | whitening raises z-gap ≥2x and cuts nnz/token ~10x | **falsified** | z-gap ratio <1 at every layer below 12; nnz cut 2.2x, and most of that comes from centering, not whitening |
| **H4** | sense ≥0.8; ColBERTv2 above both candidates on every metric | **partly** | max sense 0.814 at the selected layer's encoder; ColBERTv2 leads on identity but trails e5 on related-term MRR |
| **H5** | R1 is hubbier than R2 and has a weaker df–freq correlation | **half reversed** | hub skew R1 3.6 vs R2 17.2 (reversed); Spearman 0.25 vs 0.57 (as predicted) |
| **H6** | stability at k=10 reaches 0.8x the k=50 value | **narrowly not** | ratio 0.788 at layer 9; k* = 20 (layer 9) and 50 (layer 12) |
| **H7** | R1 under its own transform beats the shared token-side transform | **supported** | self-hit 0.923 vs 0.837; the shared condition's SPLADE Jaccard collapses to 0.002 |
| **H8** | best untrained run ≥0.8x BM25 on C1 | **not met** | 0.67x — the 'weak' branch, so Pilot D keeps the LoRA arm |
| **H9** | Spearman vs dense 0.5–0.7; plausibility ≥80% | **split** | Spearman 0.296 (below range); plausibility 0.97 (met) |
| **H10** | ρ ≥ 0.8, CI excludes 0.5, gap ≤10%, |gap_r| ≤ log 1.5, beats C-alias | **V1 passes on ['rare'] and falls short elsewhere** |  |
| **H11** | V3 has the best seen-only MRR but ρ ≤ 0.5 | **not supported** |  |
| **H14** | C-rand is clearly worse than V1 | **supported** |  |

## Decisions, in order

### Pilot A — geometry

* `chosen_encoder` = "e5"
* `chosen_layer` = 9
* `top2_layers` = [9, 10]
* `chosen_hit10` = 0.9230877192982456
* `chosen_sense` = 0.6805205709487825
* `metric_peaks_by_layer` = {"e5": {"hit10": 9, "related_mrr": 12, "sense": 12}, "bge": {"hit10": 10, "related_mrr": 12, "sense": 12}, "colbert": {"hit10": 12, "related_mrr": 12, "sense": 12}}
* `literal_rule_pick` = {"encoder": "bge", "layer": 12, "hit10": 0.5070035087719298, "sense": 0.8136020151133502}
* `selection_conflict` = true

> **Pilot C overrode this layer.** §A.4 selects on cross-representation self-hit@10, which peaks at layer 9; end-to-end retrieval on C1 prefers layer 12 by +0.0288 MRR@10 (+30% relative). Related-term MRR — the rule's *tie-breaker* — is the metric that tracks retrieval across layers (23x at layer 9, 210x at layer 12), not the primary one. Pilots B, C and D all use layer 12.

### Pilot B — entry representation (e5_L9)

* `chosen_rep` = "R2"
* `hub_share` = {"R1": 0.04434850066900253, "R2": 0.16079850494861603, "R3": 0.18497949838638306}
* `hub_skew` = {"R1": 3.614572048187256, "R2": 17.154434204101562, "R3": 15.827569007873535}
* `k_star` = 20
* `H5_verdict` = "reversed: R1 is markedly LESS hubby than R2"
* `H6_verdict` = "narrowly not supported"
* `H7_verdict` = "supported"
* `runaway_max` = 0.0009333333333333333
* `pilotE_mandatory` = false

### Pilot B — entry representation (e5_L10)

* `chosen_rep` = "R2"
* `hub_share` = {"R1": 0.06672199815511703, "R2": 0.16585299372673035, "R3": 0.17823849618434906}
* `hub_skew` = {"R1": 16.95754051208496, "R2": 16.38713836669922, "R3": 16.064083099365234}
* `k_star` = 20
* `H5_verdict` = "supported"
* `H6_verdict` = "narrowly not supported"
* `H7_verdict` = "supported"
* `runaway_max` = 0.0025666666666666667
* `pilotE_mandatory` = false

### Pilot B — entry representation (e5_L12)

* `chosen_rep` = "R3"
* `hub_share` = {"R1": 0.07355199754238129, "R2": 0.09764599800109863, "R3": 0.09223199635744095}
* `hub_skew` = {"R1": 4.098466873168945, "R2": 8.708099365234375, "R3": 6.455414295196533}
* `k_star` = 50
* `H5_verdict` = "reversed: R1 is markedly LESS hubby than R2"
* `H6_verdict` = "narrowly not supported"
* `H7_verdict` = "supported"
* `runaway_max` = 0.0
* `pilotE_mandatory` = false

### Pilot C — training-free retrieval

* `encoder` = "e5"
* `layer` = 12
* `rep` = "R2"
* `k_d` = 128
* `k_q` = 16
* `saturation` = "log1p"
* `nnz_d` = 120
* `nnz_q` = 30
* `tau` = 0.21997069567419203
* `verdict` = "weak"
* `ratio_to_bm25` = 0.6650116250685782
* `include_lora_arm` = true

#### References on C1

| system | MRR@10 | R@100 | R@1000 |
|---|---|---|---|
| bm25 | 0.1882 | 0.6722 | 0.9143 |
| spladepp | 0.3817 | 0.9089 | 0.9908 |
| dense_e5 | 0.3542 | 0.8872 | 0.9903 |

### Pilot D — held-out vocabulary

* `recipe` = "frozen encoder + light head"
* `passing_runs` = ["V1_rare"]
* `H10_verdict` = "V1 passes on ['rare'] and falls short elsewhere"
* `H11_verdict` = "not supported"
* `H12_vd_effect` = {"V1": [null, 0.6849068734413967], "V2": [null, null], "V3": [null, null]}
* `H13_difficulty_order` = {"random": 0.6849068734413967, "cluster": 0.5666578016273517, "rare": 0.9322161080540272}
* `H14_verdict` = "supported"
* `systematic_gap_runs` = ["V1_cluster"]
* `pilotE_triggered` = false
* `summary` = "1 of 4 runs meet H10. No systematic signed gap; if no arm passes the problem is representational, not calibrational (\u00a7D.7 final branch)."

### Pilot E — insertion-time calibration

_not run_

## Measurement caveats we can quantify

* **Sense labels.** Manual verification of a 10% sample found 28 of 240 labels wrong (agreement 0.883), traced to ambiguous inflections and over-generic cues. After dropping 61 indicators, agreement on a fresh sample is 0.963. Every sense number here uses the corrected labels.
* **Why the untrained run is weak.** Query profiles are peaked (top-1 carries 18.5% of the mass, participation ratio 14 of 30 non-zeros) but document profiles are flat (3.5%, 63 of 119). Untrained max-pooled cosine spreads a document's mass over ~63 effectively-equal dimensions.
* **The query-side transform buys nothing measurable.** §0.5 mandates a separate transform when the query token distribution differs by >5%; ours differs by 16–54%. End to end the difference is +0.0013 MRR@10, CI [-0.0013, +0.0038] — indistinguishable from zero.
* **The FLOPS regulariser is inert at the protocol's λ.** Document density lands at 46.3–46.4 non-zeros across a 10x range of λ_d, and the penalty is ~5e-4 of the cross-entropy at the top of the grid. Sparsity comes from the learned threshold.
* **C1 is mildly optimistic for us**: it contains every reference system's own candidates but not ours (they cannot exist before the method does). The full-corpus run is the headline number.
