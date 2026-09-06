# Pilot C — training-free end-to-end retrieval on C1

Encoder **e5**, layer **12**, entry representation **R2**; C1 = 1,469,456 passages, dev-small queries.

## References on C1

| system | MRR@10 | R@100 | R@1000 |
|---|---|---|---|
| bm25 | 0.1889 | 0.6759 | 0.9224 |
| spladepp | 0.3817 | 0.9100 | 0.9933 |
| dense_e5 | 0.3564 | 0.9223 | 0.9934 |

## Grid (36 cells, all re-truncations of one stored profile set)

| cell | MRR@10 | 95% CI | R@100 | R@1000 |
|---|---|---|---|---|
| kd64_kq16_nnz120_log1p | 0.1298 | [0.1236, 0.1363] | 0.6152 | 0.8621 |
| kd128_kq16_nnz120_log1p | 0.1294 | [0.1231, 0.1358] | 0.6171 | 0.8647 |
| kd256_kq16_nnz120_log1p | 0.1293 | [0.1231, 0.1357] | 0.6169 | 0.8643 |
| kd64_kq32_nnz120_log1p | 0.1292 | [0.1229, 0.1357] | 0.6123 | 0.8634 |
| kd128_kq32_nnz120_log1p | 0.1289 | [0.1226, 0.1354] | 0.6143 | 0.8660 |
| kd64_kq64_nnz120_log1p | 0.1288 | [0.1225, 0.1353] | 0.6119 | 0.8638 |
| kd256_kq32_nnz120_log1p | 0.1284 | [0.1222, 0.1348] | 0.6136 | 0.8658 |
| kd128_kq64_nnz120_log1p | 0.1281 | [0.1218, 0.1346] | 0.6133 | 0.8655 |
| kd256_kq64_nnz120_log1p | 0.1278 | [0.1216, 0.1342] | 0.6126 | 0.8652 |
| kd64_kq32_nnz120_none | 0.1243 | [0.1181, 0.1307] | 0.6034 | 0.8606 |
| kd128_kq32_nnz120_none | 0.1242 | [0.1181, 0.1306] | 0.6040 | 0.8640 |
| kd256_kq32_nnz120_none | 0.1241 | [0.1178, 0.1305] | 0.6032 | 0.8638 |
| kd64_kq64_nnz120_none | 0.1240 | [0.1177, 0.1304] | 0.6029 | 0.8604 |
| kd128_kq64_nnz120_none | 0.1239 | [0.1178, 0.1304] | 0.6042 | 0.8641 |
| kd128_kq16_nnz120_none | 0.1238 | [0.1177, 0.1302] | 0.6055 | 0.8614 |
| kd256_kq64_nnz120_none | 0.1237 | [0.1176, 0.1301] | 0.6037 | 0.8640 |
| kd256_kq16_nnz120_none | 0.1235 | [0.1174, 0.1299] | 0.6049 | 0.8616 |
| kd64_kq16_nnz120_none | 0.1233 | [0.1172, 0.1296] | 0.6035 | 0.8582 |
| kd64_kq32_nnz60_log1p | 0.1145 | [0.1085, 0.1207] | 0.5772 | 0.8398 |
| kd64_kq64_nnz60_log1p | 0.1145 | [0.1085, 0.1206] | 0.5770 | 0.8398 |
| kd128_kq16_nnz60_log1p | 0.1140 | [0.1080, 0.1202] | 0.5765 | 0.8383 |
| kd64_kq16_nnz60_log1p | 0.1139 | [0.1080, 0.1201] | 0.5766 | 0.8392 |
| kd256_kq64_nnz60_log1p | 0.1139 | [0.1080, 0.1201] | 0.5762 | 0.8392 |
| kd128_kq64_nnz60_log1p | 0.1139 | [0.1080, 0.1201] | 0.5762 | 0.8392 |
| kd256_kq32_nnz60_log1p | 0.1139 | [0.1080, 0.1201] | 0.5760 | 0.8392 |
| kd128_kq32_nnz60_log1p | 0.1139 | [0.1079, 0.1201] | 0.5760 | 0.8392 |
| kd256_kq16_nnz60_log1p | 0.1139 | [0.1080, 0.1201] | 0.5765 | 0.8383 |
| kd128_kq64_nnz60_none | 0.1067 | [0.1007, 0.1127] | 0.5678 | 0.8364 |
| kd256_kq64_nnz60_none | 0.1066 | [0.1007, 0.1127] | 0.5678 | 0.8364 |
| kd128_kq32_nnz60_none | 0.1066 | [0.1007, 0.1127] | 0.5678 | 0.8364 |
| kd256_kq32_nnz60_none | 0.1066 | [0.1006, 0.1126] | 0.5678 | 0.8364 |
| kd64_kq64_nnz60_none | 0.1066 | [0.1006, 0.1126] | 0.5679 | 0.8371 |
| kd64_kq32_nnz60_none | 0.1064 | [0.1005, 0.1124] | 0.5678 | 0.8371 |
| kd256_kq16_nnz60_none | 0.1060 | [0.1001, 0.1120] | 0.5669 | 0.8357 |
| kd128_kq16_nnz60_none | 0.1060 | [0.1001, 0.1120] | 0.5669 | 0.8357 |
| kd64_kq16_nnz60_none | 0.1057 | [0.0999, 0.1118] | 0.5683 | 0.8365 |

## Operating point and sparsity

| target nnz | tau | mean nnz (S) | token percentile |
|---|---|---|---|
| doc 120 | 0.2200 | 120.2 | 99.5995 |
| doc 60 | 0.2678 | 60.1 | 99.7995 |
| query 30 | 0.2249 | 30.0 | — |
| query 15 | 0.2769 | 15.0 | — |

## Score preservation vs the dense encoder

* Spearman over the dense top-100 within C1: mean 0.318, median 0.336
* Jaccard of top-100 with dense: 0.267

* SPLADE++ top-20 Jaccard on the truncated representation: 0.351 (coverage 0.85)

## Decision (§C.4)

* best cell **kd64_kq16_nnz120_log1p**: MRR@10 = 0.1298
* BM25 on C1 = 0.1889; ratio = **0.69x** (H8 wants >= 0.80x)
* dense = 0.3564, SPLADE++ = 0.3817; is the untrained run between BM25 and dense? no
* verdict: **weak** — Pilot D includes the LoRA arm from the start

```json
{
  "encoder": "e5",
  "layer": 12,
  "rep": "R2",
  "r1_prefix": "none",
  "tau": 0.21997069567419203,
  "tau_q": 0.22485350817373728,
  "k_d": 64,
  "k_q": 16,
  "saturation": "log1p",
  "nnz_d": 120,
  "nnz_q": 30,
  "lam_d": 0.0003,
  "lam_q": 0.0009,
  "verdict": "weak",
  "ratio_to_bm25": 0.6870622573947582,
  "include_lora_arm": true
}
```