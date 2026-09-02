# Pilot C — training-free end-to-end retrieval on C1

Encoder **e5**, layer **12**, entry representation **R2**; C1 = 1,852,472 passages, dev-small queries.

## References on C1

| system | MRR@10 | R@100 | R@1000 |
|---|---|---|---|
| bm25 | 0.1882 | 0.6722 | 0.9143 |
| spladepp | 0.3817 | 0.9089 | 0.9908 |
| dense_e5 | 0.3542 | 0.8872 | 0.9903 |

## Grid (36 cells, all re-truncations of one stored profile set)

| cell | MRR@10 | 95% CI | R@100 | R@1000 |
|---|---|---|---|---|
| kd128_kq16_nnz120_log1p | 0.1251 | [0.1190, 0.1315] | 0.5862 | 0.8457 |
| kd64_kq16_nnz120_log1p | 0.1249 | [0.1188, 0.1313] | 0.5866 | 0.8423 |
| kd256_kq16_nnz120_log1p | 0.1249 | [0.1188, 0.1312] | 0.5853 | 0.8452 |
| kd128_kq32_nnz120_log1p | 0.1246 | [0.1184, 0.1309] | 0.5828 | 0.8457 |
| kd64_kq32_nnz120_log1p | 0.1245 | [0.1183, 0.1310] | 0.5831 | 0.8426 |
| kd128_kq64_nnz120_log1p | 0.1243 | [0.1182, 0.1307] | 0.5822 | 0.8460 |
| kd64_kq64_nnz120_log1p | 0.1243 | [0.1181, 0.1307] | 0.5834 | 0.8427 |
| kd256_kq32_nnz120_log1p | 0.1241 | [0.1179, 0.1304] | 0.5816 | 0.8454 |
| kd256_kq64_nnz120_log1p | 0.1237 | [0.1176, 0.1301] | 0.5817 | 0.8452 |
| kd128_kq32_nnz120_none | 0.1203 | [0.1142, 0.1265] | 0.5743 | 0.8441 |
| kd64_kq32_nnz120_none | 0.1202 | [0.1140, 0.1264] | 0.5727 | 0.8427 |
| kd64_kq64_nnz120_none | 0.1200 | [0.1138, 0.1263] | 0.5726 | 0.8430 |
| kd128_kq64_nnz120_none | 0.1200 | [0.1139, 0.1262] | 0.5741 | 0.8446 |
| kd256_kq32_nnz120_none | 0.1200 | [0.1139, 0.1262] | 0.5725 | 0.8430 |
| kd128_kq16_nnz120_none | 0.1197 | [0.1137, 0.1260] | 0.5756 | 0.8431 |
| kd256_kq64_nnz120_none | 0.1196 | [0.1135, 0.1259] | 0.5723 | 0.8438 |
| kd256_kq16_nnz120_none | 0.1193 | [0.1133, 0.1256] | 0.5750 | 0.8431 |
| kd64_kq16_nnz120_none | 0.1191 | [0.1132, 0.1254] | 0.5727 | 0.8413 |
| kd64_kq64_nnz60_log1p | 0.1104 | [0.1045, 0.1165] | 0.5455 | 0.8206 |
| kd64_kq32_nnz60_log1p | 0.1103 | [0.1044, 0.1164] | 0.5455 | 0.8206 |
| kd256_kq64_nnz60_log1p | 0.1101 | [0.1042, 0.1162] | 0.5456 | 0.8197 |
| kd128_kq32_nnz60_log1p | 0.1101 | [0.1042, 0.1162] | 0.5456 | 0.8197 |
| kd256_kq32_nnz60_log1p | 0.1100 | [0.1041, 0.1161] | 0.5456 | 0.8197 |
| kd128_kq64_nnz60_log1p | 0.1099 | [0.1041, 0.1160] | 0.5456 | 0.8197 |
| kd256_kq16_nnz60_log1p | 0.1099 | [0.1041, 0.1160] | 0.5455 | 0.8190 |
| kd128_kq16_nnz60_log1p | 0.1098 | [0.1040, 0.1159] | 0.5455 | 0.8190 |
| kd64_kq16_nnz60_log1p | 0.1098 | [0.1039, 0.1159] | 0.5460 | 0.8197 |
| kd256_kq64_nnz60_none | 0.1031 | [0.0973, 0.1090] | 0.5342 | 0.8176 |
| kd128_kq64_nnz60_none | 0.1029 | [0.0972, 0.1089] | 0.5342 | 0.8176 |
| kd256_kq32_nnz60_none | 0.1029 | [0.0971, 0.1089] | 0.5342 | 0.8178 |
| kd128_kq32_nnz60_none | 0.1029 | [0.0971, 0.1089] | 0.5342 | 0.8178 |
| kd64_kq64_nnz60_none | 0.1027 | [0.0970, 0.1087] | 0.5354 | 0.8182 |
| kd64_kq32_nnz60_none | 0.1026 | [0.0968, 0.1086] | 0.5354 | 0.8182 |
| kd128_kq16_nnz60_none | 0.1020 | [0.0963, 0.1079] | 0.5347 | 0.8172 |
| kd256_kq16_nnz60_none | 0.1020 | [0.0962, 0.1079] | 0.5346 | 0.8170 |
| kd64_kq16_nnz60_none | 0.1019 | [0.0962, 0.1078] | 0.5356 | 0.8170 |

## Operating point and sparsity

| target nnz | tau | mean nnz (S) | token percentile |
|---|---|---|---|
| doc 120 | 0.2200 | 120.1 | 99.5995 |
| doc 60 | 0.2678 | 60.1 | 99.7995 |
| query 30 | 0.2249 | 30.0 | — |
| query 15 | 0.2769 | 15.0 | — |

## Score preservation vs the dense encoder

* Spearman over the dense top-100 within C1: mean 0.296, median 0.311
* Jaccard of top-100 with dense: 0.224

* SPLADE++ top-20 Jaccard on the truncated representation: 0.357 (coverage 0.85)

## Decision (§C.4)

* best cell **kd128_kq16_nnz120_log1p**: MRR@10 = 0.1251
* BM25 on C1 = 0.1882; ratio = **0.67x** (H8 wants >= 0.80x)
* dense = 0.3542, SPLADE++ = 0.3817; is the untrained run between BM25 and dense? no
* verdict: **weak** — Pilot D includes the LoRA arm from the start

```json
{
  "encoder": "e5",
  "layer": 12,
  "rep": "R2",
  "r1_prefix": "none",
  "tau": 0.21997069567419203,
  "tau_q": 0.22485350817373728,
  "k_d": 128,
  "k_q": 16,
  "saturation": "log1p",
  "nnz_d": 120,
  "nnz_q": 30,
  "lam_d": 0.0003,
  "lam_q": 0.0009,
  "verdict": "weak",
  "ratio_to_bm25": 0.6650116250685782,
  "include_lora_arm": true
}
```