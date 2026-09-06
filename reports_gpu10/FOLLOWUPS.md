# Follow-up experiments — what the pilots left open

Every number here comes from the rebuild on dslab-gpu10 (notes/deviations.md D12), so arms are compared to baselines re-run in the same rebuild, never to the A100 numbers. rho is the recovery ratio of §D.6 (share of the oracle's held-out benefit that survives insertion); signed gap_r > 0 means held-out entries over-fire relative to seen entries of the same frequency decile; |gap_r| <= log 1.5 = 0.405 is the H10 calibration bound.

## 1. Setup on this machine

C1 = 1,469,456 passages (qrels 7,433, BM25 top-100 608,032, SPLADE++ top-100 640,251, random 500,000; no dense candidates — D12).

| reference on C1 | MRR@10 | R@100 | R@1000 |
|---|---|---|---|
| bm25 | 0.1889 | 0.6759 | 0.9224 |
| spladepp | 0.3817 | 0.9100 | 0.9933 |
| dense_e5 | 0.3564 | 0.9223 | 0.9934 |

## 2. Untrained retrieval at layer 12: contextual prototypes vs bare-string entries

Pilot C never ran bare-string (R1) entries at the layer that retrieves best; if R1 were close to R2 here, insertion would cost one forward pass instead of ~50 occurrences.

| entries | best cell | MRR@10 | 95% CI | R@100 | R@1000 | vs BM25 |
|---|---|---|---|---|---|---|
| R2 contextual prototype | kd64_kq16_nnz120_log1p | 0.1298 | [0.1236, 0.1363] | 0.6152 | 0.8621 | 0.69x |
| R1 bare string | kd128_kq32_nnz120_log1p | 0.1416 | [0.1350, 0.1485] | 0.6495 | 0.9048 | 0.75x |

Bare-string entries reach **1.09x** the prototype MRR@10 at layer 12 (at layer 9 the pilot found 0.65x). They are close: one-forward-pass insertion is back on the table.

## 3. Cluster split: does semantic vocabulary dropout fix over-firing?

The pilot's one clear negative: entries from an unseen *region* over-fire (signed gap_r +0.47 on the A100 run) and recover only 57% of the oracle's benefit. Entry-wise dropout cannot address it (every dropped entry keeps trained neighbours); cluster-wise dropout removes whole regions per step; the meta arm additionally keeps some regions out of the ranking loss for the whole run and trains their activation rate to match trained entries (a training-time gap_r). The linear head is the capacity control: if it cannot over-fire, co-adaptation of the MLP is the cause.

| arm | MRR@10 all | seen-only | rho (Q_H) | rho (lex) | gap to oracle | signed gap_r | abs gap_r | C-alias MRR | nnz d/q | n(Q_H) |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 (frozen, no dropout) | 0.1922 | 0.1807 | 0.313 [0.23, 0.40] | 0.315 | 0.195 | 0.386 | 0.386 | 0.1157 ✓ | 77/25 | 1732 |
| V1 + entry-wise VD | 0.1925 | 0.1821 | 0.288 [0.21, 0.37] | 0.283 | 0.203 | 0.345 | 0.345 | 0.1262 ✓ | 117/38 | 1732 |
| V1 + cluster-wise VD | 0.1935 | 0.1849 | 0.273 [0.19, 0.36] | 0.262 | 0.195 | 0.394 | 0.394 | 0.1246 ✓ | 133/44 | 1732 |
| V1 + cluster VD + meta calibration | 0.1895 | 0.1820 | 0.232 [0.15, 0.32] | 0.249 | 0.223 | 0.366 | 0.366 | 0.1267 ✓ | 170/58 | 1732 |
| V1, linear head | 0.1818 | 0.1752 | 0.198 [0.11, 0.29] | 0.221 | 0.272 | 0.464 | 0.464 | 0.1048 ✓ | 87/26 | 1732 |
| V1, bare-string entries | 0.2214 | 0.2331 | -0.048 [-0.15, 0.05] | -0.011 | 0.171 | 0.769 | 0.769 | 0.1880 ✓ | 218/98 | 1973 |
| V1, bare-string entries + cluster-wise VD | 0.2366 | 0.2502 | -0.153 [-0.24, -0.07] | -0.141 | 0.092 | 0.856 | 0.856 | 0.2217 ✓ | 352/160 | 1973 |
| V1, entries whitened with the token-side transform | 0.2374 | 0.2248 | 0.333 [0.26, 0.41] | 0.356 | 0.116 | 0.307 | 0.307 | 0.1769 ✓ | 131/41 | 1801 |
| V1 + shared entry-side map | 0.2452 | 0.2728 | -0.565 [-0.72, -0.43] | -0.579 | 0.240 | 1.469 | 1.469 | 0.2122 ✓ | 679/204 | 1757 |
| V1 + entry-side map + tail normalisation | 0.2651 | 0.2770 | -0.309 [-0.47, -0.16] | -0.344 | 0.115 | 1.230 | 1.230 | 0.2428 ✓ | 698/207 | 1764 |
| V1, k=100 prototypes | 0.1918 | 0.1798 | 0.302 [0.22, 0.38] | 0.305 | 0.198 | 0.391 | 0.391 | 0.1126 ✓ | 76/25 | 1745 |
| V1 + tail normalisation | 0.1849 | 0.1607 | 0.636 [0.57, 0.70] | 0.631 | 0.063 | 0.179 | 0.179 | 0.1119 ✓ | 58/19 | 1854 |
| V1, seed 2 | 0.1914 | 0.1791 | 0.313 [0.23, 0.40] | 0.320 | 0.202 | 0.379 | 0.379 | 0.1146 ✓ | 77/25 | 1732 |
| V1 + tail normalisation, seed 2 | 0.1842 | 0.1599 | 0.646 [0.58, 0.72] | 0.627 | 0.063 | 0.158 | 0.158 | 0.1120 ✓ | 58/19 | 1854 |
| V1 + moment normalisation (contrast) | 0.1803 | 0.1585 | 0.582 [0.51, 0.66] | 0.578 | 0.104 | 0.335 | 0.335 | 0.1025 ✓ | 66/21 | 1847 |
| V1 + tail normalisation + token-side entry transform | 0.2050 | 0.1845 | 0.516 [0.44, 0.59] | 0.503 | 0.093 | 0.141 | 0.141 | 0.1458 ✓ | 83/24 | 1882 |
| V1, bare-string entries + tail normalisation | 0.2128 | 0.2052 | 0.289 [0.21, 0.37] | 0.299 | 0.066 | 0.210 | 0.210 | 0.1756 ✓ | 121/55 | 1926 |
| V3 (full fine-tuning) | 0.2426 | 0.2289 | 0.366 [0.28, 0.45] | 0.383 | -0.053 | 0.345 | 0.345 | 0.1912 ✓ | 150/36 | 1732 |

Smallest |gap_r|: **V1 + tail normalisation + token-side entry transform** (0.141); highest rho: **V1 + tail normalisation, seed 2** (0.646).

Training tail (mean of the last 10 logged steps):

| arm | steps | CE | in-batch acc | nnz(d) | calibration loss | t | b |
|---|---|---|---|---|---|---|---|
| V1 (frozen, no dropout) | 3100 | 1.647 | 0.557 | 56 | 0.0000 | 20.1 | 4.17 |
| V1 + entry-wise VD | 3100 | 1.782 | 0.523 | 61 | 0.0000 | 20.2 | 4.10 |
| V1 + cluster-wise VD | 3100 | 2.138 | 0.470 | 70 | 0.0000 | 20.2 | 4.11 |
| V1 + cluster VD + meta calibration | 3100 | 2.124 | 0.477 | 86 | 0.0459 | 20.2 | 4.11 |
| V1, linear head | 3100 | 1.702 | 0.537 | 63 | 0.0000 | 19.9 | 4.29 |
| V1, bare-string entries | 3100 | 1.220 | 0.641 | 155 | 0.0000 | 20.1 | 2.96 |
| V1, bare-string entries + cluster-wise VD | 3100 | 1.504 | 0.578 | 168 | 0.0000 | 20.2 | 2.89 |
| V1, entries whitened with the token-side transform | 3100 | 1.469 | 0.588 | 103 | 0.0000 | 20.0 | 4.28 |
| V1 + shared entry-side map | 3100 | 1.102 | 0.645 | 430 | 0.0000 | 19.9 | 4.36 |
| V1 + entry-side map + tail normalisation | 3100 | 1.131 | 0.641 | 508 | 0.0000 | 19.9 | 4.35 |
| V1, k=100 prototypes | 3100 | 1.642 | 0.558 | 56 | 0.0000 | 20.1 | 4.18 |
| V1 + tail normalisation | 3100 | 2.010 | 0.508 | 45 | 0.0000 | 20.2 | 4.12 |
| V1, seed 2 | 3100 | 1.563 | 0.578 | 55 | 0.0000 | 20.1 | 4.17 |
| V1 + tail normalisation, seed 2 | 3100 | 1.942 | 0.509 | 44 | 0.0000 | 20.2 | 4.12 |
| V1 + moment normalisation (contrast) | 3100 | 2.019 | 0.514 | 48 | 0.0000 | 20.1 | 4.14 |
| V1 + tail normalisation + token-side entry transform | 3100 | 1.783 | 0.548 | 62 | 0.0000 | 20.1 | 4.19 |
| V1, bare-string entries + tail normalisation | 3100 | 1.572 | 0.582 | 94 | 0.0000 | 20.1 | 2.91 |
| V3 (full fine-tuning) | 3000 | 1.331 | 0.616 | 98 | 0.0000 | 20.0 | 4.24 |

## 4. Pilot E: insertion-time calibration on the cluster model

| variant | rho | 95% CI | gap to oracle | signed gap_r | gap_w | nnz(d) | Δnnz(d) |
|---|---|---|---|---|---|---|---|
| raw | 0.313 | [0.23, 0.40] | 0.195 | 0.3857 | -0.0391 | 77 | +0 |
| Z | 0.494 | [0.42, 0.57] | 0.084 | 0.2293 | -0.0219 | 71 | -6 |
| DF | 0.539 | [0.46, 0.62] | 0.057 | 0.1535 | -0.2340 | 80 | +3 |
| SA | -0.048 | [-0.14, 0.04] | 0.416 | 0.3857 | -0.0420 | 87 | +10 |

Verdict: no variant reaches 0.8; the gap is representational (§D.7 final branch) (H15: not supported)

## 5. Random split: baselines, seeds, capacity, distillation, controls

| arm | MRR@10 all | seen-only | rho (Q_H) | rho (lex) | gap to oracle | signed gap_r | abs gap_r | C-alias MRR | nnz d/q | n(Q_H) |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 (frozen, no dropout) | 0.2063 | 0.1871 | 0.774 [0.69, 0.87] | 0.748 | -0.019 | -0.032 | 0.061 | 0.1636 ✓ | 72/23 | 3504 |
| V1, seed 2 | 0.2069 | 0.1864 | 0.801 [0.71, 0.90] | 0.769 | -0.020 | -0.043 | 0.063 | 0.1637 ✓ | 72/22 | 3504 |
| V1 + entry-wise VD | 0.2043 | 0.1894 | 0.658 [0.56, 0.76] | 0.614 | -0.014 | -0.026 | 0.037 | 0.1669 ✓ | 109/35 | 3504 |
| V1 + entry-wise VD, seed 2 | 0.2040 | 0.1895 | 0.660 [0.56, 0.76] | 0.618 | -0.012 | -0.039 | 0.052 | 0.1662 ✓ | 109/35 | 3504 |
| V1 + cluster-wise VD | 0.2090 | 0.1932 | 0.669 [0.57, 0.77] | 0.664 | -0.036 | -0.040 | 0.046 | 0.1732 ✓ | 121/39 | 3504 |
| V1 + cluster VD + meta calibration | 0.2048 | 0.1918 | 0.598 [0.50, 0.70] | 0.597 | -0.025 | -0.039 | 0.041 | 0.1743 ✓ | 146/47 | 3504 |
| V1, linear head | 0.1961 | 0.1823 | 0.598 [0.50, 0.70] | 0.591 | 0.023 | -0.023 | 0.031 | 0.1579 ✓ | 82/23 | 3504 |
| V1 + dense-teacher distillation | 0.2169 | 0.1970 | 0.651 [0.58, 0.72] | 0.695 | -0.001 | -0.003 | 0.067 | 0.1717 ✓ | 60/19 | 2828 |
| V1, bare-string entries | 0.2434 | 0.2322 | 0.599 [0.50, 0.70] | 0.554 | -0.034 | 0.063 | 0.063 | 0.2123 ✓ | 198/87 | 3176 |
| V1, entries whitened with the token-side transform | 0.2497 | 0.2342 | 0.592 [0.51, 0.68] | 0.615 | -0.023 | 0.023 | 0.051 | 0.2215 ✓ | 116/36 | 2944 |
| V1 + tail normalisation | 0.1891 | 0.1705 | 0.785 [0.70, 0.87] | 0.763 | -0.021 | -0.053 | 0.055 | 0.1636 ✓ | 60/19 | 2958 |
| V1 + shared entry-side map | 0.2780 | 0.2740 | 0.376 [0.24, 0.52] | 0.382 | -0.027 | 0.005 | 0.056 | 0.2680 ✓ | 564/150 | 3118 |
| V1, trained entry rows (parameterized-vocabulary control) | 0.2750 | 0.2721 | 0.158 [0.07, 0.24] | 0.266 | -0.002 | -0.156 | 0.156 | 0.2656 ✓ | 251/93 | 3073 |
| C-rand (random vocabulary) | 0.0000 | 0.0000 | 0.000 [0.00, 0.00] | 0.000 | 1.000 | 0.000 | 0.000 | 0.0000 ✗ | 0/0 | 3504 |

Effectiveness ceiling (all entries, C1) — the fork between a pure text-defined vocabulary and a hybrid extension of a fixed LSR model turns on this table:

| model | MRR@10 | R@100 |
|---|---|---|
| V1 oracle (frozen encoder, all entries) | 0.2046 | 0.7609 |
| V1 oracle + distillation from the dense score | 0.2176 | 0.7723 |
| V3 oracle (full fine-tuning, all entries) | 0.2539 | 0.8330 |
| BM25 | 0.1889 | 0.6759 |
| dense e5 (the teacher) | 0.3564 | 0.9223 |
| SPLADE++ | 0.3817 | 0.9100 |

## 6. Rare split

| arm | MRR@10 all | seen-only | rho (Q_H) | rho (lex) | gap to oracle | signed gap_r | abs gap_r | C-alias MRR | nnz d/q | n(Q_H) |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 (frozen, no dropout) | 0.2072 | 0.2020 | 0.942 [0.83, 1.06] | 0.935 | -0.008 | n/a | n/a | 0.1198 ✓ | 60/19 | 351 |
| V1 + entry-wise VD | 0.2066 | 0.2014 | 0.817 [0.65, 1.01] | 0.757 | -0.034 | n/a | n/a | 0.1331 ✓ | 90/29 | 351 |
| V1 + tail normalisation | 0.1864 | 0.1816 | 0.852 [0.75, 0.95] | 0.879 | 0.032 | n/a | n/a | 0.0909 ✓ | 48/15 | 511 |

## 7. Phrase / entity insertion

2,000 Title-case bigrams with >= 100 passages in P (of 14,593 qualifying; collection frequency 488–217,285). Examples: united states, new york, best answer, los angeles, north america, social security, world war, york city, north carolina, new jersey, report abuse, united kingdom, san francisco, getty images, civil war.

| arm | MRR@10 all | seen-only | rho (Q_H) | rho (lex) | gap to oracle | signed gap_r | abs gap_r | C-alias MRR | nnz d/q | n(Q_H) |
|---|---|---|---|---|---|---|---|---|---|---|
| V1oracle + 2,000 inserted phrases | 0.1880 | 0.2046 | -0.783 [-1.45, -0.35] | -0.443 | 0.184 | 1.702 | 1.702 | 0.1864 ✓ | 70/24 | 887 |
| V1oracle + 2,000 phrases + tail normalisation | 0.1647 | 0.1663 | 0.012 [-0.30, 0.26] | 0.001 | 0.230 | 0.689 | 0.689 | 0.1872 ✓ | 40/13 | 887 |

| model | MRR@10 all entries |
|---|---|
| word-only oracle (V1oracle, 30k words) | 0.2046 |
| V1oracle + phrases inserted, no retraining | 0.1880 |
| oracle trained with words + phrases | 0.2037 |

Read these three with the Q_H numbers above, not instead of them. Overall MRR@10 barely moves because the phrases matter to only ~13% of dev queries; on the Q_H subset where they do matter, the retrained oracle gains 0.042 (0.2645 vs 0.2222 with them zeroed). So the phrases *are* useful entries when trained in, and post-hoc insertion is what fails to capture that.

## 8. Encoder-training arms

| arm | MRR@10 all | seen-only | rho (Q_H) | rho (lex) | gap to oracle | signed gap_r | abs gap_r | C-alias MRR | nnz d/q | n(Q_H) |
|---|---|---|---|---|---|---|---|---|---|---|
| V2 (LoRA) | 0.0087 | 0.0090 | 0.002 [-0.02, 0.02] | 0.007 | 0.964 | -0.066 | 0.066 | 0.0065 ✗ | 1024/255 | 3504 |
| V2 (LoRA) + VD | 0.0239 | 0.0236 | 0.055 [0.02, 0.09] | 0.043 | 0.878 | -0.044 | 0.044 | 0.0212 ✓ | 1024/218 | 3504 |
| V3 (full fine-tuning) | 0.2502 | 0.2381 | 0.528 [0.44, 0.62] | 0.548 | 0.021 | -0.027 | 0.068 | 0.2244 ✓ | 151/38 | 2461 |
| V3 + VD | 0.2414 | 0.2318 | 0.406 [0.33, 0.49] | 0.422 | 0.048 | -0.036 | 0.036 | 0.2232 ✓ | 217/70 | 2461 |

## Pilot D decision rule on the rebuild

```json
{
  "passing_runs": [
    "V1_rare",
    "V1_s2",
    "V1norm_rare",
    "V1vd_rare"
  ],
  "H10_verdict": "V1 passes on ['rare', 's2', 'V1norm_rare', 'V1vd_rare'] and falls short elsewhere",
  "H11_V3_rho": 0.5282345394381672,
  "H11_verdict": "not supported",
  "H12_vd_effect": {
    "V1": [
      0.6578161039534046,
      0.773673268323695
    ],
    "V2": [
      0.05460429514919566,
      0.0018456180850875245
    ],
    "V3": [
      0.4056953891948081,
      0.5282345394381672
    ]
  },
  "H13_difficulty_order": {
    "random": 0.773673268323695,
    "cluster": 0.31316994950879634,
    "rare": 0.9422112657515885
  },
  "H14_crand_worse": true,
  "H14_verdict": "supported",
  "recipe": "frozen encoder + light head",
  "signed_gaps": {
    "Crand": 0.0,
    "V1": -0.031695803034897475,
    "V1_R1": 0.06251759152618197,
    "V1_Wh": 0.02271957509539546,
    "V1_cluster": 0.38565733080369424,
    "V1_cluster_EDF": 0.15353162092787692,
    "V1_cluster_ESA": 0.38565733080369424,
    "V1_cluster_EZ": 0.22928045635215497,
    "V1_cluster_Eraw": 0.38565733080369424,
    "V1_cluster_R1": 0.7692039523679297,
    "V1_cluster_Wh": 0.3066570556548003,
    "V1_cluster_s2": 0.37869579792140967,
    "V1_phrase": 1.7023189914778414,
    "V1_phrase_norm": 0.6890252096439154,
    "V1_rare": NaN,
    "V1_s2": -0.043030730538743414,
    "V1cvd": -0.03974000465069942,
    "V1cvd_cluster": 0.3939058607561561,
    "V1cvd_cluster_R1": 0.8556897795182842,
    "V1dist": -0.0028602866968609544,
    "V1eh": 0.004810294388518164,
    "V1eh_cluster": 1.469025416967095,
    "V1ehnorm_cluster": 1.2295907272308026,
    "V1k100_cluster": 0.39084986816510614,
    "V1lin": -0.022863313368725866,
    "V1lin_cluster": 0.46404314553144094,
    "V1meta": -0.03872093608285914,
    "V1meta_cluster": 0.3662231258863064,
    "V1norm": -0.05335541131998729,
    "V1normWh_cluster": 0.14110673185770845,
    "V1norm_cluster": 0.17896354924490093,
    "V1norm_cluster_R1": 0.2099433198473673,
    "V1norm_cluster_s2": 0.15833453855652502,
    "V1norm_rare": NaN,
    "V1normz_cluster": 0.33458852331326516,
    "V1param": -0.1561867366390466,
    "V1vd": -0.02626849591749955,
    "V1vd_cluster": 0.34544618938109917,
    "V1vd_rare": NaN,
    "V1vd_s2": -0.039481638204227905,
    "V2": -0.06598974179942851,
    "V2vd": -0.044311287627407686,
    "V3": -0.027074336215203262,
    "V3_cluster": 0.34501297727605157,
    "V3vd": -0.035566446993446396
  },
  "pilotE_triggered": false,
  "summary": "4 of 45 runs meet H10. No systematic signed gap; if no arm passes the problem is representational, not calibrational (\u00a7D.7 final branch).",
  "systematic_gap_runs": [
    "V1_cluster_R1",
    "V1_phrase",
    "V1_phrase_norm",
    "V1cvd_cluster_R1",
    "V1eh_cluster",
    "V1ehnorm_cluster",
    "V1lin_cluster"
  ]
}
```
