# Pilot B — bare string vs contextual prototype

## e5_L9

| rep | hub skew | hub share | Gini | Spearman df~freq | runaway % | dead % | nnz/doc | in-context hit@10 | self-act rank | P(rank1) | sense |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 3.6 | 0.044 | 0.259 | 0.249 | 0.00 | 0.00 | 121 | 0.923 | 1 | 0.939 | 0.717 |
| R1-shared | 12.9 | 0.054 | 0.338 | 0.090 | 0.00 | 17.87 | 1 | 0.837 | 1 | 0.900 | 0.698 |
| R2 | 17.2 | 0.161 | 0.615 | 0.566 | 0.02 | 0.04 | 121 | 0.998 | 1 | 0.958 | 0.774 |
| R3 | 15.8 | 0.185 | 0.643 | 0.614 | 0.09 | 0.02 | 120 | 0.999 | 1 | 0.977 | 0.793 |

### 20 largest hubs

* **R1**: `you`(720), `it`(595), `american`(541), `has`(526), `merican`(498), `can`(486), `one`(481), `we`(477), `hasn`(472), `couldn`(472), `minutes`(471), `hour`(466), `us`(464), `youre`(463), `long`(454), `every`(454), `ourselves`(445), `cause`(443), `could`(438), `americans`(435)
* **R1-shared**: `flattering`(2275), `positive`(2262), `negative`(2096), `inconclusive`(1819), `satisfied`(1791), `dissatisfied`(1644), `decrease`(1582), `no`(1561), `maybe`(1468), `increase`(1272), `sports`(1265), `duplicates`(1189), `athlete`(1068), `film`(1028), `business`(983), `building`(927), `neither`(921), `never`(904), `false`(867), `always`(814)
* **R2**: `that`(6474), `and`(6179), `but`(6136), `which`(5999), `or`(5670), `thats`(4769), `though`(4653), `the`(4308), `anyways`(3531), `is`(2820), `cuz`(2608), `an`(2527), `your`(2386), `you`(2385), `nonspecific`(2362), `nor`(2311), `youre`(2305), `a`(2264), `are`(2185), `yours`(1887)
* **R3**: `and`(6818), `that`(5932), `or`(5823), `the`(5754), `for`(5592), `which`(5421), `but`(5138), `an`(4243), `a`(4215), `who`(4192), `with`(3879), `nor`(3521), `in`(3401), `to`(3330), `though`(3264), `is`(3070), `que`(2991), `while`(2693), `your`(2680), `you`(2520)

### Stability curve (R2, disjoint occurrence sets)

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Jaccard@50 | 0.316 | 0.451 | 0.515 | 0.598 | 0.674 | 0.759 |
| whitened cos | 0.602 | 0.772 | 0.833 | 0.896 | 0.938 | 0.970 |

**k\*** (smallest k reaching 0.8 x the k=50 Jaccard) = 20

### Decision (§B.4)

```json
{
  "chosen_rep": "R2",
  "chosen_rep_by_skew": "R3",
  "eligible": [
    "R2",
    "R3"
  ],
  "unusable": [],
  "hub_share": {
    "R1": 0.04434850066900253,
    "R2": 0.16079850494861603,
    "R3": 0.18497949838638306
  },
  "hub_skew": {
    "R1": 3.614572048187256,
    "R2": 17.154434204101562,
    "R3": 15.827569007873535
  },
  "k_star": 20,
  "H5": {
    "hub_skew_ratio": 0.21070773918751717,
    "spearman_R1": 0.2489120663116368,
    "spearman_R2": 0.5660657177388423,
    "in_context_R1": 0.9230877192982456,
    "in_context_R2": 0.9977263157894737
  },
  "H7": {
    "own": 0.9230877192982456,
    "shared": 0.8366315789473684
  },
  "runaway_max": 0.0009333333333333333,
  "H5_verdict": "reversed: R1 is markedly LESS hubby than R2",
  "H6_k10_ratio": 0.7884511631186724,
  "H6_verdict": "narrowly not supported",
  "H7_verdict": "supported",
  "pilotE_mandatory": false,
  "note": "The rule ranks by hubness among representations not worse on in-context self-hit. R1 is excluded because its in-context self-hit is ~7 points below R2/R3 -- but note that self-hit against R2/R3 is structurally favoured (a prototype is an average of token states of the same word), so Pilot C evaluates R1, R2 and R3 end to end and lets retrieval arbitrate."
}
```

## e5_L10

| rep | hub skew | hub share | Gini | Spearman df~freq | runaway % | dead % | nnz/doc | in-context hit@10 | self-act rank | P(rank1) | sense |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 17.0 | 0.067 | 0.325 | 0.241 | 0.00 | 0.00 | 120 | 0.923 | 1 | 0.964 | 0.740 |
| R1-shared | 24.2 | 0.075 | 0.367 | 0.157 | 0.00 | 96.15 | 1 | 0.811 | 1 | 0.950 | 0.758 |
| R2 | 16.4 | 0.166 | 0.623 | 0.651 | 0.26 | 0.09 | 120 | 0.996 | 1 | 0.948 | 0.797 |
| R3 | 16.1 | 0.178 | 0.638 | 0.160 | 0.01 | 65.77 | 12 | 0.998 | 1 | 0.972 | 0.808 |

### 20 largest hubs

* **R1**: `natalie`(3554), `criminal`(2542), `networking`(2493), `geological`(2408), `cybersecurity`(2043), `ipl`(1973), `phishing`(1862), `guaranteed`(1681), `entente`(1539), `skater`(1438), `cuff`(1423), `endangered`(1270), `alcoholism`(1005), `gnu`(915), `thy`(902), `priests`(887), `health`(820), `dalton`(792), `correct`(779), `cambrian`(724)
* **R1-shared**: `false`(3983), `d`(3942), `satisfied`(3810), `incorrect`(3608), `dissatisfied`(3211), `bad`(3052), `b`(2948), `positive`(2879), `uh`(2859), `yes`(2831), `c`(2806), `flattering`(2685), `no`(2594), `negative`(2531), `maybe`(2524), `increase`(2450), `good`(2361), `inconclusive`(2276), `true`(1977), `decrease`(1810)
* **R2**: `pok`(5126), `that`(5011), `and`(4947), `saut`(4926), `fianc`(4792), `espa`(4768), `or`(4689), `beyonc`(4686), `which`(4678), `clich`(4614), `andr`(4539), `jalape`(4529), `though`(4502), `but`(4352), `cuz`(4232), `nestl`(4211), `thats`(4099), `speci`(3235), `nor`(3022), `is`(2312)
* **R3**: `here`(6602), `s`(5926), `aren`(5750), `para`(5685), `se`(5684), `bis`(5262), `cant`(5183), `m`(4752), `e`(4629), `ni`(3489), `the`(3300), `t`(3150), `a`(3034), `qui`(2849), `we`(2714), `in`(2492), `you`(2445), `is`(2414), `your`(2360), `and`(2298)

### Stability curve (R2, disjoint occurrence sets)

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Jaccard@50 | 0.299 | 0.424 | 0.494 | 0.583 | 0.663 | 0.753 |
| whitened cos | 0.569 | 0.748 | 0.815 | 0.884 | 0.931 | 0.967 |

**k\*** (smallest k reaching 0.8 x the k=50 Jaccard) = 20

### Decision (§B.4)

```json
{
  "chosen_rep": "R2",
  "chosen_rep_by_skew": "R2",
  "eligible": [
    "R2"
  ],
  "unusable": [
    "R3"
  ],
  "hub_share": {
    "R1": 0.06672199815511703,
    "R2": 0.16585299372673035,
    "R3": 0.17823849618434906
  },
  "hub_skew": {
    "R1": 16.95754051208496,
    "R2": 16.38713836669922,
    "R3": 16.064083099365234
  },
  "k_star": 20,
  "H5": {
    "hub_skew_ratio": 1.0348079165881015,
    "spearman_R1": 0.24055703535294848,
    "spearman_R2": 0.6513278006305605,
    "in_context_R1": 0.9226947368421052,
    "in_context_R2": 0.9959298245614036
  },
  "H7": {
    "own": 0.9226947368421052,
    "shared": 0.8108912280701754
  },
  "runaway_max": 0.0025666666666666667,
  "H5_verdict": "supported",
  "H6_k10_ratio": 0.7744312613505638,
  "H6_verdict": "narrowly not supported",
  "H7_verdict": "supported",
  "pilotE_mandatory": false,
  "note": "The rule ranks by hubness among representations not worse on in-context self-hit. R1 is excluded because its in-context self-hit is ~7 points below R2/R3 -- but note that self-hit against R2/R3 is structurally favoured (a prototype is an average of token states of the same word), so Pilot C evaluates R1, R2 and R3 end to end and lets retrieval arbitrate."
}
```
