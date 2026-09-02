# Pilot D — held-out vocabulary generalisation

rho compares a model to an oracle of its **own architecture** (§D.3). Rows
marked (arch\*) have no same-architecture oracle on their split — the
protocol's own run list provides only a V1 oracle for the cluster and rare
splits — so their rho is reported for completeness but is not a like-for-like
recovery ratio.

| run | split | oracle | MRR@10 all | seen-only | rho (Q_H) | 95% CI | denom | valid |
|---|---|---|---|---|---|---|---|---|
| Crand | random | V1oracle | 0.0000 | 0.0000 | 0.000 | [0.00, 0.00] | 0.0609 | True |
| V1 | random | V1oracle | 0.2010 | 0.1850 | 0.685 | [0.60, 0.77] | 0.0609 | True |
| V1_cluster | cluster | V1oracle_cluster | 0.1944 | 0.1765 | 0.567 | [0.49, 0.65] | 0.1407 | True |
| V1_rare | rare | V1oracle | 0.2017 | 0.1961 | 0.932 | [0.80, 1.07] | 0.0946 | True |

| run | gap to oracle on Q_H | signed gap_r | abs gap_r | gap_w | Wasserstein | nnz(d) | nnz(q) | \|Q_H\| | \|Q_H-lex\| | C-alias MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| Crand | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0 | 0 | 3346 | None | — |
| V1 | -0.012 | -0.0053 | 0.0668 | 0.0052 | 0.00032 | 72 | 22 | 3346 | None | 0.1595 |
| V1_cluster | 0.092 | 0.4727 | 0.4727 | -0.0458 | 0.00142 | 74 | 24 | 1675 | None | 0.1129 |
| V1_rare | -0.007 | nan | nan | nan | nan | 62 | 19 | 352 | None | — |

## Decision (§D.7)

```json
{
  "passing_runs": [
    "V1_rare"
  ],
  "H10_verdict": "V1 passes on ['rare'] and falls short elsewhere",
  "H11_V3_rho": null,
  "H11_verdict": "not supported",
  "H12_vd_effect": {
    "V1": [
      null,
      0.6849068734413967
    ],
    "V2": [
      null,
      null
    ],
    "V3": [
      null,
      null
    ]
  },
  "H13_difficulty_order": {
    "random": 0.6849068734413967,
    "cluster": 0.5666578016273517,
    "rare": 0.9322161080540272
  },
  "H14_crand_worse": true,
  "H14_verdict": "supported",
  "recipe": "frozen encoder + light head",
  "signed_gaps": {
    "Crand": 0.0,
    "V1": -0.005347633119368454,
    "V1_cluster": 0.4727370443514086,
    "V1_rare": NaN
  },
  "pilotE_triggered": false,
  "summary": "1 of 4 runs meet H10. No systematic signed gap; if no arm passes the problem is representational, not calibrational (\u00a7D.7 final branch).",
  "systematic_gap_runs": [
    "V1_cluster"
  ]
}
```