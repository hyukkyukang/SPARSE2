# Pilot D — held-out vocabulary generalisation

| run | split | MRR@10 all | seen-only | rho (Q_H) | 95% CI | denom | valid |
|---|---|---|---|---|---|---|---|
| smoke_V1 | random | 0.1179 | 0.1098 | 1.000 | [1.00, 1.00] | 0.0209 | True |

| run | gap to oracle on Q_H | signed gap_r | abs gap_r | gap_w | Wasserstein | nnz(d) | nnz(q) | \|Q_H\| | \|Q_H-lex\| | C-alias MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke_V1 | 0.000 | -0.0311 | 0.0416 | 0.0028 | 0.00179 | 543 | 68 | 3009 | None | 0.0970 |

## Decision (§D.7)

```json
{
  "passing_runs": [
    "smoke_V1"
  ],
  "H10_verdict": "no arm passes",
  "H11_V3_rho": null,
  "H11_verdict": "not supported",
  "H12_vd_effect": {
    "V1": [
      null,
      null
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
    "random": null,
    "cluster": null,
    "rare": null
  },
  "H14_crand_worse": false,
  "H14_verdict": "NOT supported \u2014 stop and reconsider",
  "recipe": "no arm passes",
  "signed_gaps": {
    "smoke_V1": -0.031111365293229637
  },
  "pilotE_triggered": false,
  "summary": "1 of 1 runs meet H10. No systematic signed gap; if no arm passes the problem is representational, not calibrational (\u00a7D.7 final branch).",
  "systematic_gap_runs": []
}
```