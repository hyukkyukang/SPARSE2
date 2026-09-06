# Pilot E — insertion-time calibration

| variant | rho | 95% CI | gap to oracle | signed gap_r | gap_w | nnz(d) | Δnnz(d) |
|---|---|---|---|---|---|---|---|
| raw | 0.313 | [0.23, 0.40] | 0.195 | 0.3857 | -0.0391 | 77 | +0 |
| Z | 0.494 | [0.42, 0.57] | 0.084 | 0.2293 | -0.0219 | 71 | -6 |
| DF | 0.539 | [0.46, 0.62] | 0.057 | 0.1535 | -0.2340 | 80 | +3 |
| SA | -0.048 | [-0.14, 0.04] | 0.416 | 0.3857 | -0.0420 | 87 | +10 |

A variant that 'fixes' rho by making documents denser has not fixed
anything — read Δnnz(d) beside every rho.

## Decision (§E.3)

```json
{
  "rows": {
    "raw": {
      "rho": 0.31316994950879634,
      "rho_lo": 0.22769848472027876,
      "rho_hi": 0.39668938120873143,
      "gap_to_oracle": 0.1952961218904411,
      "signed_gap_r": 0.38565733080369424,
      "gap_w": -0.03911571872474203,
      "nnz_d": 77.13795715213158,
      "delta_nnz_d": 0.0
    },
    "Z": {
      "rho": 0.4943736467631241,
      "rho_lo": 0.4226792913358352,
      "rho_hi": 0.5657652655242464,
      "gap_to_oracle": 0.08418884092139917,
      "signed_gap_r": 0.22928045635215497,
      "gap_w": -0.021927767749367073,
      "nnz_d": 71.39298853062108,
      "delta_nnz_d": -5.744968621510495
    },
    "DF": {
      "rho": 0.5388954178528715,
      "rho_lo": 0.46343558973941035,
      "rho_hi": 0.6155718718935659,
      "gap_to_oracle": 0.05693657375447557,
      "signed_gap_r": 0.15353162092787692,
      "gap_w": -0.2339937137563599,
      "nnz_d": 79.78886243958739,
      "delta_nnz_d": 2.6509052874558137
    },
    "SA": {
      "rho": -0.04755835683513371,
      "rho_lo": -0.13892238753894637,
      "rho_hi": 0.03873033471909014,
      "gap_to_oracle": 0.4161018970996047,
      "signed_gap_r": 0.38565733080369424,
      "gap_w": -0.041980345493879384,
      "nnz_d": 86.94553848373369,
      "delta_nnz_d": 9.807581331602108
    }
  },
  "chosen_variant": null,
  "H15_verdict": "not supported",
  "verdict": "no variant reaches 0.8; the gap is representational (\u00a7D.7 final branch)"
}
```