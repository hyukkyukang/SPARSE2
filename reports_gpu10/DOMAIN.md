# Domain-shift results

A model trained only on MS MARCO, given entries for terminology a *different*
corpus uses and it has never had a dimension for, evaluated on that corpus.
Entries are chosen from corpus text by frequency alone — never from queries or
relevance labels — which is what a deployment indexing a corpus can do.

`value (Δ)` is the metric with the entries inserted, and its change against that
backbone's baseline row; `*` marks a paired bootstrap CI over queries excluding
zero. Italic rows are reference systems, not variants of our model.

## scifact — 5,183 passages, 300 queries, 1.1 relevant per query

| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |
|---|---|---|---|---|---|---|
| _BM25 (k1=0.82, b=0.68)_ | — | — | _0.6290_ | _0.6617_ | _0.8852_ | — |
| _SPLADE++ CoCondenser-EnsembleDistil_ | — | — | _0.6484_ | _0.6786_ | _0.9203_ | 175 |
| _SPLADE-v3_ | — | — | _0.6574_ | _0.6858_ | _0.9293_ | 236 |
| _e5-base-v2 dense (backbone of the block below)_ | — | — | _0.6690_ | _0.7014_ | _0.9460_ | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.4119** | **0.4373** | **0.7700** | 73 |
| this corpus's terms, by document frequency | 2003 | — | 0.4635 (+0.0516*) | 0.4944 (+0.0570*) | 0.8320 (+0.0620*) | 769 |
| this corpus's terms, by tf-idf | 2003 | 0.056 | 0.4646 (+0.0528*) | 0.4959 (+0.0586*) | 0.8353 (+0.0653*) | 769 |
| this corpus's terms, df ceiling 10% | 2003 | 0.056 | 0.4635 (+0.0516*) | 0.4944 (+0.0570*) | 0.8320 (+0.0620*) | 769 |
| df terms + tail calibration | 2003 | — | 0.4625 (+0.0506*) | 0.4962 (+0.0589*) | 0.8228 (+0.0528*) | 770 |
| random vectors (capacity control) | 2003 | — | 0.4119 (+0.0000) | 0.4373 (+0.0000) | 0.7700 (+0.0000) | 73 |
| nfcorpus terms (domain control) | 957 | — | 0.4170 (+0.0051) | 0.4421 (+0.0048) | 0.7733 (+0.0033) | 272 |

## nfcorpus — 3,633 passages, 323 queries, 38.2 relevant per query

| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |
|---|---|---|---|---|---|---|
| _BM25 (k1=0.82, b=0.68)_ | — | — | _0.5086_ | _0.3033_ | _0.2381_ | — |
| _SPLADE++ CoCondenser-EnsembleDistil_ | — | — | _0.5611_ | _0.3415_ | _0.2822_ | 172 |
| _SPLADE-v3_ | — | — | _0.5817_ | _0.3511_ | _0.2949_ | 223 |
| _e5-base-v2 dense (backbone of the block below)_ | — | — | _0.5641_ | _0.3570_ | _0.3171_ | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.4899** | **0.2943** | **0.2705** | 77 |
| this corpus's terms, by document frequency | 957 | — | 0.5166 (+0.0267*) | 0.3216 (+0.0273*) | 0.2854 (+0.0149*) | 359 |
| this corpus's terms, by tf-idf | 957 | 0.099 | 0.5166 (+0.0267*) | 0.3216 (+0.0273*) | 0.2854 (+0.0149*) | 359 |
| this corpus's terms, df ceiling 10% | 957 | 0.099 | 0.5166 (+0.0267*) | 0.3216 (+0.0273*) | 0.2854 (+0.0149*) | 359 |
| df terms + tail calibration | 957 | — | 0.5039 (+0.0141) | 0.3144 (+0.0201*) | 0.2827 (+0.0122*) | 394 |
| random vectors (capacity control) | 957 | — | 0.4899 (+0.0000) | 0.2943 (+0.0000) | 0.2705 (+0.0000) | 77 |
| scifact terms (domain control) | 957 | — | 0.4902 (+0.0003) | 0.2986 (+0.0043*) | 0.2720 (+0.0015) | 348 |

## trec-covid — 171,332 passages, 50 queries, 493.5 relevant per query

| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |
|---|---|---|---|---|---|---|
| _BM25 (k1=0.82, b=0.68)_ | — | — | … | … | … | — |
| _SPLADE++ CoCondenser-EnsembleDistil_ | — | — | … | … | … | — |
| _SPLADE-v3_ | — | — | … | … | … | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.6942** | **0.5543** | **0.0892** | 61 |
| this corpus's terms, by document frequency | 2974 | — | 0.3719 (-0.3223*) | 0.1967 (-0.3576*) | 0.0270 (-0.0622*) | 736 |
| this corpus's terms, by tf-idf | 2976 | 0.338 | 0.4432 (-0.2509*) | 0.2322 (-0.3221*) | 0.0291 (-0.0601*) | 677 |
| this corpus's terms, df ceiling 10% | 2969 | 0.028 | 0.4166 (-0.2776*) | 0.2032 (-0.3511*) | 0.0273 (-0.0620*) | 735 |
| df terms + tail calibration | 2974 | — | 0.3295 (-0.3647*) | 0.1697 (-0.3846*) | 0.0237 (-0.0656*) | 694 |
| random vectors (capacity control) | 2974 | — | 0.6942 (+0.0000) | 0.5532 (-0.0011) | 0.0891 (-0.0002) | 61 |
| scifact terms (domain control) | … | — | … | … | … | … |

## Reading it

* **Italic rows are reference systems**: BM25, two SPLADE models, and each
  backbone's own dense retrieval, all on the same corpus, queries, qrels and
  metrics. A backbone's dense row is the ceiling a frozen-encoder sparse
  projection of that encoder is working against.
* **One baseline row per backbone** — one encode over the trained vocabulary,
  shared by every variant below it. Under the earlier bug it varied by selection rule.
* **rand ≈ 0 everywhere** would mean extra dimensions are inert without meaning, so
  any gain is not capacity.
* **ctl-\* ≈ 0** would mean the gain is not "any real vocabulary" but this corpus's.
* **df vs tfidf vs dfcap** isolates how entries are *chosen*. On corpora where fewer
  than `--n-add` candidates clear the occurrence floor, all three take every eligible
  term and the rows are identical by construction, not by finding.
* **nnz(d)** is the cost side: insertion multiplies non-zeros per document several
  fold, which is index size and query latency.
* **Backbones** are compared within a corpus only; each has its own layer, threshold
  and whitening (`notes/backbones.md`), chosen by the same rule.