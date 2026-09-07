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
| _Octen-Embedding-0.6B dense (backbone of the block below)_ | — | — | _0.6703_ | _0.7079_ | _0.9467_ | — |
| _jina-embeddings-v5-text-small dense (backbone of the block below)_ | — | — | _0.7077_ | _0.7440_ | _0.9483_ | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.4119** | **0.4373** | **0.7700** | 72 |
| this corpus's terms, by document frequency | 2003 | — | 0.4629 (+0.0511*) | 0.4939 (+0.0565*) | 0.8320 (+0.0620*) | 767 |
| this corpus's terms, by tf-idf | 2003 | 0.056 | 0.4624 (+0.0505*) | 0.4942 (+0.0569*) | 0.8353 (+0.0653*) | 767 |
| this corpus's terms, df ceiling 10% | 2003 | 0.056 | 0.4629 (+0.0511*) | 0.4939 (+0.0565*) | 0.8320 (+0.0620*) | 767 |
| df terms + tail calibration | 2003 | — | 0.4563 (+0.0444*) | 0.4911 (+0.0538*) | 0.8320 (+0.0620*) | 779 |
| random vectors (capacity control) | 2003 | — | 0.4119 (+0.0000) | 0.4373 (+0.0000) | 0.7700 (+0.0000) | 72 |
| nfcorpus terms (domain control) | 957 | — | 0.4173 (+0.0054) | 0.4431 (+0.0058) | 0.7700 (+0.0000) | 270 |
| **jina-embeddings-v5-text-small — sparse, trained vocabulary only** | 0 | — | **0.4179** | **0.4504** | **0.7661** | 93 |
| this corpus's terms, by document frequency | 2005 | — | 0.4086 (-0.0092) | 0.4373 (-0.0130) | 0.7576 (-0.0086) | 226 |
| this corpus's terms, by tf-idf | 2005 | 0.056 | 0.4067 (-0.0112) | 0.4358 (-0.0145) | 0.7576 (-0.0086) | 226 |
| this corpus's terms, df ceiling 10% | 2005 | 0.056 | 0.4086 (-0.0092) | 0.4373 (-0.0130) | 0.7576 (-0.0086) | 226 |
| df terms + tail calibration | 2005 | — | 0.3207 (-0.0972*) | 0.3522 (-0.0981*) | 0.6952 (-0.0709*) | 380 |
| random vectors (capacity control) | 2005 | — | 0.4179 (+0.0000) | 0.4504 (+0.0000) | 0.7661 (+0.0000) | 93 |
| nfcorpus terms (domain control) | 961 | — | 0.4377 (+0.0198) | 0.4680 (+0.0177) | 0.7856 (+0.0194) | 119 |

## nfcorpus — 3,633 passages, 323 queries, 38.2 relevant per query

| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |
|---|---|---|---|---|---|---|
| _BM25 (k1=0.82, b=0.68)_ | — | — | _0.5086_ | _0.3033_ | _0.2381_ | — |
| _SPLADE++ CoCondenser-EnsembleDistil_ | — | — | _0.5611_ | _0.3415_ | _0.2822_ | 172 |
| _SPLADE-v3_ | — | — | _0.5817_ | _0.3511_ | _0.2949_ | 223 |
| _e5-base-v2 dense (backbone of the block below)_ | — | — | _0.5641_ | _0.3570_ | _0.3171_ | — |
| _Octen-Embedding-0.6B dense (backbone of the block below)_ | — | — | _0.5781_ | _0.3652_ | _0.3353_ | — |
| _jina-embeddings-v5-text-small dense (backbone of the block below)_ | — | — | _0.6008_ | _0.3907_ | _0.3680_ | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.4899** | **0.2945** | **0.2688** | 77 |
| this corpus's terms, by document frequency | 957 | — | 0.5166 (+0.0267*) | 0.3218 (+0.0273*) | 0.2837 (+0.0149*) | 358 |
| this corpus's terms, by tf-idf | 957 | 0.099 | 0.5166 (+0.0267*) | 0.3218 (+0.0273*) | 0.2838 (+0.0149*) | 358 |
| this corpus's terms, df ceiling 10% | 957 | 0.099 | 0.5166 (+0.0267*) | 0.3218 (+0.0273*) | 0.2837 (+0.0149*) | 358 |
| df terms + tail calibration | 957 | — | 0.5106 (+0.0207*) | 0.3202 (+0.0257*) | 0.2879 (+0.0191*) | 415 |
| random vectors (capacity control) | 957 | — | 0.4899 (+0.0000) | 0.2945 (+0.0000) | 0.2688 (+0.0000) | 77 |
| scifact terms (domain control) | 957 | — | 0.4902 (+0.0003) | 0.2986 (+0.0041*) | 0.2703 (+0.0015) | 346 |
| **jina-embeddings-v5-text-small — sparse, trained vocabulary only** | 0 | — | **0.4960** | **0.3017** | **0.2807** | 98 |
| this corpus's terms, by document frequency | 961 | — | 0.5248 (+0.0288*) | 0.3232 (+0.0215*) | 0.2893 (+0.0085*) | 134 |
| this corpus's terms, by tf-idf | 961 | 0.099 | 0.5248 (+0.0288*) | 0.3232 (+0.0215*) | 0.2893 (+0.0085*) | 134 |
| this corpus's terms, df ceiling 10% | 961 | 0.099 | 0.5248 (+0.0288*) | 0.3232 (+0.0215*) | 0.2893 (+0.0085*) | 134 |
| df terms + tail calibration | 961 | — | 0.5239 (+0.0279*) | 0.3199 (+0.0182*) | 0.2914 (+0.0106*) | 183 |
| random vectors (capacity control) | 961 | — | 0.4960 (+0.0000) | 0.3017 (+0.0000) | 0.2807 (+0.0000) | 98 |
| scifact terms (domain control) | 961 | — | 0.5044 (+0.0084*) | 0.3110 (+0.0093*) | 0.2838 (+0.0031) | 120 |

## trec-covid — 171,332 passages, 50 queries, 493.5 relevant per query

| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |
|---|---|---|---|---|---|---|
| _BM25 (k1=0.82, b=0.68)_ | — | — | _0.7676_ | _0.6265_ | _0.0953_ | — |
| _SPLADE++ CoCondenser-EnsembleDistil_ | — | — | _0.8883_ | _0.8027_ | _0.1264_ | 153 |
| _SPLADE-v3_ | — | — | _0.9153_ | _0.8264_ | _0.1368_ | 199 |
| _e5-base-v2 dense (backbone of the block below)_ | — | — | _0.9133_ | _0.7899_ | _0.1297_ | — |
| _Octen-Embedding-0.6B dense (backbone of the block below)_ | — | — | _0.9300_ | _0.8300_ | _0.1605_ | — |
| _jina-embeddings-v5-text-small dense (backbone of the block below)_ | — | — | _0.8967_ | _0.8283_ | _0.1600_ | — |
| **e5-base-v2 — sparse, trained vocabulary only** | 0 | — | **0.6942** | **0.5532** | **0.0891** | 61 |
| this corpus's terms, by document frequency | 2974 | — | 0.3746 (-0.3196*) | 0.1984 (-0.3548*) | 0.0272 (-0.0619*) | 735 |
| this corpus's terms, by tf-idf | 2976 | 0.338 | 0.4560 (-0.2382*) | 0.2341 (-0.3191*) | 0.0289 (-0.0601*) | 676 |
| this corpus's terms, df ceiling 10% | 2969 | 0.028 | 0.4166 (-0.2776*) | 0.2032 (-0.3500*) | 0.0273 (-0.0618*) | 735 |
| df terms + tail calibration | 2974 | — | 0.3796 (-0.3146*) | 0.1889 (-0.3642*) | 0.0246 (-0.0645*) | 700 |
| random vectors (capacity control) | 2974 | — | 0.6942 (+0.0000) | 0.5532 (+0.0000) | 0.0891 (+0.0000) | 61 |
| scifact terms (domain control) | 2003 | — | 0.7142 (+0.0200) | 0.5639 (+0.0107) | 0.0881 (-0.0009) | 354 |
| **jina-embeddings-v5-text-small — sparse, trained vocabulary only** | 0 | — | **0.5965** | **0.4909** | **0.0905** | 78 |
| this corpus's terms, by document frequency | 2974 | — | 0.8517 (+0.2551*) | 0.6383 (+0.1474*) | 0.0964 (+0.0059) | 165 |
| this corpus's terms, by tf-idf | 2976 | 0.338 | 0.8367 (+0.2401*) | 0.6292 (+0.1383*) | 0.0957 (+0.0052) | 177 |
| this corpus's terms, df ceiling 10% | 2969 | 0.028 | 0.8133 (+0.2168*) | 0.6154 (+0.1244*) | 0.0990 (+0.0085) | 164 |
| df terms + tail calibration | 2974 | — | 0.8008 (+0.2043*) | 0.5879 (+0.0970) | 0.0850 (-0.0055) | 200 |
| random vectors (capacity control) | 2974 | — | 0.5965 (+0.0000) | 0.4909 (+0.0000) | 0.0905 (+0.0000) | 78 |
| scifact terms (domain control) | 2005 | — | 0.6124 (+0.0158) | 0.5076 (+0.0167) | 0.0915 (+0.0010) | 107 |

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