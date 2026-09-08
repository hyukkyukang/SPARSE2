# Query-time gate

Every rule here looks only at the query being answered (its count of firing inserted
entries) and at the index. None uses a set of queries. `Δ` is the change in MRR@10
against the backbone's trained-vocabulary baseline; `*` marks a paired bootstrap CI
excluding zero; `(Nq)` is the number of queries on which the gate acted.

## Two posting stores, per-query gate: all nine cells

The index keeps trained and inserted postings separately, so inserting never evicts a
trained entry (`--two-store`). The gate counts the inserted entries firing on the query;
above K it acts on the inserted side only. `all` answers from the trained store alone.

| variant | e5 / nfcorpus | octen / nfcorpus | jina / nfcorpus | e5 / scifact | octen / scifact | jina / scifact | e5 / trec-covid | octen / trec-covid | jina / trec-covid |
|---|---|---|---|---|---|---|---|---|---|
| one store, insert all, no gate (the domain table's number) | +0.0267* | +0.0297* | +0.0288* | +0.0511* | +0.0445* | -0.0092 | -0.3196* | +0.1863* | +0.2551* |
| two stores, insert all, no gate | +0.0267* | +0.0297* | +0.0288* | +0.0547* | +0.0445* | -0.0092 | -0.2730* | +0.1863* | +0.2551* |
| two stores, gate K=20: answer from the trained store | +0.0267* (1q) | +0.0297* (0q) | +0.0257* (3q) | +0.0331* (43q) | +0.0479* (23q) | +0.0143 (126q) | +0.0000 (49q) | +0.1963* (4q) | +0.2351* (6q) |
| two stores, gate K=50: answer from the trained store | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0481* (12q) | +0.0445* (0q) | +0.0142 (80q) | +0.0000 (49q) | +0.1863* (0q) | +0.2551* (0q) |
| two stores, gate K=20: drop inserted entries on >5% of documents | +0.0267* (1q) | +0.0297* (0q) | +0.0291* (3q) | +0.0368* (43q) | +0.0408* (23q) | +0.0446* (126q) | +0.0200 (49q) | +0.1963* (4q) | +0.2551* (6q) |
| two stores, gate K=50: drop inserted entries on >5% of documents | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0515* (12q) | +0.0445* (0q) | +0.0237 (80q) | +0.0200 (49q) | +0.1863* (0q) | +0.2551* (0q) |
| two stores, gate K=20: keep the 20 strongest inserted | +0.0267* (1q) | +0.0297* (0q) | +0.0288* (3q) | +0.0570* (43q) | +0.0455* (23q) | +0.0208 (126q) | +0.0352 (49q) | +0.1863* (4q) | +0.2551* (6q) |
| two stores, gate K=50: keep the 50 strongest inserted | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0560* (12q) | +0.0445* (0q) | -0.0052 (80q) | -0.0651 (49q) | +0.1863* (0q) | +0.2551* (0q) |

## One merged store, for comparison

The same gates on one store with a 1,024-entry per-document cap. Where the inserted
entries fill the cap (e5 on trec-covid and scifact, table below), trained entries are
gone from the index and no query-side rule recovers them: drop-all leaves e5/trec-covid
at -0.066 instead of 0.

| variant | e5 / nfcorpus | octen / nfcorpus | jina / nfcorpus | e5 / scifact | octen / scifact | jina / scifact | e5 / trec-covid | octen / trec-covid | jina / trec-covid |
|---|---|---|---|---|---|---|---|---|---|
| one store, drop all inserted, K=10 | +0.0267* (2q) | +0.0297* (1q) | +0.0300* (5q) | +0.0310* (67q) | +0.0204* (72q) | +0.0091 (151q) | -0.0663 (49q) | +0.1523* (16q) | +0.1580* (34q) |
| one store, drop all inserted, K=20 | +0.0267* (1q) | +0.0297* (0q) | +0.0257* (3q) | +0.0272* (43q) | +0.0479* (23q) | +0.0143 (126q) | -0.0663 (49q) | +0.1963* (4q) | +0.2351* (6q) |
| one store, drop all inserted, K=30 | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0402* (18q) | +0.0463* (7q) | +0.0168 (110q) | -0.0663 (49q) | +0.1863* (1q) | +0.2451* (1q) |
| one store, keep the 20 strongest, K=20 | +0.0267* (1q) | +0.0297* (0q) | +0.0288* (3q) | +0.0514* (43q) | +0.0455* (23q) | +0.0208 (126q) | +0.0105 (49q) | +0.1863* (4q) | +0.2551* (6q) |
| one store, drop those on >5% of documents, K=20 | +0.0267* (1q) | +0.0297* (0q) | +0.0291* (3q) | +0.0309* (43q) | +0.0408* (23q) | +0.0446* (126q) | -0.0575 (49q) | +0.1963* (4q) | +0.2551* (6q) |
| gated queries answered from the trained store, K=10 | +0.0267* (2q) | +0.0297* (1q) | +0.0300* (5q) | +0.0317* (67q) | +0.0204* (72q) | +0.0091 (151q) | +0.0000 (49q) | +0.1523* (16q) | +0.1580* (34q) |
| gated queries answered from the trained store, K=20 | +0.0267* (1q) | +0.0297* (0q) | +0.0257* (3q) | +0.0279* (43q) | +0.0479* (23q) | +0.0143 (126q) | +0.0000 (49q) | +0.1963* (4q) | +0.2351* (6q) |
| gated queries answered from the trained store, K=30 | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0403* (18q) | +0.0463* (7q) | +0.0168 (110q) | +0.0000 (49q) | +0.1863* (1q) | +0.2451* (1q) |
| gated queries answered from the trained store, K=50 | +0.0267* (0q) | +0.0297* (0q) | +0.0288* (0q) | +0.0428* (12q) | +0.0445* (0q) | +0.0142 (80q) | +0.0000 (49q) | +0.1863* (0q) | +0.2551* (0q) |

## Inserted entries firing per query

What the gate sees. The same 2,000–3,000 inserted entries fire on a handful of
entries per query for the decoders and on hundreds for e5 on trec-covid.

| cell | mean | median |
|---|---|---|
| e5 / nfcorpus | 0.5 | 0 |
| octen / nfcorpus | 0.5 | 0 |
| jina / nfcorpus | 0.7 | 0 |
| e5 / scifact | 8.8 | 2 |
| octen / scifact | 7.3 | 5 |
| jina / scifact | 31.8 | 11 |
| e5 / trec-covid | 209.5 | 238 |
| octen / trec-covid | 10.6 | 8 |
| jina / trec-covid | 14.4 | 14 |

## Cap eviction: what a one-store gate cannot undo

The evaluation store keeps at most 1,024 entries per document. When the inserted
entries push a document past that, trained entries are dropped from the index and no
query-side rule can restore them. `evicted` is the mean number of trained entries a
document loses; a backbone with 0% at cap has no eviction and its one-store and
two-store gates coincide.

| cell | documents at cap | nnz(d) extended | nnz(d) trained | evicted / doc | documents losing any |
|---|---|---|---|---|---|
| e5 / nfcorpus | 0.0% | 358 | 77.0 | 0.0 | 0.0% |
| octen / nfcorpus | 0.0% | 103 | 90.1 | 0.0 | 0.0% |
| jina / nfcorpus | 0.0% | 134 | 97.7 | 0.0 | 0.0% |
| e5 / scifact | 35.0% | 767 | 72.3 | 5.9 | 33.8% |
| octen / scifact | 0.0% | 123 | 87.3 | 0.0 | 0.0% |
| jina / scifact | 0.0% | 226 | 93.2 | 0.0 | 0.0% |
| e5 / trec-covid | 51.6% | 735 | 61.2 | 17.8 | 51.1% |
| octen / trec-covid | 0.0% | 105 | 69.9 | 0.0 | 0.0% |
| jina / trec-covid | 0.0% | 165 | 77.5 | 0.0 | 0.0% |

## Reading it

* **Two stores + `dfdrop` at K=20** is the rule to report: both failures repaired
  (e5/trec-covid −0.320 → +0.020, jina/scifact −0.009 → +0.045*), every gaining cell
  within 0.018 of its ungated two-store number, octen/trec-covid up. It is the only
  compliant rule that makes jina/scifact significant, because that failure is not a
  firing explosion (median 11 inserted entries per query) but broad entries firing on
  many documents, which the index statistic identifies.
* **Two stores + `all`** is the guarantee: the trained store is untouched, so a gated
  query can never score below the baseline (e5/trec-covid: exactly 0 at every K). Its
  cost is bounded by the gated fraction: K=20 costs e5/scifact 0.022 and jina/trec-covid
  0.020; K=50 costs at most 0.007 anywhere; K=10 costs the decoders 0.03–0.10 on
  trec-covid.
* **`topk` is fragile in K**: the best e5/scifact number (+0.057) at K=20, but K=50
  leaves e5/trec-covid at −0.065 and jina/scifact at −0.005. Keeping more inserted
  entries reintroduces the crowding. Not recommended.
* **The gate's margin is large.** e5/trec-covid fires a median of 238 inserted entries
  per query; the decoders' maxima on the same corpus are 33 and 48. Any K in [50, 200]
  gates all of e5's covid queries and none of theirs, so K is not a tuned number.
* **K=20 and the 5% document-firing threshold were fixed before any run.** K=10/30/50
  were added afterwards for sensitivity, so K=50 is post hoc. octen's cells were never
  used in choosing either value. The data contain one explosion-type failure.
* **Price of the constraint**: the query-sample gate (`97_qf_filter.py --gate`, an upper
  bound) reaches +0.081 on jina/scifact and +0.074 on e5/trec-covid; the compliant rule
  reaches +0.045 and +0.020. On the other seven cells the two are within 0.018.
* Cells where fewer than a few queries are gated are unchanged by construction, not
  by finding (nfcorpus for every backbone).
