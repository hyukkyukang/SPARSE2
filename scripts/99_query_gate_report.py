"""Query-time gate deliverable: one table over all nine (backbone x corpus) cells.

Constraint of the study: nothing about the test queries is known ahead of time. A rule may
look at the single query it is answering, never at statistics over a set of queries. The
query-sample filters of `scripts/97_qf_filter.py` break that (they are an upper bound only);
everything in this report obeys it.

The gate counts how many inserted entries fire on the query being answered. If more than K
do, it acts, in one of four ways (`scripts/92_domain_eval.py --qgate K --qgate-mode ...`):

  all       zero every inserted entry on the query side
  topk      keep the K strongest inserted entries, zero the rest
  dfdrop    zero the inserted entries that fire on more than 5% of the indexed documents
            (a statistic of the index, computed with no query)
  fallback  the index keeps trained and inserted postings as two stores; answer the query
            from the trained store alone

The first three act on one merged store, so a document whose inserted entries filled its
per-document cap has already lost trained entries that the query side cannot bring back.
`fallback` has no such loss, which is why it is the mode this report leads with. The cap
table quantifies the eviction per cell.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger

lg = get_logger("qgreport", "99_query_gate_report.log")

CORPORA = ["nfcorpus", "scifact", "trec-covid"]
BACKBONES = [("V1oracle", "e5"), ("V1oracle_octen", "octen"), ("V1oracle_jina5s", "jina")]
CELLS = [(c, m, f"{lab} / {c}") for c in CORPORA for m, lab in BACKBONES]
# main table: two posting stores (trained = baseline encode, inserted = inserted part of
# the extended encode), so inserted entries never evict trained ones
TWO_STORE = [("", "one store, insert all, no gate (the domain table's number)"),
             ("_2s", "two stores, insert all, no gate"),
             ("_2s_qg20all", "two stores, gate K=20: answer from the trained store"),
             ("_2s_qg50all", "two stores, gate K=50: answer from the trained store"),
             ("_2s_qg20dfdrop", "two stores, gate K=20: drop inserted entries on >5% of documents"),
             ("_2s_qg50dfdrop", "two stores, gate K=50: drop inserted entries on >5% of documents"),
             ("_2s_qg20topk", "two stores, gate K=20: keep the 20 strongest inserted"),
             ("_2s_qg50topk", "two stores, gate K=50: keep the 50 strongest inserted")]
# earlier runs on one merged store (subject to cap eviction), and the first two-store form,
# which substituted the baseline ranking on gated queries only
ONE_STORE = [("_qg10all", "one store, drop all inserted, K=10"),
             ("_qg20all", "one store, drop all inserted, K=20"),
             ("_qg30all", "one store, drop all inserted, K=30"),
             ("_qg20topk", "one store, keep the 20 strongest, K=20"),
             ("_qg20dfdrop", "one store, drop those on >5% of documents, K=20"),
             ("_qg10fallback", "gated queries answered from the trained store, K=10"),
             ("_qg20fallback", "gated queries answered from the trained store, K=20"),
             ("_qg30fallback", "gated queries answered from the trained store, K=30"),
             ("_qg50fallback", "gated queries answered from the trained store, K=50")]


def load(c, m, sfx):
    p = paths.RESULTS / f"92_domain_{c}_{m}{sfx}.json"
    return json.load(open(p)) if p.exists() else None


def cell(d):
    if d is None:
        return "…"
    x = d["delta_mrr"]
    s = f"{x['diff']:+.4f}{'*' if x['excludes_zero'] else ''}"
    g = d.get("qgate")
    return s + (f" ({g['queries_gated']}q)" if g else "")


def cap_table():
    """How many trained entries the inserted ones evict from the per-document store."""
    rows = []
    for c in CORPORA:
        d = paths.DATA / "domain" / c
        for m, lab in BACKBONES:
            ext = sorted(glob.glob(str(d / f"ext_{m}_df_cap*_*.npz")))
            base = sorted(glob.glob(str(d / f"base_{m}_cap*_*.npz")))
            if not ext or not base:
                rows.append(f"| {lab} / {c} | … | … | … | … | … |")
                continue
            z, bz = np.load(ext[-1]), np.load(base[-1])
            di, dv, bv = z["di"], z["dv"], bz["dv"]
            nB0 = int(bz["di"].max()) + 1 if "nB0" not in z else int(z["nB0"])
            nnz, bnnz, cap = (dv > 0).sum(1), (bv > 0).sum(1), dv.shape[1]
            kept = ((dv > 0) & (di < nB0)).sum(1)
            lost = np.clip(bnnz - kept, 0, None)
            rows.append(f"| {lab} / {c} | {(nnz >= cap).mean() * 100:.1f}% | {nnz.mean():.0f} | "
                        f"{bnnz.mean():.1f} | {lost.mean():.1f} | {(lost > 0).mean() * 100:.1f}% |")
    return rows


def main(a):
    md = ["# Query-time gate", "",
          "Every rule here looks only at the query being answered (its count of firing inserted",
          "entries) and at the index. None uses a set of queries. `Δ` is the change in MRR@10",
          "against the backbone's trained-vocabulary baseline; `*` marks a paired bootstrap CI",
          "excluding zero; `(Nq)` is the number of queries on which the gate acted.", "",
          "## Two posting stores, per-query gate: all nine cells", "",
          "The index keeps trained and inserted postings separately, so inserting never evicts a",
          "trained entry (`--two-store`). The gate counts the inserted entries firing on the query;",
          "above K it acts on the inserted side only. `all` answers from the trained store alone.", "",
          "| variant | " + " | ".join(l for _, _, l in CELLS) + " |",
          "|---|" + "---|" * len(CELLS)]
    filled = total = 0

    def rows(variants):
        nonlocal filled, total
        out = []
        for sfx, lab in variants:
            cells = []
            for c, m, _ in CELLS:
                d = load(c, m, sfx)
                total += 1
                filled += d is not None
                cells.append(cell(d))
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
        return out
    md += rows(TWO_STORE)
    md += ["", "## One merged store, for comparison", "",
           "The same gates on one store with a 1,024-entry per-document cap. Where the inserted",
           "entries fill the cap (e5 on trec-covid and scifact, table below), trained entries are",
           "gone from the index and no query-side rule recovers them: drop-all leaves e5/trec-covid",
           "at -0.066 instead of 0.", "",
           "| variant | " + " | ".join(l for _, _, l in CELLS) + " |",
           "|---|" + "---|" * len(CELLS)]
    md += rows(ONE_STORE)
    md += ["", "## Inserted entries firing per query", "",
           "What the gate sees. The same 2,000–3,000 inserted entries fire on a handful of",
           "entries per query for the decoders and on hundreds for e5 on trec-covid.", "",
           "| cell | mean | median |", "|---|---|---|"]
    for c, m, lab in CELLS:
        d = load(c, m, "_qg20all") or load(c, m, "_qg20fallback")
        if d and d.get("qgate"):
            g = d["qgate"]
            md.append(f"| {lab} | {g['n_ins_mean']:.1f} | {g['n_ins_median']:.0f} |")
        else:
            md.append(f"| {lab} | … | … |")
    md += ["", "## Cap eviction: what a one-store gate cannot undo", "",
           "The evaluation store keeps at most 1,024 entries per document. When the inserted",
           "entries push a document past that, trained entries are dropped from the index and no",
           "query-side rule can restore them. `evicted` is the mean number of trained entries a",
           "document loses; a backbone with 0% at cap has no eviction and its one-store and",
           "two-store gates coincide.", "",
           "| cell | documents at cap | nnz(d) extended | nnz(d) trained | evicted / doc | documents losing any |",
           "|---|---|---|---|---|---|"] + cap_table()
    md += ["", "## Reading it", "",
           "* **Two stores + `dfdrop` at K=20** is the rule to report: both failures repaired",
           "  (e5/trec-covid −0.320 → +0.020, jina/scifact −0.009 → +0.045*), every gaining cell",
           "  within 0.018 of its ungated two-store number, octen/trec-covid up. It is the only",
           "  compliant rule that makes jina/scifact significant, because that failure is not a",
           "  firing explosion (median 11 inserted entries per query) but broad entries firing on",
           "  many documents, which the index statistic identifies.",
           "* **Two stores + `all`** is the guarantee: the trained store is untouched, so a gated",
           "  query can never score below the baseline (e5/trec-covid: exactly 0 at every K). Its",
           "  cost is bounded by the gated fraction: K=20 costs e5/scifact 0.022 and jina/trec-covid",
           "  0.020; K=50 costs at most 0.007 anywhere; K=10 costs the decoders 0.03–0.10 on",
           "  trec-covid.",
           "* **`topk` is fragile in K**: the best e5/scifact number (+0.057) at K=20, but K=50",
           "  leaves e5/trec-covid at −0.065 and jina/scifact at −0.005. Keeping more inserted",
           "  entries reintroduces the crowding. Not recommended.",
           "* **The gate's margin is large.** e5/trec-covid fires a median of 238 inserted entries",
           "  per query; the decoders' maxima on the same corpus are 33 and 48. Any K in [50, 200]",
           "  gates all of e5's covid queries and none of theirs, so K is not a tuned number.",
           "* **K=20 and the 5% document-firing threshold were fixed before any run.** K=10/30/50",
           "  were added afterwards for sensitivity, so K=50 is post hoc. octen's cells were never",
           "  used in choosing either value. The data contain one explosion-type failure.",
           "* **Price of the constraint**: the query-sample gate (`97_qf_filter.py --gate`, an upper",
           "  bound) reaches +0.081 on jina/scifact and +0.074 on e5/trec-covid; the compliant rule",
           "  reaches +0.045 and +0.020. On the other seven cells the two are within 0.018.",
           "* Cells where fewer than a few queries are gated are unchanged by construction, not",
           "  by finding (nfcorpus for every backbone)."]
    out = paths.REPORTS / "QUERY_GATE.md"
    out.write_text("\n".join(md) + "\n")
    lg.info(f"wrote {out}  ({filled}/{total} cells filled)")
    if a.show:
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    main(ap.parse_args())
