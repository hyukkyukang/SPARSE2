"""Domain-shift deliverable: one table per corpus, a block per sparse backbone.

The baseline is its own encode over the trained vocabulary only, so it is a single row per
backbone that every variant is measured against. That is the point of the correction in
`scripts/92_domain_eval.py`: deriving the baseline by masking inserted entries out of the
extended encode made it depend on which entries were inserted, because on a dense corpus
the inserted entries evict base entries from the fixed-size per-document store.

Reference rows (italic) are other systems on the same corpus, queries, qrels and metrics:
BM25, SPLADE++ and SPLADE-v3 (encoder-independent), and each backbone's own dense
retrieval -- the ceiling a frozen-encoder sparse projection of that encoder works against.

Every variant row is the same model and the same corpus; only the appended entries differ:

  df      entries ranked by document frequency in the target corpus
  tfidf   ranked by total occurrences x log(N / document frequency)
  dfcap   ranked by document frequency, after dropping anything in >10% of passages
  norm    df entries, with the Pilot F tail correction applied at insertion
  rand    random unit vectors: same count, no linguistic content (capacity control)
  ctl-*   another corpus's vocabulary: same count and construction, wrong domain
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger

lg = get_logger("domreport", "93_domain_report.log")

CORPORA = ["scifact", "nfcorpus", "trec-covid"]
# (encoder key, checkpoint name, label) -- notes/backbones.md
BACKBONES = [("e5", "V1oracle", "e5-base-v2"),
             ("octen", "V1oracle_octen", "Octen-Embedding-0.6B"),
             ("jina5s", "V1oracle_jina5s", "jina-embeddings-v5-text-small")]
VARIANTS = [("df", "", "this corpus's terms, by document frequency"),
            ("tfidf", "_tfidf", "this corpus's terms, by tf-idf"),
            ("dfcap", "_dfcap", "this corpus's terms, df ceiling 10%"),
            ("norm", "_norm", "df terms + tail calibration"),
            ("rand", "_ctl-rand", "random vectors (capacity control)"),
            ("ctl", None, "another corpus's terms (domain control)")]
REFS = [("bm25", "BM25 (k1=0.82, b=0.68)"),
        ("splade_pp", "SPLADE++ CoCondenser-EnsembleDistil"),
        ("splade_v3", "SPLADE-v3")]


def load(corpus, model, sfx):
    p = paths.RESULTS / f"92_domain_{corpus}_{model}{sfx}.json"
    return json.load(open(p)) if p.exists() else None


def cell(d, key, field):
    if d is None:
        return "…"
    delta = d[f"delta_{key}"]
    star = "*" if delta["excludes_zero"] else ""
    return f"{d['with'][field]:.4f} ({delta['diff']:+.4f}{star})"


def ref_row(lab, r, nnz="—"):
    return (f"| _{lab}_ | — | — | _{r['mrr']:.4f}_ | _{r['ndcg']:.4f}_ | _{r['r100']:.4f}_ | "
            f"{nnz} |")


def main(a):
    md = ["# Domain-shift results", "",
          "A model trained only on MS MARCO, given entries for terminology a *different*",
          "corpus uses and it has never had a dimension for, evaluated on that corpus.",
          "Entries are chosen from corpus text by frequency alone — never from queries or",
          "relevance labels — which is what a deployment indexing a corpus can do.", "",
          "`value (Δ)` is the metric with the entries inserted, and its change against that",
          "backbone's baseline row; `*` marks a paired bootstrap CI over queries excluding",
          "zero. Italic rows are reference systems, not variants of our model.", ""]
    filled = total = 0
    for c in CORPORA:
        meta = paths.RESULTS / f"90_domain_{c}.json"
        if not meta.exists():
            continue
        m = json.load(open(meta))
        other = "nfcorpus" if c == "scifact" else "scifact"
        md += [f"## {c} — {m['n_passages']:,} passages, {m['n_queries']} queries, "
               f"{m['mean_rel_per_query']:.1f} relevant per query", "",
               "| system / inserted entries | n | max doc share | MRR@10 | nDCG@10 | R@100 | nnz(d) |",
               "|---|---|---|---|---|---|---|"]
        bp = paths.RESULTS / f"94_domain_baselines_{c}.json"
        ref = json.load(open(bp)) if bp.exists() else {}
        for key, lab in REFS:
            if key in ref:
                md.append(ref_row(lab, ref[key], f"{ref[key]['nnz_d']:.0f}" if "nnz_d" in ref[key] else "—"))
            else:
                md.append(f"| _{lab}_ | — | — | … | … | … | — |")
        for enc, model, lab in BACKBONES:
            r = ref.get(f"dense_{enc}") or (ref.get("dense") if enc == "e5" else None)
            if r is not None:
                md.append(ref_row(f"{lab} dense (backbone of the block below)", r))
        for enc, model, lab in BACKBONES:
            rows = {name: load(c, model, sfx if sfx is not None else f"_ctl-{other}")
                    for name, sfx, _ in VARIANTS}
            if not any(rows.values()) and not (paths.CKPT / model / "head.pt").exists():
                continue                      # backbone not run on this corpus (yet)
            base = next((d for d in rows.values() if d), None)
            if base is not None:
                b = base["without"]
                md.append(f"| **{lab} — sparse, trained vocabulary only** | 0 | — | "
                          f"**{b['mrr']:.4f}** | **{b['ndcg']:.4f}** | **{b['r100']:.4f}** | "
                          f"{b['nnz_d']:.0f} |")
            else:
                md.append(f"| **{lab} — sparse, trained vocabulary only** | 0 | — | … | … | … | … |")
            for name, sfx, label in VARIANTS:
                d = rows[name]
                total += 1
                filled += d is not None
                vp = paths.RESULTS / (f"91_domain_vocab_{c}"
                                      f"{'' if name in ('df','norm','rand','ctl') else '_' + name}.json")
                share = json.load(open(vp)).get("doc_share_max") if vp.exists() else None
                lab2 = label.replace("another corpus's", f"{other}") if name == "ctl" else label
                if d:
                    md.append(f"| {lab2} | {d['n_inserted']} | "
                              f"{('%.3f' % share) if share is not None and name in ('tfidf','dfcap') else '—'} | "
                              f"{cell(d,'mrr','mrr')} | {cell(d,'ndcg','ndcg')} | "
                              f"{cell(d,'r100','r100')} | {d['with']['nnz_d']:.0f} |")
                else:
                    md.append(f"| {lab2} | … | — | … | … | … | … |")
        md.append("")
    md += ["## Reading it", "",
           "* **Italic rows are reference systems**: BM25, two SPLADE models, and each",
           "  backbone's own dense retrieval, all on the same corpus, queries, qrels and",
           "  metrics. A backbone's dense row is the ceiling a frozen-encoder sparse",
           "  projection of that encoder is working against.",
           "* **One baseline row per backbone** — one encode over the trained vocabulary,",
           "  shared by every variant below it. Under the earlier bug it varied by selection rule.",
           "* **rand ≈ 0 everywhere** would mean extra dimensions are inert without meaning, so",
           "  any gain is not capacity.",
           "* **ctl-\\* ≈ 0** would mean the gain is not \"any real vocabulary\" but this corpus's.",
           "* **df vs tfidf vs dfcap** isolates how entries are *chosen*. On corpora where fewer",
           "  than `--n-add` candidates clear the occurrence floor, all three take every eligible",
           "  term and the rows are identical by construction, not by finding.",
           "* **nnz(d)** is the cost side: insertion multiplies non-zeros per document several",
           "  fold, which is index size and query latency.",
           "* **Backbones** are compared within a corpus only; each has its own layer, threshold",
           "  and whitening (`notes/backbones.md`), chosen by the same rule."]
    out = paths.REPORTS / "DOMAIN.md"
    out.write_text("\n".join(md))
    lg.info(f"wrote {out}  ({filled}/{total} evaluation cells filled)")
    if a.show:
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    main(ap.parse_args())
