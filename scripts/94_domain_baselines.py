"""Reference systems on the domain corpora: BM25 and the dense encoder.

The domain table otherwise reports only our model with and without inserted entries, which
says whether insertion helps but not where any of it sits in absolute terms. These two rows
supply that. Both are evaluated on exactly the same corpora, queries, qrels and metrics as
`scripts/92_domain_eval.py`, so the numbers are directly comparable.

  bm25    exact, in-memory, Lucene's formula with the study's k1/b (notes: §Appendix uses
          k1=0.82, b=0.68, which reproduced the published MS MARCO number to 4 decimals)
  dense   e5-base-v2 pooled embeddings, cosine — the *backbone our own model is built on*,
          so it is the ceiling a frozen-encoder sparse projection is working against
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.encoders import get_encoder
from dvlsr.metrics import bootstrap_ci
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("dombase", "94_domain_baselines.log")
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def metrics(ranked, qids, qrels, k_mrr=10, k_r=100):
    rr, nd, rc = [], [], []
    for r, q in enumerate(qids):
        rel = qrels.get(q, set())
        if not rel:
            continue
        row = ranked[r]
        rr.append(next((1.0 / (i + 1) for i, d in enumerate(row[:k_mrr]) if int(d) in rel), 0.0))
        dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(row[:10]) if int(d) in rel)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), 10)))
        nd.append(dcg / idcg if idcg > 0 else 0.0)
        rc.append(len(rel & {int(x) for x in row[:k_r]}) / len(rel))
    return dict(mrr=float(np.mean(rr)), ndcg=float(np.mean(nd)), r100=float(np.mean(rc)),
                mrr_ci=bootstrap_ci(np.asarray(rr)), n_scored=len(rr))


def bm25(col, qids, qtexts, qrels, k1=0.82, b=0.68, k=100):
    import scipy.sparse as sp
    N = len(col)
    lens = np.zeros(N, np.float32)
    rows, cols, vals = [], [], []
    vocab = {}
    for i in range(N):
        toks = [t.lower() for t in WORD_RE.findall(col[i])]
        lens[i] = len(toks)
        for t, c in Counter(toks).items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(i); cols.append(j); vals.append(c)
    TF = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(N, len(vocab)))
    df = np.asarray((TF > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    avgdl = lens.mean()
    # BM25 weight of every (doc, term) pair, computed once
    W = TF.tocoo()
    denom = W.data + k1 * (1 - b + b * lens[W.row] / avgdl)
    W = sp.csr_matrix((W.data * (k1 + 1) / denom * idf[W.col], (W.row, W.col)), shape=TF.shape)
    ranked = np.zeros((len(qids), min(k, N)), np.int64)
    for r, qt in enumerate(qtexts):
        qi = [vocab[t.lower()] for t in WORD_RE.findall(qt) if t.lower() in vocab]
        if not qi:
            continue
        s = np.asarray(W[:, qi].sum(1)).ravel()
        ranked[r] = np.argpartition(-s, min(k, N) - 1)[:min(k, N)][np.argsort(-s[np.argpartition(-s, min(k, N) - 1)[:min(k, N)]])]
    return metrics(ranked, qids, qrels)


def dense(col, qids, qtexts, qrels, enc_name="e5", k=100, bs=128):
    enc = get_encoder(enc_name)
    D = torch.from_numpy(enc.encode_pooled(col.texts(range(len(col))), is_query=False,
                                           batch_size=bs)).cuda().float()
    Q = torch.from_numpy(enc.encode_pooled(qtexts, is_query=True, batch_size=bs)).cuda().float()
    ranked = np.zeros((len(qids), min(k, len(col))), np.int64)
    for s in range(0, len(qids), 64):
        sc = Q[s:s + 64] @ D.T
        ranked[s:s + sc.shape[0]] = torch.topk(sc, min(k, len(col)), dim=1).indices.cpu().numpy()
    del D, Q
    torch.cuda.empty_cache()
    return metrics(ranked, qids, qrels)


def splade(col, qids, qtexts, qrels, model, k=100, bs=64):
    """A SPLADE model on the same corpus. Its 30,522-dimension wordpiece vocabulary is
    fixed in the MLM head, which is exactly the design our text-defined vocabulary is an
    alternative to."""
    from dvlsr.encoders import SpladeEncoder
    import scipy.sparse as sp
    enc = SpladeEncoder.__new__(SpladeEncoder)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    # naver/splade-v3 is a gated repo: transformers re-validates against the hub even when
    # the weights are cached, so a token must be available. Read it from the environment;
    # it is never written to disk by this repository.
    tk = os.environ.get("HF_TOKEN") or None
    enc.tok = AutoTokenizer.from_pretrained(model, token=tk)
    enc.model = AutoModelForMaskedLM.from_pretrained(model, dtype=torch.float16,
                                                     token=tk).cuda().eval()
    enc.device = "cuda"
    enc.vocab = enc.tok.convert_ids_to_tokens(list(range(enc.tok.vocab_size)))
    nV = enc.model.config.vocab_size

    def csr(texts, maxlen, batch):
        rows, cols, vals = [], [], []
        for s0 in range(0, len(texts), batch):
            for r, (ii, vv) in enumerate(enc.encode(texts[s0:s0 + batch], maxlen=maxlen,
                                                    batch_size=batch)):
                rows.extend([s0 + r] * len(ii)); cols.extend(ii.tolist()); vals.extend(vv.tolist())
        return sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                             shape=(len(texts), nV))
    D = csr(col.texts(range(len(col))), paths.MAXLEN_DOC, bs)
    Q = csr(qtexts, paths.MAXLEN_QRY, bs)
    del enc
    torch.cuda.empty_cache()
    K = min(k, len(col))
    ranked = np.zeros((len(qids), K), np.int64)
    for s0 in range(0, len(qids), 32):
        sc = np.asarray((Q[s0:s0 + 32] @ D.T).todense())
        idx = np.argpartition(-sc, K - 1, axis=1)[:, :K]
        ranked[s0:s0 + sc.shape[0]] = np.take_along_axis(
            idx, np.argsort(-np.take_along_axis(sc, idx, 1), axis=1), 1)
    m = metrics(ranked, qids, qrels)
    m["nnz_d"] = float(D.getnnz(axis=1).mean())
    return m


def main(a):
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("d92", paths.REPO / "scripts" / "92_domain_eval.py")
    d92 = iu.module_from_spec(spec); spec.loader.exec_module(d92)
    col, qids, qtexts, qrels = d92.load_domain(a.name)
    out = {}
    p = paths.RESULTS / f"94_domain_baselines_{a.name}.json"
    if p.exists():
        out = json.load(open(p))
    if "bm25" not in out or a.force:
        with Timer(f"BM25 on {a.name} ({len(col)} passages)", lg):
            out["bm25"] = bm25(col, qids, qtexts, qrels)
        lg.info(f"{a.name} bm25:  MRR@10 {out['bm25']['mrr']:.4f}  "
                f"nDCG@10 {out['bm25']['ndcg']:.4f}  R@100 {out['bm25']['r100']:.4f}")
    dk = f"dense_{a.encoder}"
    if a.encoder == "e5" and "dense" in out and dk not in out:
        out[dk] = out["dense"]                  # results written before the key carried the encoder
    if dk not in out or a.force:
        with Timer(f"dense {a.encoder} on {a.name}", lg):
            out[dk] = dense(col, qids, qtexts, qrels, a.encoder)
        lg.info(f"{a.name} {dk}: MRR@10 {out[dk]['mrr']:.4f}  "
                f"nDCG@10 {out[dk]['ndcg']:.4f}  R@100 {out[dk]['r100']:.4f}")
    if a.encoder == "e5":
        out["dense"] = out[dk]
    for key, model in (("splade_v3", "naver/splade-v3"),
                       ("splade_pp", paths.SPLADE)):
        if a.skip_splade:
            break
        if key not in out or a.force:
            with Timer(f"{key} on {a.name}", lg):
                out[key] = splade(col, qids, qtexts, qrels, model)
            lg.info(f"{a.name} {key}: MRR@10 {out[key]['mrr']:.4f}  "
                    f"nDCG@10 {out[key]['ndcg']:.4f}  R@100 {out[key]['r100']:.4f}  "
                    f"nnz(d) {out[key]['nnz_d']:.0f}")
    out["corpus"] = a.name
    # two backbone drivers can compute rows for the same corpus at once: merge before saving
    if p.exists():
        cur = json.load(open(p))
        cur.update(out)
        out = cur
    save_json(out, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-splade", action="store_true",
                    help="only the dense row for --encoder (SPLADE rows are encoder-independent)")
    main(ap.parse_args())
