"""Does the query side really need its own whitening? (§0.5 check, measured end to end)

§0.5 mandates a separate query-side transform when the query token distribution
differs from the document side by more than 5%; ours differs by far more. This
script measures what that decision is worth in MRR@10 by re-encoding the dev
queries with the *document-side* transform and re-running the best cell.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.data import load_queries, qrels_dict
from dvlsr.metrics import evaluate, bootstrap_ci, paired_bootstrap
from dvlsr.retrieval import InvertedIndex
from dvlsr.sparse import profile_maxpool
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("qside", "33_qside_ablation.log")


def encode_queries(enc, layer, rep, r1_prefix, transform, query_side, top=128):
    import importlib.util as iu
    spec = iu.spec_from_file_location("e30", paths.REPO / "scripts" / "30_encode_c1.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    from dvlsr.encoders import Encoder
    E = m.entry_matrix(enc, layer, rep, r1_prefix, transform)
    tf = m.token_tf(enc, layer, query_side, transform)
    e = Encoder(enc)
    _, texts = load_queries(paths.QUERIES_DEV_SMALL)
    I = np.zeros((len(texts), top), np.int32)
    V = np.zeros((len(texts), top), np.float16)
    for s in range(0, len(texts), 256):
        chunk = texts[s:s + 256]
        wu = e.encode_word_units(chunk, [layer], is_query=True, maxlen=paths.MAXLEN_QRY)
        H = tf(wu.states[:, 0].cuda()).half()
        row = torch.as_tensor(wu.row.astype(np.int64), device="cuda")
        P = profile_maxpool(H, row, len(chunk), E)
        v, i = torch.topk(P, top, dim=1)
        I[s:s + len(chunk)] = i.cpu().numpy()
        V[s:s + len(chunk)] = v.cpu().numpy().astype(np.float16)
        del P
    return I, V


def main(a):
    d = json.load(open(paths.RESULTS / a.pilotc))
    enc, layer, rep = d["encoder"], d["layer"], d["rep"]
    transform = d.get("transform", "whitened")
    bc = d["best_cfg"]
    import importlib.util as iu
    spec = iu.spec_from_file_location("p31", paths.REPO / "scripts" / "31_pilotC.py")
    p31 = iu.module_from_spec(spec); spec.loader.exec_module(p31)
    di, dv = p31.store(enc, layer, rep, "d", transform)
    with Timer("build index", lg):
        ix = InvertedIndex(np.asarray(di), np.asarray(dv), 30000,
                           min_val=float(d["tau_d"][str(bc["nnz_d"])]["tau"]) - 1e-6)
    pids = np.load(paths.PREP / "c1_pids.npy")
    qids, _ = load_queries(paths.QUERIES_DEV_SMALL)
    qr = {int(x): set(int(y) for y in v)
          for x, v in qrels_dict(paths.QRELS_DEV_SMALL).items()}
    out, per = {}, {}
    for tag, qside in (("own_query_transform", True), ("shared_doc_transform", False)):
        with Timer(f"encode queries [{tag}]", lg):
            qi, qv = encode_queries(enc, layer, rep, a.r1_prefix, transform, qside)
        # tau_q refit for this space so density is matched, per the §B.2.3 rule
        Vt = torch.as_tensor(np.asarray(qv, np.float32), device="cuda")
        lo, hi = -0.5, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if float((Vt > mid).sum(1).float().mean()) > bc["nnz_q"]:
                lo = mid
            else:
                hi = mid
        tq = (lo + hi) / 2
        docs, _ = ix.search(qi, np.asarray(qv, np.float32), tq,
                            d["tau_d"][str(bc["nnz_d"])]["tau"], bc["k_q"], bc["k_d"],
                            bc["saturation"], k=1000)
        run = {int(q): pids[docs[i]] for i, q in enumerate(qids)}
        m, pq = evaluate(run, qr)
        m["mrr@10_ci"] = bootstrap_ci(pq["mrr@10"])
        m["tau_q"] = tq
        out[tag] = m
        per[tag] = pq["mrr@10"]
        lg.info(f"{tag}: MRR@10={m['mrr@10']:.4f} R@100={m['r@100']:.4f} tau_q={tq:.4f}")
    out["paired_bootstrap_own_minus_shared"] = paired_bootstrap(
        per["own_query_transform"], per["shared_doc_transform"])
    save_json(out, paths.RESULTS / "33_qside_ablation.json")
    lg.info(json.dumps(out["paired_bootstrap_own_minus_shared"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilotc", default="31_pilotC_e5_L9_R2.json")
    ap.add_argument("--r1-prefix", default="none")
    main(ap.parse_args())
