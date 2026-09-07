"""Domain-shift evaluation, step 3: does inserting a real domain vocabulary help?

Takes a model trained on MS MARCO, appends entries for terminology the target corpus uses
and the model has never had a dimension for (scripts/91_domain_vocab.py), and retrieves on
the target corpus twice: with those entries active, and with them zeroed.

The comparison is the honest version of the study's recovery ratio. There is no oracle
here -- nothing was trained on this corpus, which is the point -- so we report the
quantities that do not need one:

  MRR@10 / nDCG@10 / R@100  with the domain entries vs with them zeroed
  the activation gap between inserted and trained entries, within frequency decile
  BM25 on the same corpus, as the reference every deployment already has

Optionally applies the Pilot F tail correction to the inserted entries (--norm), computed
against the same frozen token bank, which is exactly what a deployment would do.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.metrics import bootstrap_ci, paired_bootstrap
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("domeval", "92_domain_eval.log")
DEV = "cuda"


def load_domain(name):
    sys.path.insert(0, str(paths.REPO / "scripts"))
    import importlib.util as iu
    spec = iu.spec_from_file_location("d91", paths.REPO / "scripts" / "91_domain_vocab.py")
    d91 = iu.module_from_spec(spec); spec.loader.exec_module(d91)
    col = d91.DomainCollection(name)
    d = paths.DATA / "domain" / name
    qids, qtexts = [], []
    for line in open(d / "queries.tsv"):
        a, _, b = line.rstrip("\n").partition("\t")
        qids.append(a); qtexts.append(b)
    pos = {i: r for r, i in enumerate(col.ids)}
    qrels = {}
    for line in open(d / "qrels.tsv"):
        q, _, doc, _ = line.split()
        if doc in pos:
            qrels.setdefault(q, set()).add(pos[doc])
    return col, qids, qtexts, qrels


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum((1.0 / np.log2(i + 2)) for i, d in enumerate(ranked[:k]) if int(d) in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def encode(col_texts, enc, head, tf, E, AB, layer, qside, cap, bs=64):
    out_i = np.zeros((len(col_texts), cap), np.int32)
    out_v = np.zeros((len(col_texts), cap), np.float32)
    from dvlsr.model import segment_max
    from dvlsr import precision
    with torch.no_grad(), precision.autocast():
        for s in range(0, len(col_texts), bs):
            chunk = col_texts[s:s + bs]
            wu = enc.encode_word_units(chunk, [layer], is_query=qside,
                                       maxlen=paths.MAXLEN_QRY if qside else paths.MAXLEN_DOC)
            if len(wu.word) == 0:
                continue
            h = tf(wu.states[:, 0].to(DEV), normalize=False)
            row = torch.as_tensor(wu.row.astype(np.int64), device=DEV)
            z = head.logits(h.to(E.dtype), E, AB)
            sv = torch.log1p(torch.relu(segment_max(z, row, len(chunk))))
            v, i = torch.topk(sv.float(), min(cap, sv.shape[1]), dim=1)
            out_i[s:s + len(chunk)] = i.cpu().numpy()
            out_v[s:s + len(chunk)] = v.cpu().numpy()
            del z, sv
    return out_i, out_v


def main(a):
    col, qids, qtexts, qrels = load_domain(a.name)
    tag = f"dom_{a.name}" + ("" if a.select == "df" else f"_{a.select}")
    z = np.load(paths.vocab_file(tag))
    decile = z["decile"]

    import importlib.util as iu
    spec = iu.spec_from_file_location("e43", paths.REPO / "scripts" / "43_encode_eval.py")
    e43 = iu.module_from_spec(spec); spec.loader.exec_module(e43)
    cfg, head, enc, E_base, AB_base = e43.load_ckpt(a.name_model)
    layer = cfg["layer"]
    li = paths.layers_for(cfg["encoder"]).index(layer)

    # The word list of a domain tag is encoder-independent, but whether an entry has a
    # prototype is not (an encoder may drop a word its tokenizer cannot place). So the
    # inserted set is the tag's domain words that THIS encoder built a prototype for.
    pz = np.load(paths.proto_file(cfg["encoder"], tag))
    # `is_domain` (the tag's word list) rather than the split file's `domain`, which older
    # files wrote as e5's own has-a-prototype mask and would cap every backbone at e5's set
    dom = z["is_domain"] & (pz["cntA"] > 0)
    lg.info(f"{a.name}: {len(col)} passages, {len(qids)} queries, "
            f"{int(dom.sum())} inserted entries (encoder {cfg['encoder']}, layer {layer})")

    # inserted rows: the frozen entry-side map of the trained model applied to the domain
    # prototypes -- exactly what "adding an entry" means (§D.4)
    blob = torch.load(paths.CKPT / a.name_model / "head.pt", map_location="cpu", weights_only=False)
    proto = pz["protoA"][:, li]
    Vd = torch.as_tensor(np.asarray(proto[dom], np.float32))
    mu, W = torch.as_tensor(blob["muV"]), torch.as_tensor(blob["WV"])
    Ed = F.normalize((Vd - mu) @ W, dim=-1).to(DEV)
    dec_ins = decile[dom]              # frequency band of each INSERTED entry
    if a.control == "rand":
        # same number of dimensions, no linguistic content: isolates capacity from terminology
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        Ed = F.normalize(torch.randn(Ed.shape[0], Ed.shape[1], generator=g), dim=-1).to(DEV)
        lg.info(f"CONTROL rand: {Ed.shape[0]} random unit vectors instead of domain entries")
    elif a.control:
        # the OTHER corpus's domain vocabulary: same count, same construction, wrong domain
        ot = f"dom_{a.control}"
        oz = np.load(paths.vocab_file(ot))
        opz = np.load(paths.proto_file(cfg["encoder"], ot))
        odom = oz["is_domain"] & (opz["cntA"] > 0)
        op = opz["protoA"][:, li]
        keep = slice(0, min(int(odom.sum()), Ed.shape[0]))
        Vo = torch.as_tensor(np.asarray(op[odom], np.float32))[keep]
        Ed = F.normalize((Vo - mu) @ W, dim=-1).to(DEV)
        dec_ins = oz["decile"][odom][keep]     # the other corpus's own bands
        lg.info(f"CONTROL {a.control}: {Ed.shape[0]} entries from a DIFFERENT corpus, "
                f"e.g. {[str(w) for w in oz['words'][odom][:8]]}")
    E = torch.cat([E_base, Ed], 0)
    lg.info(f"entry matrix: {E_base.shape[0]} trained + {Ed.shape[0]} inserted")

    mu_h, Wh = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_H.npz")
    tfH = whitening.Transform("whitened", mu_h, Wh, DEV)
    mu_q, Wq = whitening.load(paths.ART / "whiten" / f"{cfg['encoder']}_L{layer}_Q.npz")
    tfQ = whitening.Transform("whitened", mu_q, Wq, DEV)

    AB = None
    if a.norm:
        spec2 = iu.spec_from_file_location("d82", paths.REPO / "scripts" / "82_phrase_ckpt.py")
        d82 = iu.module_from_spec(spec2); spec2.loader.exec_module(d82)
        seen = np.concatenate([np.ones(E_base.shape[0], bool), np.zeros(Ed.shape[0], bool)])
        dec = np.concatenate([decile[:E_base.shape[0]], dec_ins])
        AB = d82.entry_norm_for(head, E, dec, seen, cfg["encoder"], layer).to(DEV)
        # Insertion-time calibration of the INSERTED entries only. entry_norm_for also maps
        # every trained entry to its decile median; leaving that in would recalibrate the
        # trained vocabulary too, and the row would no longer differ from the baseline by
        # the inserted entries alone.
        seen_t = torch.as_tensor(seen, device=DEV)
        AB[0][seen_t] = 1.0
        AB[1][seen_t] = 0.0
        lg.info(f"tail correction applied to the inserted entries only "
                f"(mean shift {float(AB[1][~seen_t].mean()):+.4f})")

    with Timer(f"encode {a.name} (extended vocabulary)", lg):
        di, dv = encode(col.texts(range(len(col))), enc, head, tfH, E, AB, layer, False, a.cap)
        qi, qv = encode(qtexts, enc, head, tfQ, E, AB, layer, True, 256)
    # The baseline must be its OWN encode over the trained vocabulary only. Deriving it by
    # masking the inserted entries out of the extended encode is wrong whenever a document
    # has more than `cap` non-zero entries: the inserted entries win storage slots and push
    # base entries out of the store, and masking cannot bring them back. That made the
    # "no addition" baseline depend on which entries were inserted (trec-covid: 0.6284 under
    # one selection rule, 0.6630 under another, for what is the same model).
    # this baseline depends only on (corpus, model), never on the selection rule or the
    # calibration, so cache it: on a 171k-passage corpus it is half the run time
    # keyed on the checkpoint's modification time too, so retraining a model never reuses
    # a stale baseline; written atomically so an interrupted run cannot leave a partial file
    ck_mtime = int(os.path.getmtime(paths.CKPT / a.name_model / "head.pt"))
    bc = paths.DATA / "domain" / a.name / f"base_{a.name_model}_cap{a.cap}_{ck_mtime}.npz"
    if bc.exists():
        z0 = np.load(bc)
        bi, bv, bqi, bqv = z0["di"], z0["dv"], z0["qi"], z0["qv"]
        lg.info(f"baseline encode reused from {bc.name}")
    else:
        with Timer(f"encode {a.name} (trained vocabulary only)", lg):
            bi, bv = encode(col.texts(range(len(col))), enc, head, tfH, E_base, AB_base,
                            layer, False, a.cap)
            bqi, bqv = encode(qtexts, enc, head, tfQ, E_base, AB_base, layer, True, 256)
        tmp = bc.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, di=bi, dv=bv, qi=bqi, qv=bqv)
        os.replace(tmp, bc)

    nV = E.shape[0]
    ins = np.zeros(nV, bool); ins[E_base.shape[0]:] = True
    res = dict(model=a.name_model, corpus=a.name, norm=bool(a.norm), control=a.control or None,
               select=a.select,
               n_passages=len(col), n_queries=len(qids), n_inserted=int(ins.sum()))
    per = {}
    for cond in ("with", "without"):
        if cond == "with":
            Qi, Q, Di, Dv, nc = qi, qv, di, dv, nV
        else:
            Qi, Q, Di, Dv, nc = bqi, bqv, bi, bv, E_base.shape[0]
        # Exact dense scoring, chunked over documents so a 170k-passage corpus does not
        # need a (n_docs x |V|) matrix resident at once.
        K = min(100, len(col))
        Qm = torch.zeros(len(qids), nc, device=DEV)
        Qm.scatter_(1, torch.as_tensor(Qi.astype(np.int64), device=DEV),
                    torch.as_tensor(Q, device=DEV))
        best_v = torch.full((len(qids), K), -1.0, device=DEV)
        best_i = torch.zeros((len(qids), K), dtype=torch.long, device=DEV)
        for s in range(0, len(col), a.dchunk):
            e = min(s + a.dchunk, len(col))
            Dm = torch.zeros(e - s, nc, device=DEV)
            Dm.scatter_(1, torch.as_tensor(Di[s:e].astype(np.int64), device=DEV),
                        torch.as_tensor(Dv[s:e], device=DEV))
            sc = Qm @ Dm.T
            k = min(K, e - s)
            v, i = torch.topk(sc, k, dim=1)
            cv = torch.cat([best_v, v], 1)
            ci = torch.cat([best_i, i + s], 1)
            o = torch.topk(cv, K, dim=1).indices
            best_v = torch.gather(cv, 1, o); best_i = torch.gather(ci, 1, o)
            del Dm, sc
        ranked = best_i.cpu().numpy()
        del Qm, best_v, best_i
        torch.cuda.empty_cache()
        rr, nd, rc = [], [], []
        for r, q in enumerate(qids):
            rel = qrels.get(q, set())
            if not rel:
                continue
            row = ranked[r]
            rr.append(next((1.0 / (i + 1) for i, d in enumerate(row[:10]) if int(d) in rel), 0.0))
            nd.append(ndcg_at_k(row, rel, 10))
            rc.append(len(rel & {int(x) for x in row[:100]}) / len(rel))
        per[cond] = dict(mrr=np.asarray(rr), ndcg=np.asarray(nd), r100=np.asarray(rc))
        res[cond] = dict(**{k: float(v.mean()) for k, v in per[cond].items()},
                         mrr_ci=bootstrap_ci(per[cond]["mrr"]),
                         nnz_d=float((Dv > 0).sum(1).mean()),
                         cap_hit=float(((Dv > 0).sum(1) >= a.cap).mean()), n_scored=len(rr))
        lg.info(f"{cond:8s} entries: MRR@10 {np.mean(rr):.4f}  nDCG@10 {np.mean(nd):.4f}  "
                f"R@100 {np.mean(rc):.4f}")
    for k in ("mrr", "ndcg", "r100"):
        res[f"delta_{k}"] = paired_bootstrap(per["with"][k], per["without"][k])
    d = res["delta_mrr"]
    lg.info(f"inserting the domain vocabulary: MRR@10 {d['diff']:+.4f} "
            f"[{d['lo']:+.4f}, {d['hi']:+.4f}], significant={d['excludes_zero']}")

    # activation gap, inserted vs trained entries, within frequency decile
    df = np.bincount(di[dv > 0], minlength=nV) / len(col)
    dec_all = np.concatenate([decile[:E_base.shape[0]], dec_ins])
    logs = []
    for dd in range(10):
        sel = dec_all == dd
        h, s = sel & ins, sel & ~ins
        if h.sum() < 5 or s.sum() < 20:
            continue
        logs.append(float(np.log((np.median(df[h]) + 1e-9) / (np.median(df[s]) + 1e-9))))
    res["signed_gap_r"] = float(np.median(logs)) if logs else float("nan")
    lg.info(f"activation gap, inserted vs trained: {res['signed_gap_r']:+.3f}")
    sfx = ("" if a.select == "df" else f"_{a.select}") + ("_norm" if a.norm else "") \
        + (f"_ctl-{a.control}" if a.control else "")
    save_json(res, paths.RESULTS / f"92_domain_{a.name}_{a.name_model}{sfx}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--name-model", default="V1oracle")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--control", default="",
                    help="'rand' for random unit vectors, or another corpus name for its "
                         "vocabulary: both keep the dimension count and remove the terminology")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--dchunk", type=int, default=20000)
    ap.add_argument("--select", default="df", choices=["df", "tfidf", "dfcap"])
    main(ap.parse_args())
