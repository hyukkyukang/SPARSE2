"""Encode C1 (and dev queries / the statistics sample S) with a trained Pilot D model.

Stores the top-1024 (entry, weight) pairs per document -- the §D.6 safety cap --
so that "all entries" and "held-out entries zeroed" are two maskings of one
encoding rather than two passes.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.data import Collection, load_queries, load_splits
from dvlsr.encoders import Encoder
from dvlsr.model import Head, segment_max
from dvlsr.util import get_logger, gpu_map, Timer

lg = get_logger("encD", "43_encode_eval.log")
CAP = 1024
CAP_Q = 256


def load_ckpt(name, device="cuda"):
    d = paths.CKPT / name
    cfg = json.load(open(d / "config.json"))
    blob = torch.load(d / "head.pt", map_location=device, weights_only=False)
    head = Head(768, 768, kind=cfg.get("head", "mlp")).to(device)
    head.load_state_dict(blob["head"])
    head.eval()
    enc = Encoder(cfg["encoder"], dtype=torch.float32)
    if cfg["variant"] == "V2":
        from peft import PeftModel
        enc.model = PeftModel.from_pretrained(enc.model, str(d / "lora")).merge_and_unload()
    elif cfg["variant"] == "V3":
        from transformers import AutoModel
        enc.model = AutoModel.from_pretrained(str(d / "encoder"),
                                              dtype=torch.float32).to(device)
    enc.model.eval()
    E = blob["E_all"].to(device)
    AB = blob.get("AB")
    AB = None if AB is None else AB.to(device)      # Pilot F entry normalisation
    return cfg, head, enc, E, AB


def out_paths(name, kind):
    b = paths.EMB / f"D_{name}_{kind}"
    return b.with_name(b.name + "_idx.npy"), b.with_name(b.name + "_val.npy")


def wait_for_gpu(min_free_gb=5.0, timeout_s=600, poll_s=20):
    """Block until this shard's card has room. Several instances share the cards, so an
    encode can start just as a trainer allocates; waiting costs minutes, an out-of-memory
    crash costs the whole encode (and every shard with it)."""
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free >= min_free_gb or time.time() - t0 > timeout_s:
            return free
        print(f"waiting for GPU memory: {free:.1f} GB free, need {min_free_gb:.1f}", flush=True)
        time.sleep(poll_s)


def worker(shard, n_shards, name, kind, bs, calib=None, out_tag=None, min_free_gb=5.0):
    if min_free_gb > 0:
        print(f"shard {shard}: {wait_for_gpu(min_free_gb):.1f} GB free at start", flush=True)
    cfg, head, enc, E, AB = load_ckpt(name)
    cal = None
    if calib:
        c = np.load(calib)
        cal = (torch.as_tensor(c[0], device="cuda"), torch.as_tensor(c[1], device="cuda"))
    layer = cfg["layer"]
    mu, W = whitening.load(paths.ART / "whiten" /
                           f"{cfg['encoder']}_L{layer}_{'Q' if kind == 'q' else 'H'}.npz")
    tf = whitening.Transform("whitened", mu, W, "cuda")
    if kind == "q":
        _, texts = load_queries(paths.QUERIES_DEV_SMALL)
        n, cap, qside = len(texts), CAP_Q, True
        get = lambda sel: [texts[i] for i in sel]
    else:
        pids = (np.load(paths.PREP / "c1_pids.npy") if kind == "d"
                else load_splits()["S"])
        col = Collection()
        n, cap, qside = len(pids), CAP, False
        get = lambda sel: col.texts(pids[sel])
        lens = None
    b = np.linspace(0, n, n_shards + 1).astype(np.int64)
    lo, hi = int(b[shard]), int(b[shard + 1])
    tag = out_tag or name
    AI = np.lib.format.open_memmap(out_paths(tag, kind)[0], mode="r+")
    AV = np.lib.format.open_memmap(out_paths(tag, kind)[1], mode="r+")
    t0 = time.time()
    with torch.no_grad(), precision.autocast():
        for s in range(lo, hi, bs):
            sel = np.arange(s, min(s + bs, hi))
            wu = enc.encode_word_units(get(sel), [layer], is_query=qside,
                                       maxlen=paths.MAXLEN_QRY if qside
                                       else paths.MAXLEN_DOC)
            if len(wu.word) == 0:
                continue
            h = tf(wu.states[:, 0].cuda(), normalize=False)
            row = torch.as_tensor(wu.row.astype(np.int64), device="cuda")
            z = head.logits(h.to(E.dtype), E, AB)
            if cal is not None:
                z = z * cal[0] + cal[1]
            p = segment_max(z, row, len(sel))
            sv = torch.log1p(torch.relu(p))
            v, i = torch.topk(sv.float(), min(cap, sv.shape[1]), dim=1)
            AI[sel] = i.cpu().numpy().astype(np.int32)
            AV[sel] = v.cpu().numpy().astype(np.float16)
            del z, p, sv
            if (s - lo) % (bs * 200) == 0:
                print(f"shard {shard}: {s-lo}/{hi-lo} "
                      f"{(s-lo+bs)/max(time.time()-t0,1e-9):.0f}/s", flush=True)
    AI.flush(); AV.flush()
    print(f"shard {shard} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=32)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="d", choices=["d", "q", "s"])
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    a = ap.parse_args()
    if a.shard >= 0:
        worker(a.shard, a.n_shards, a.name, a.kind, a.bs, a.calib, a.out_tag, a.min_free_gb)
    else:
        n = {"d": len(np.load(paths.PREP / "c1_pids.npy")),
             "q": len(load_queries(paths.QUERIES_DEV_SMALL)[0]),
             "s": len(load_splits()["S"])}[a.kind]
        cap = CAP_Q if a.kind == "q" else CAP
        tag = a.out_tag or a.name
        pi, pv = out_paths(tag, a.kind)
        if not pi.exists():
            np.lib.format.open_memmap(pi, mode="w+", dtype=np.int32, shape=(n, cap))
            np.lib.format.open_memmap(pv, mode="w+", dtype=np.float16, shape=(n, cap))
        ns = a.n_shards if a.kind != "q" else 8
        with Timer(f"encode {a.kind} for {a.name} ({n})", lg):
            extra = ["--name", a.name, "--kind", a.kind, "--bs", str(a.bs),
                     "--min-free-gb", str(a.min_free_gb)]
            if a.calib:
                extra += ["--calib", a.calib]
            if a.out_tag:
                extra += ["--out-tag", a.out_tag]
            gpu_map(os.path.abspath(__file__), ns, extra, logger=lg, per_gpu=a.per_gpu)
