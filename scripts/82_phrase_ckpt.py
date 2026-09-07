"""Phrase / entity insertion pilot, step 3: derive the *inserted* model.

A model trained on the word vocabulary with the phrases absent is exactly V1oracle
(all 30k words seen, phrases not in V). Inserting the phrases means applying the
frozen entry-side map (mu_V, W_V from the seen words) to the phrase prototypes and
appending the rows -- no training. This writes ckpt/V1_phrase from ckpt/V1oracle so
the standard encode/eval stages can treat it like any other Pilot D run, with
V1oracle_phrase (trained on words + phrases) as its oracle.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("phrckpt", "82_phrase_ckpt.log")


def entry_norm_for(head, E, decile, seen, enc_name, layer, bank_n=32768, q=0.999,
                   chunk_rows=2048, chunk_ent=15000, device="cuda"):
    """The Pilot F affine for an *extended* vocabulary, computed the same way training
    computes it: background statistics of each entry against the frozen token bank, mapped
    onto the median of seen entries in the same frequency decile. Here 'seen' is the word
    vocabulary and the phrases are the inserted entries, so this is exactly what a
    deployment would compute when adding them."""
    from dvlsr import whitening
    from dvlsr.model import entry_norm_affine
    mu_h, W = whitening.load(paths.ART / "whiten" / f"{enc_name}_L{layer}_H.npz")
    tf = whitening.Transform("whitened", mu_h, W, device)
    bank = np.load(paths.ART / f"bankH_{enc_name}.npy", mmap_mode="r")
    li = paths.layers_for(enc_name).index(layer)
    g = np.random.default_rng(1)
    sel = np.sort(g.choice(bank.shape[0], size=min(bank_n, bank.shape[0]), replace=False))
    B = torch.as_tensor(np.asarray(bank[sel, li], np.float32), device=device)
    nV = E.shape[0]
    s1 = torch.zeros(nV, device=device, dtype=torch.float64)
    s2 = torch.zeros(nV, device=device, dtype=torch.float64)
    mtop = max(int((1 - q) * B.shape[0]), 8)
    top = torch.full((nV, mtop), -2.0, device=device)
    n = 0
    with torch.no_grad():
        for r in range(0, B.shape[0], chunk_rows):
            h = tf(B[r:r + chunk_rows], normalize=False)
            gh = F.normalize(head(h), dim=-1)
            n += gh.shape[0]
            for e in range(0, nV, chunk_ent):
                a = (gh @ E[e:e + chunk_ent].T).float()
                s1[e:e + chunk_ent] += a.sum(0).double()
                s2[e:e + chunk_ent] += (a * a).sum(0).double()
                cat = torch.cat([top[e:e + chunk_ent], a.T], 1)
                top[e:e + chunk_ent] = torch.topk(cat, mtop, dim=1).values
                del a, cat
    mu = (s1 / n).float()
    sd = (s2 / n - (s1 / n) ** 2).clamp(min=1e-12).sqrt().float()
    qv = top[:, -1].contiguous()
    dec = torch.as_tensor(decile.astype(np.int64), device=device)
    seen_t = torch.as_tensor(seen, device=device)
    q_star = torch.empty_like(qv)
    for d in range(10):
        m = dec == d
        if not bool(m.any()):
            continue
        selm = m & seen_t
        if not bool(selm.any()):
            selm = seen_t
        q_star[m] = qv[selm].median()
    held = torch.as_tensor(~seen, device=device)
    lg.info(f"phrase entries: background tail {float(qv[held].median()):.4f} vs word median "
            f"{float(qv[seen_t].median()):.4f}; mean shift {float((q_star - qv)[held].mean()):+.4f}")
    return torch.stack([torch.ones_like(qv), q_star - qv])


def main(src="V1oracle", dst="V1_phrase", enc="e5", norm=False):
    sd = paths.CKPT / src
    cfg = json.load(open(sd / "config.json"))
    blob = torch.load(sd / "head.pt", map_location="cpu", weights_only=False)
    z = np.load(paths.vocab_file("phr"))
    isp = z["is_phrase"]
    proto = np.load(paths.proto_file(enc, "phr"))["protoA"][:, paths.layers_for(enc).index(cfg["layer"])]
    Vp = torch.as_tensor(np.asarray(proto[isp], np.float32))
    mu, W = torch.as_tensor(blob["muV"]), torch.as_tensor(blob["WV"])
    Ep = F.normalize((Vp - mu) @ W, dim=-1)
    E_all = torch.cat([blob["E_all"].float(), Ep], 0)
    assert E_all.shape[0] == len(isp), (E_all.shape, len(isp))
    AB = None
    if norm:
        from dvlsr.model import Head
        head = Head(int(E_all.shape[1]), int(E_all.shape[1]), kind=cfg.get("head", "mlp")).to("cuda")
        head.load_state_dict(blob["head"]); head.eval()
        AB = entry_norm_for(head, E_all.to("cuda"), z["decile"], ~isp, enc, cfg["layer"]).cpu()
    dd = paths.CKPT / dst
    dd.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=blob["head"], muV=blob["muV"], WV=blob["WV"], E_all=E_all, AB=AB),
               dd / "head.pt")
    cfg2 = dict(cfg, name=dst, split="phrase", oracle=False, vocab_tag="phr",
                n_seen=int((~isp).sum()), n_held=int(isp.sum()), derived_from=src)
    save_json(cfg2, dd / "config.json")
    lg.info(f"wrote {dd}: {int((~isp).sum())} seen words + {int(isp.sum())} inserted phrases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="V1oracle")
    ap.add_argument("--dst", default="V1_phrase")
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--norm", action="store_true",
                    help="also compute the Pilot F tail-normalisation affine for the phrases")
    a = ap.parse_args()
    main(a.src, a.dst, a.encoder, a.norm)
