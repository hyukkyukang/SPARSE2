"""Pilot D training (§D.2, §D.3).

Variants
  V1  frozen encoder, fixed entry matrix, trained head
  V2  LoRA encoder, entries refreshed in a staggered way, trained head
  V3  fully fine-tuned encoder, entries fixed at initialisation, trained head
Options: vocabulary dropout (VD), oracle (train on the full vocabulary),
C-rand (replace V with random unit vectors).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening
from dvlsr.data import Collection, load_queries
from dvlsr.encoders import Encoder
from dvlsr.model import Head, sparse_rep, flops, build_entry_matrix
from dvlsr.util import get_logger, save_json, set_seed, Timer

lg = get_logger("trainD", "42_train.log")
DEV = "cuda"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def entry_raw(enc, layer, rep, r1_prefix):
    if rep == "R2":
        return np.asarray(np.load(paths.ART / f"proto_{enc}.npz")["protoA"][
            :, paths.LAYERS.index(layer)], np.float32)
    return np.asarray(np.load(paths.ART / f"r1_{enc}.npz")[r1_prefix], np.float32)


def token_tf(enc, layer, qside):
    mu, W = whitening.load(paths.ART / "whiten" / f"{enc}_L{layer}_{'Q' if qside else 'H'}.npz")
    return whitening.Transform("whitened", mu, W, DEV)


class Trainer:
    def __init__(self, a):
        self.a = a
        set_seed(a.seed)
        self.col = Collection()
        z = np.load(paths.ART / "vocab.npz")
        self.words = [str(w) for w in z["words"]]
        self.nV = len(self.words)
        sp = np.load(paths.PREP / "vocab_splits.npz")
        held = np.zeros(self.nV, bool) if (a.oracle or a.split == "none") else sp[a.split]
        self.held = held
        self.seen = ~held
        lg.info(f"split={a.split} oracle={a.oracle} seen={int(self.seen.sum())} "
                f"held={int(held.sum())}")

        V = entry_raw(a.encoder, a.layer, a.rep, a.r1_prefix)
        if a.crand:
            g = np.random.default_rng(a.seed)
            V = g.normal(size=V.shape).astype(np.float32)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
        self.E_all, self.muV, self.WV = build_entry_matrix(V, self.seen, DEV)
        self.seen_idx = torch.as_tensor(np.flatnonzero(self.seen), device=DEV)
        self.E_seen = self.E_all[self.seen_idx]

        self.enc = Encoder(a.encoder, dtype=torch.float32)
        if a.variant == "V1":
            self.enc.model.requires_grad_(False)
            self.enc.model.eval()
        elif a.variant == "V2":
            from peft import LoraConfig, get_peft_model
            # q/k/v/o as §D.3 specifies: "dense" alone would also catch both FFN
            # projections, which is a different (larger) adapter than the protocol asks for
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                             target_modules=["query", "key", "value",
                                             "attention.output.dense"])
            self.enc.model = get_peft_model(self.enc.model, cfg)
            self.enc.model.train()
        else:
            self.enc.model.requires_grad_(True)
            self.enc.model.train()

        self.tfH = token_tf(a.encoder, a.layer, False)
        self.tfQ = token_tf(a.encoder, a.layer, True)
        self.head = Head(768, 768, t_init=20.0, b_init=20.0 * a.tau).to(DEV)

        groups = [dict(params=list(self.head.parameters()), lr=a.lr_head)]
        if a.variant != "V1":
            enc_lr = 1e-4 if a.variant == "V2" else 2e-5
            groups.append(dict(params=[p for p in self.enc.model.parameters()
                                       if p.requires_grad], lr=enc_lr))
        self.opt = torch.optim.AdamW(groups, weight_decay=0.01)

        d = np.load(paths.PREP / "train_negatives.npz")
        self.qidx, self.pos, self.negs = d["qidx"], d["pos"], d["negs"]
        _, qtexts = load_queries(paths.QUERIES_TRAIN)
        self.qtexts = [qtexts[i] for i in self.qidx]
        self.n_q = len(self.qidx)
        self.bs = a.batch_queries
        self.steps = (self.n_q // self.bs) * a.epochs
        if a.max_steps:
            self.steps = min(self.steps, a.max_steps)
        self.ramp = int(0.2 * self.steps)
        lg.info(f"{self.n_q} queries, batch {self.bs}x8, {self.steps} steps")

    # ---------------------------------------------------------------- entries
    def refresh_entries(self, step):
        """V2: re-encode a random 10% of the seen entries' contexts (§D.3)."""
        a = self.a
        g = np.random.default_rng(a.seed * 1000 + step)
        pick = g.choice(np.flatnonzero(self.seen),
                        size=int(a.refresh_frac * self.seen.sum()), replace=False)
        occ = np.load(paths.ART / "vocab.npz")["occ_pid"][pick, : a.refresh_k]
        pids = np.unique(occ.reshape(-1))
        inv = {self.words[j]: j for j in pick}
        acc = torch.zeros(self.nV, 768, device=DEV)
        cnt = torch.zeros(self.nV, device=DEV)
        order = np.argsort([self.col.off[p + 1] - self.col.off[p] for p in pids])
        was = self.enc.model.training
        self.enc.model.eval()
        with torch.no_grad():
            for s in range(0, len(order), 384):
                sel = pids[order[s:s + 384]]
                wu = self.enc.encode_word_units(self.col.texts(sel), [a.layer],
                                                keep=lambda w: w in inv)
                if len(wu.word) == 0:
                    continue
                j = torch.tensor([inv[w.lower()] for w in wu.word], device=DEV)
                acc.index_add_(0, j, wu.states[:, 0].float())
                cnt.index_add_(0, j, torch.ones(len(j), device=DEV))
        if was:
            self.enc.model.train()
        upd = cnt > 0
        V = acc[upd] / cnt[upd].unsqueeze(1)
        V = (V - torch.as_tensor(self.muV, device=DEV)) @ torch.as_tensor(self.WV, device=DEV)
        self.E_all[upd] = F.normalize(V, dim=-1)
        self.E_seen = self.E_all[self.seen_idx]
        return int(upd.sum())

    # ---------------------------------------------------------------- step
    def encode(self, texts, qside, grad):
        wu = self.enc.encode_word_units(
            texts, [self.a.layer], is_query=qside,
            maxlen=paths.MAXLEN_QRY if qside else paths.MAXLEN_DOC, grad=grad)
        tf = self.tfQ if qside else self.tfH
        h = tf(wu.states[:, 0].to(DEV), normalize=False)
        return h, torch.as_tensor(wu.row.astype(np.int64), device=DEV)

    def run(self):
        a = self.a
        grad_enc = a.variant != "V1"
        sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / a.warmup) *
            max(0.0, 1 - max(0, s - a.warmup) / max(1, self.steps - a.warmup)))
        step = 0
        t0 = time.time()
        hist = []
        for ep in range(a.epochs):
            g = np.random.default_rng(a.seed * 7 + ep)
            order = g.permutation(self.n_q)
            for bi in range(self.n_q // self.bs):
                qs = order[bi * self.bs:(bi + 1) * self.bs]
                qtexts = [self.qtexts[i] for i in qs]
                dtexts, pos_idx = [], []
                for k, i in enumerate(qs):
                    cand = self.negs[i]
                    cand = cand[cand >= 0]
                    nn = g.choice(cand, size=min(7, len(cand)), replace=False)
                    pos_idx.append(len(dtexts))
                    dtexts.append(self.col[int(self.pos[i])])
                    dtexts.extend(self.col.texts(nn))
                pos_idx = torch.as_tensor(pos_idx, device=DEV)

                mask = None
                lam_scale = 1.0
                if a.vd:
                    gg = torch.Generator(device=DEV); gg.manual_seed(a.seed * 99991 + step)
                    keep = torch.rand(len(self.seen_idx), device=DEV, generator=gg) >= 0.3
                    mask = keep
                    lam_scale = 1.0 / 0.7

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hq, rq = self.encode(qtexts, True, grad_enc)
                    hd, rd = self.encode(dtexts, False, grad_enc)
                    sq = sparse_rep(self.head, hq, rq, len(qtexts), self.E_seen, mask)
                    sd = sparse_rep(self.head, hd, rd, len(dtexts), self.E_seen, mask)
                    scores = sq @ sd.T
                    ce = F.cross_entropy(scores.float(), pos_idx)
                    ramp = min(1.0, ((step + 1) / max(self.ramp, 1)) ** 2)
                    lq = a.lam_q * ramp * lam_scale
                    ld = a.lam_d * ramp * lam_scale
                    loss = ce + lq * flops(sq.float()) + ld * flops(sd.float())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for gp in self.opt.param_groups for p in gp["params"]], 1.0)
                self.opt.step(); sched.step()
                step += 1

                if step % 50 == 0:
                    with torch.no_grad():
                        nq = float((sq > 0).sum(1).float().mean())
                        nd = float((sd > 0).sum(1).float().mean())
                        acc = float((scores.argmax(1) == pos_idx).float().mean())
                    hist.append(dict(step=step, ce=float(ce), nnz_q=nq, nnz_d=nd,
                                     acc=acc, t=float(self.head.t), b=float(self.head.b)))
                    lg.info(f"[{a.name}] step {step}/{self.steps} ce={float(ce):.4f} "
                            f"acc={acc:.3f} nnz_q={nq:.0f} nnz_d={nd:.0f} "
                            f"t={float(self.head.t):.2f} b={float(self.head.b):.2f} "
                            f"({(time.time()-t0)/step:.2f}s/step)")
                if a.max_steps and step >= a.max_steps:
                    self.save(hist); return
                if a.variant == "V2" and step % a.refresh_every == 0:
                    n = self.refresh_entries(step)
                    lg.info(f"[{a.name}] refreshed {n} entries at step {step}")
        self.save(hist)

    def save(self, hist):
        a = self.a
        d = paths.CKPT / a.name
        d.mkdir(parents=True, exist_ok=True)
        torch.save(dict(head=self.head.state_dict(),
                        muV=self.muV, WV=self.WV,
                        E_all=self.E_all.cpu()), d / "head.pt")
        if a.variant == "V2":
            self.enc.model.save_pretrained(str(d / "lora"))
        elif a.variant == "V3":
            self.enc.model.save_pretrained(str(d / "encoder"))
        save_json(dict(vars(a), n_seen=int(self.seen.sum()), n_held=int(self.held.sum())),
                  d / "config.json")
        save_json(hist, d / "history.json")
        lg.info(f"saved {d}")


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="V1", choices=["V1", "V2", "V3"])
    ap.add_argument("--split", default="random",
                    choices=["random", "cluster", "rare", "none"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--crand", action="store_true")
    ap.add_argument("--vd", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rep", default="R2")
    ap.add_argument("--r1-prefix", default="none")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--lam-d", type=float, default=3e-4)
    ap.add_argument("--lam-q", type=float, default=9e-4)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-queries", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refresh-every", type=int, default=100)
    ap.add_argument("--refresh-frac", type=float, default=0.10)
    ap.add_argument("--refresh-k", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args(argv)
    if a.batch_queries is None:
        a.batch_queries = 128 if a.variant == "V1" else 32
    return a


if __name__ == "__main__":
    a = build_args()
    with Timer(f"train {a.name}", lg):
        Trainer(a).run()
