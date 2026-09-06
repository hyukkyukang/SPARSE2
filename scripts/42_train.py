"""Pilot D training (§D.2, §D.3), plus the arms added by the follow-up study.

Variants
  V1  frozen encoder, fixed entry matrix, trained head
  V2  LoRA encoder, entries refreshed, trained head
  V3  fully fine-tuned encoder, entries fixed at initialisation, trained head

Options
  --vd                    vocabulary dropout (§D.3); --vd-mode entry|cluster
                          entry:   mask a random 30% of seen entries per step
                          cluster: mask whole k-means clusters of the seen entry space
                                   until ~30% of entries are masked -- the training-time
                                   analogue of the §D.4 cluster split
  --meta-frac f           reserve whole clusters (~f of the seen entries) that are
                          *never* in the ranking objective; a calibration loss pushes
                          their activation rate to match the trained entries' within
                          each frequency decile (dvlsr.model.calibration_loss)
  --head mlp|linear       the §D.1 residual MLP, or a residual linear map (capacity control)
  --teacher none|dense    distil the encoder's own pooled (dense) score into the sparse
                          score (KL over the batch candidates); V1 only
  --oracle / --crand      train on the full vocabulary / replace V with random vectors
  --vocab-tag             use an extended vocabulary (e.g. the phrase pilot's)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths, whitening, precision
from dvlsr.data import Collection, load_queries
from dvlsr.encoders import Encoder
from dvlsr.model import (Head, EntryHead, sparse_rep, flops, build_entry_matrix,
                         kmeans_labels, calibration_loss, entry_norm_affine)
from dvlsr.util import get_logger, save_json, set_seed, Timer
from torch.utils.checkpoint import checkpoint

lg = get_logger("trainD", "42_train.log")
DEV = "cuda"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def entry_raw(enc, layer, rep, r1_prefix, tag=""):
    if rep == "R2":
        return np.asarray(np.load(paths.proto_file(enc, tag))["protoA"][
            :, paths.LAYERS.index(layer)], np.float32)
    return np.asarray(np.load(paths.ART / f"r1_{enc}.npz")[r1_prefix], np.float32)


def token_tf(enc, layer, qside):
    mu, W = whitening.load(paths.ART / "whiten" / f"{enc}_L{layer}_{'Q' if qside else 'H'}.npz")
    return whitening.Transform("whitened", mu, W, DEV)


def wait_for_gpu(min_free_gb, timeout_s=2400, poll_s=30):
    """Block until the assigned GPU has room, instead of crashing on the first backward.

    Several driver instances share the cards, so a run can start just as an encode grabs
    memory. Waiting costs minutes; an out-of-memory crash costs the whole run.
    """
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free >= min_free_gb:
            return free
        if time.time() - t0 > timeout_s:
            lg.warning(f"only {free:.1f} GB free after {timeout_s}s; starting anyway")
            return free
        lg.info(f"waiting for GPU memory: {free:.1f} GB free, need {min_free_gb:.1f} GB")
        time.sleep(poll_s)


class Trainer:
    def __init__(self, a):
        self.a = a
        set_seed(a.seed)
        if a.min_free_gb > 0:
            lg.info(f"GPU has {wait_for_gpu(a.min_free_gb):.1f} GB free at start")
        self.col = Collection()
        z = np.load(paths.vocab_file(a.vocab_tag))
        self.words = [str(w) for w in z["words"]]
        self.decile = z["decile"].astype(np.int64)
        self.nV = len(self.words)
        if a.oracle or a.split == "none":
            held = np.zeros(self.nV, bool)
        else:
            held = np.load(paths.vsplits_file(a.vocab_tag))[a.split]
        self.held = held
        self.seen = ~held
        lg.info(f"split={a.split} oracle={a.oracle} vocab_tag='{a.vocab_tag}' |V|={self.nV} "
                f"seen={int(self.seen.sum())} held={int(held.sum())}")

        V = entry_raw(a.encoder, a.layer, a.rep, a.r1_prefix, a.vocab_tag)
        if a.crand:
            g = np.random.default_rng(a.seed)
            V = g.normal(size=V.shape).astype(np.float32)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
        if a.entry_tf == "shared":
            # token-side whitening for the entry side too: a transform that does not depend
            # on which entries are seen (tests whether seen-only entry whitening is what makes
            # a held-out *region* over-fire)
            muH, WH = whitening.load(paths.ART / "whiten" / f"{a.encoder}_L{a.layer}_H.npz")
            X = torch.as_tensor(V, device=DEV)
            X = (X - torch.as_tensor(muH, device=DEV)) @ torch.as_tensor(WH, device=DEV)
            self.E_all, self.muV, self.WV = F.normalize(X, dim=-1), muH, WH
        else:
            self.E_all, self.muV, self.WV = build_entry_matrix(V, self.seen, DEV)
        seen_np = np.flatnonzero(self.seen)
        self.seen_idx = torch.as_tensor(seen_np, device=DEV)
        self.E_seen = self.E_all[self.seen_idx]
        self.n_seen = int(len(seen_np))
        self.seen_decile = torch.as_tensor(self.decile[seen_np], device=DEV)

        # ---- cluster structure over the SEEN entries (cluster-VD, meta clusters) ----
        self.vd_labels = None
        self.meta = None                                   # bool over seen entries
        if (a.vd and a.vd_mode == "cluster") or a.meta_frac > 0:
            with Timer(f"k-means k={a.vd_clusters} over seen entries", lg):
                self.vd_labels = kmeans_labels(self.E_seen.float(), a.vd_clusters, seed=a.seed)
            sizes = torch.bincount(self.vd_labels, minlength=a.vd_clusters)
            lg.info(f"cluster sizes: min={int(sizes.min())} med={float(sizes.float().median()):.0f} "
                    f"max={int(sizes.max())}")
        if a.meta_frac > 0:
            g = np.random.default_rng(a.seed * 31 + 7)
            meta = torch.zeros(self.n_seen, dtype=torch.bool, device=DEV)
            for c in g.permutation(a.vd_clusters):
                if int(meta.sum()) >= a.meta_frac * self.n_seen:
                    break
                meta |= self.vd_labels == int(c)
            self.meta = meta
            lg.info(f"meta-held-out: {int(meta.sum())} of {self.n_seen} seen entries "
                    f"({float(meta.float().mean()):.3f}) never enter the ranking loss")
        self.trainable = torch.ones(self.n_seen, dtype=torch.bool, device=DEV)
        if self.meta is not None:
            self.trainable &= ~self.meta

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
            if a.grad_ckpt:   # encoder activations do not fit a 24 GB card otherwise
                self.enc.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            self.enc.model = get_peft_model(self.enc.model, cfg)
            self.enc.model.train()
        else:
            self.enc.model.requires_grad_(True)
            if a.grad_ckpt:
                self.enc.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            self.enc.model.train()
        if a.teacher != "none":
            assert a.variant == "V1", "the dense teacher is the frozen encoder's own score"

        self.tfH = token_tf(a.encoder, a.layer, False)
        self.tfQ = token_tf(a.encoder, a.layer, True)
        self.AB = self.AB_seen = None          # entry-norm affine (see refresh_entry_norm)
        self._bank = None
        self.head = Head(768, 768, t_init=20.0, b_init=20.0 * a.tau, kind=a.head).to(DEV)

        self.E_head = None
        if a.entry_head:
            self.E_head = EntryHead(768, 768).to(DEV)
            lg.info(f"entry-side head: {sum(p.numel() for p in self.E_head.parameters()):,} "
                    f"shared parameters, identity at step 0")

        self.E_param = None
        if a.entry_param:
            # The control that isolates what defining the vocabulary by text costs: the SEEN
            # rows become free parameters (as in SPLADE, where each output dimension owns a
            # learned row), while a held-out entry is still inserted as its text-defined
            # vector. rho then measures what a new dimension is worth once the trained
            # vocabulary has drifted into a space of its own.
            import torch.nn as nn
            self.E_param = nn.Parameter(self.E_all[self.seen_idx].clone())
            lg.info(f"entry-param control: {tuple(self.E_param.shape)} trainable entry rows")

        groups = [dict(params=list(self.head.parameters()), lr=a.lr_head)]
        if self.E_param is not None:
            groups.append(dict(params=[self.E_param], lr=a.lr_head))
        if self.E_head is not None:
            groups.append(dict(params=list(self.E_head.parameters()), lr=a.lr_head))
        if a.variant != "V1":
            enc_lr = 1e-4 if a.variant == "V2" else 2e-5
            groups.append(dict(params=[p for p in self.enc.model.parameters()
                                       if p.requires_grad], lr=enc_lr))
        self.opt = torch.optim.AdamW(groups, weight_decay=0.01)
        self.scaler = precision.grad_scaler()
        lg.info(f"autocast dtype {precision.autocast_dtype()}, loss scaling "
                f"{'on' if self.scaler.is_enabled() else 'off'}")

        # a live-PID marker so an orchestrator restart never launches this run twice
        self.ckdir = paths.CKPT / a.name
        self.ckdir.mkdir(parents=True, exist_ok=True)
        (self.ckdir / "RUNNING").write_text(str(os.getpid()))

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
    def refresh_entries(self, step, frac=None, over_all=False):
        """V2: re-encode entry contexts with the current encoder (§D.3).

        During training only *seen* entries are refreshed. At save time the whole
        vocabulary is re-encoded once, because §D.4 defines insertion for V2 as
        "a re-encoding with the trained LoRA encoder" -- so held-out rows must come
        from the final encoder, exactly as a genuinely new entry would.
        """
        a = self.a
        frac = a.refresh_frac if frac is None else frac
        pool = np.arange(self.nV) if over_all else np.flatnonzero(self.seen)
        g = np.random.default_rng(a.seed * 1000 + step)
        n = len(pool) if frac >= 1.0 else int(frac * len(pool))
        pick = pool if frac >= 1.0 else g.choice(pool, size=n, replace=False)
        occ = np.load(paths.vocab_file(a.vocab_tag))["occ_pid"][pick, : a.refresh_k]
        pids = np.unique(occ.reshape(-1))
        pids = pids[pids >= 0]
        inv = {self.words[j]: j for j in pick}
        # Only the entry's *own* sampled contexts may contribute, exactly as the
        # initial prototype build does. Accumulating every occurrence of the word in
        # the gathered passages instead pulls frequent entries towards the corpus mean
        # and turns them into hubs.
        key = np.sort(occ.reshape(-1).astype(np.int64) * self.nV
                      + np.repeat(pick.astype(np.int64), occ.shape[1]))
        key_t = torch.from_numpy(key).to(DEV)
        acc = torch.zeros(self.nV, 768, device=DEV)
        cnt = torch.zeros(self.nV, device=DEV)
        order = np.argsort([self.col.off[p + 1] - self.col.off[p] for p in pids])
        was = self.enc.model.training
        self.enc.model.eval()
        with torch.no_grad():
            for st in range(0, len(order), 384):
                sel = pids[order[st:st + 384]]
                wu = self.enc.encode_word_units(self.col.texts(sel), [a.layer],
                                                keep=lambda w: w in inv)
                if len(wu.word) == 0:
                    continue
                j = torch.tensor([inv[w.lower()] for w in wu.word], device=DEV,
                                 dtype=torch.int64)
                pp = torch.from_numpy(sel.astype(np.int64)).to(DEV)[
                    torch.from_numpy(wu.row.astype(np.int64)).to(DEV)]
                q = pp * self.nV + j
                pos = torch.searchsorted(key_t, q).clamp(max=len(key_t) - 1)
                hit = key_t[pos] == q
                if not hit.any():
                    continue
                acc.index_add_(0, j[hit], wu.states[hit, 0].float())
                cnt.index_add_(0, j[hit], torch.ones(int(hit.sum()), device=DEV))
        if was:
            self.enc.model.train()
        upd = cnt > 0
        V = acc[upd] / cnt[upd].unsqueeze(1)
        V = (V - torch.as_tensor(self.muV, device=DEV)) @ torch.as_tensor(self.WV, device=DEV)
        V = F.normalize(V, dim=-1)
        self.last_drift = float(F.cosine_similarity(self.E_all[upd], V, dim=-1).mean())
        self.E_all[upd] = V
        self.E_seen = self.E_all[self.seen_idx]
        return int(upd.sum())

    # ---------------------------------------------------------------- entry normalisation
    def refresh_entry_norm(self, chunk_rows=2048, chunk_ent=15000):
        """Per-entry background statistics of cos(g(h), v_j) over the document-side token
        bank, turned into the affine of dvlsr.model.entry_norm_affine.

        The targets mu*, sd* are the medians over *seen* entries of the same frequency
        decile, so the frequency structure of activation survives while each entry's own
        geometric idiosyncrasy is removed. Decile is known from corpus frequency at
        insertion time and the statistics are one matmul against a frozen bank, so a new
        entry is normalised exactly like a trained one, with no fitting.

        g moves during training, so this is recomputed every --norm-every steps and once
        more before the checkpoint is written.
        """
        a = self.a
        if a.entry_norm == "none":
            return None
        if self._bank is None:
            bank = np.load(paths.ART / f"bankH_{a.encoder}.npy", mmap_mode="r")
            li = paths.layers_for(a.encoder).index(a.layer)
            g = np.random.default_rng(a.seed)
            sel = np.sort(g.choice(bank.shape[0], size=min(a.norm_bank, bank.shape[0]),
                                   replace=False))
            self._bank = torch.as_tensor(np.asarray(bank[sel, li], np.float32), device=DEV)
        nV = self.E_all.shape[0]
        s1 = torch.zeros(nV, device=DEV, dtype=torch.float64)
        s2 = torch.zeros(nV, device=DEV, dtype=torch.float64)
        # the upper tail is what max-pooling selects, so keep a running top-m per entry
        mtop = max(int((1 - a.norm_q) * self._bank.shape[0]), 8)
        top = torch.full((nV, mtop), -2.0, device=DEV)
        n = 0
        was = self.head.training
        self.head.eval()
        with torch.no_grad():
            for r in range(0, self._bank.shape[0], chunk_rows):
                h = self.tfH(self._bank[r:r + chunk_rows], normalize=False)
                gh = F.normalize(self.head(h), dim=-1)
                n += gh.shape[0]
                for e in range(0, nV, chunk_ent):
                    aa = (gh @ self.E_all[e:e + chunk_ent].T).float()
                    s1[e:e + chunk_ent] += aa.sum(0).double()
                    s2[e:e + chunk_ent] += (aa * aa).sum(0).double()
                    cat = torch.cat([top[e:e + chunk_ent], aa.T], 1)
                    top[e:e + chunk_ent] = torch.topk(cat, mtop, dim=1).values
                    del aa, cat
        if was:
            self.head.train()
        mu = (s1 / n).float()
        sd = (s2 / n - (s1 / n) ** 2).clamp(min=1e-12).sqrt().float()
        qv = top[:, -1].contiguous()                  # the norm_q quantile of the background
        dec = torch.as_tensor(self.decile, device=DEV)
        seen_t = torch.as_tensor(self.seen, device=DEV)
        mu_star, sd_star, q_star = (torch.empty_like(mu), torch.empty_like(sd),
                                    torch.empty_like(qv))
        for d in range(10):
            m = dec == d
            if not bool(m.any()):
                continue
            sel = m & seen_t
            if not bool(sel.any()):        # whole decile held out (the rare split)
                sel = seen_t
            mu_star[m] = mu[sel].median()
            sd_star[m] = sd[sel].median()
            q_star[m] = qv[sel].median()
        alpha = a.norm_alpha
        if alpha != 1.0:      # shrink the target toward the entry's own statistics
            mu_star = mu + alpha * (mu_star - mu)
            sd_star = sd + alpha * (sd_star - sd)
            q_star = qv + alpha * (q_star - qv)
        if a.entry_norm == "z":
            # match the first two moments (scripts/48_entry_stats.py shows sd carries
            # almost no information about the firing rate; kept as the contrast)
            self.AB = entry_norm_affine(mu, sd, mu_star, sd_star)
        else:
            # match the upper tail: a pure shift, since it is the tail that the threshold
            # cuts and the tail that predicts the realised activation rate
            self.AB = torch.stack([torch.ones_like(qv), q_star - qv])
        self.AB_seen = self.AB[:, self.seen_idx]
        h_ = self.held
        g_ = lambda v: float(v[h_].median() - v[torch.as_tensor(self.seen, device=DEV)].median()) \
            if h_.any() else float("nan")
        return dict(mode=a.entry_norm, A_mean=float(self.AB[0].mean()),
                    B_held=float(self.AB[1][h_].mean()) if h_.any() else float("nan"),
                    mu_gap=g_(mu), sd_gap=g_(sd), q_gap=g_(qv))

    # ---------------------------------------------------------------- step
    def encode(self, texts, qside, grad):
        """Whitened word-unit states (+ pooled teacher embeddings when distilling).

        The frozen arm encodes in chunks: the batch of 1,024 passages does not need
        to sit in memory at once when no gradient flows through the encoder."""
        a = self.a
        want_pooled = a.teacher == "dense"
        layer, maxlen = a.layer, (paths.MAXLEN_QRY if qside else paths.MAXLEN_DOC)
        C = len(texts) if (grad or a.enc_chunk <= 0) else a.enc_chunk
        S, R, P = [], [], []
        for s in range(0, len(texts), C):
            wu = self.enc.encode_word_units(texts[s:s + C], [layer], is_query=qside,
                                            maxlen=maxlen, grad=grad, pooled=want_pooled)
            S.append(wu.states[:, 0])
            R.append(wu.row.astype(np.int64) + s)
            if want_pooled:
                P.append(wu.pooled)
        states = torch.cat(S) if len(S) > 1 else S[0]
        rows = np.concatenate(R) if len(R) > 1 else R[0]
        tf = self.tfQ if qside else self.tfH
        h = tf(states.to(DEV), normalize=False)
        pooled = (torch.cat(P) if len(P) > 1 else P[0]) if want_pooled else None
        return h, torch.as_tensor(rows, device=DEV), pooled

    def entries(self):
        """The entry matrix the score sees: frozen text-defined rows, the trained ones
        under --entry-param, or the text-defined rows passed through the shared
        entry-side map under --entry-head."""
        E = self.E_seen if self.E_param is None else F.normalize(self.E_param, dim=-1)
        return E if self.E_head is None else self.E_head(E)

    def rep(self, h, row, n, mask):
        """sparse_rep over text chunks under activation checkpointing.

        The (units x entries) logit block is the whole memory budget of a step; autograd
        keeps two copies of it and needs two more in backward. Recomputing it per chunk
        of --rep-chunk texts bounds the peak at one chunk's block (24 GB cards), with
        identical arithmetic."""
        C = self.a.rep_chunk
        if C <= 0 or n <= C or not torch.is_grad_enabled():
            return sparse_rep(self.head, h, row, n, self.entries(), mask, self.AB_seen)
        starts = list(range(0, n, C)) + [n]
        pos = torch.searchsorted(row, torch.tensor(starts, device=row.device)).tolist()
        outs = []
        for k in range(len(starts) - 1):
            a, b, t0, t1 = pos[k], pos[k + 1], starts[k], starts[k + 1]

            def f(hc, rc, t0=t0, t1=t1):
                return sparse_rep(self.head, hc, rc - t0, t1 - t0, self.entries(), mask,
                                  self.AB_seen)
            outs.append(checkpoint(f, h[a:b], row[a:b], use_reentrant=False))
        return torch.cat(outs, 0)

    def step_mask(self, step):
        """Entries in this step's ranking objective (bool over seen entries), and the
        FLOPS rescale (§D.3: 1/(1-p), generalised to the realised kept fraction)."""
        a = self.a
        keep = self.trainable.clone()
        if a.vd:
            gg = torch.Generator(device=DEV)
            gg.manual_seed(a.seed * 99991 + step)
            if a.vd_mode == "entry":
                drop = torch.rand(self.n_seen, device=DEV, generator=gg) < a.vd_rate
            else:
                drop = torch.zeros(self.n_seen, dtype=torch.bool, device=DEV)
                target = a.vd_rate * int(keep.sum())
                n_drop = 0
                for c in torch.randperm(a.vd_clusters, device=DEV, generator=gg).tolist():
                    if n_drop >= target:
                        break
                    sel = (self.vd_labels == c) & keep
                    drop |= sel
                    n_drop += int(sel.sum())
            keep &= ~drop
        lam_scale = float(self.n_seen) / max(int(keep.sum()), 1)
        return keep, lam_scale

    def run(self):
        a = self.a
        grad_enc = a.variant != "V1"
        sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / a.warmup) *
            max(0.0, 1 - max(0, s - a.warmup) / max(1, self.steps - a.warmup)))
        step = 0
        t0 = time.time()
        hist = []
        if a.entry_norm != "none":
            with Timer(f"entry-norm statistics over {a.norm_bank} bank tokens", lg):
                st = self.refresh_entry_norm()
            lg.info(f"[{a.name}] entry-norm at step 0: {st}")
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

                keep, lam_scale = self.step_mask(step)
                full_needed = self.meta is not None and a.meta_weight > 0
                mask = None if bool(keep.all()) else keep

                with precision.autocast():
                    hq, rq, pq = self.encode(qtexts, True, grad_enc)
                    hd, rd, pd = self.encode(dtexts, False, grad_enc)
                    if full_needed:
                        Sq = self.rep(hq, rq, len(qtexts), None)
                        Sd = self.rep(hd, rd, len(dtexts), None)
                        sq, sd = Sq[:, keep], Sd[:, keep]
                    else:
                        sq = self.rep(hq, rq, len(qtexts), mask)
                        sd = self.rep(hd, rd, len(dtexts), mask)
                    scores = sq @ sd.T
                    ce = F.cross_entropy(scores.float(), pos_idx)
                    ramp = min(1.0, ((step + 1) / max(self.ramp, 1)) ** 2)
                    lq = a.lam_q * ramp * lam_scale
                    ld = a.lam_d * ramp * lam_scale
                    fq, fd = flops(sq.float()), flops(sd.float())
                    loss = ce + lq * fq + ld * fd
                    cal = kl = torch.zeros((), device=DEV)
                    if full_needed:
                        cal = calibration_loss(Sd.float(), self.meta, keep, self.seen_decile)
                        loss = loss + a.meta_weight * cal
                    if a.teacher == "dense":
                        St = (pq @ pd.T).float() / a.teacher_temp
                        kl = F.kl_div(F.log_softmax(scores.float(), 1), F.softmax(St, 1),
                                      reduction="batchmean")
                        loss = loss + a.distill_weight * kl

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for gp in self.opt.param_groups for p in gp["params"]], 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                sched.step()
                step += 1

                if step % 50 == 0:
                    with torch.no_grad():
                        nq = float((sq > 0).sum(1).float().mean())
                        nd = float((sd > 0).sum(1).float().mean())
                        acc = float((scores.argmax(1) == pos_idx).float().mean())
                        reg = float((lq * fq + ld * fd) / max(float(ce), 1e-9))
                    rec = dict(step=step, ce=float(ce), nnz_q=nq, nnz_d=nd, acc=acc,
                               t=float(self.head.t), b=float(self.head.b),
                               flops_q=float(fq), flops_d=float(fd), reg_share=reg,
                               kept=int(keep.sum()), cal=float(cal), kl=float(kl))
                    hist.append(rec)
                    lg.info(f"[{a.name}] step {step}/{self.steps} ce={float(ce):.4f} "
                            f"acc={acc:.3f} nnz_q={nq:.0f} nnz_d={nd:.0f} "
                            f"t={float(self.head.t):.2f} b={float(self.head.b):.2f} "
                            f"reg/ce={reg:.2e} kept={int(keep.sum())}"
                            + (f" cal={float(cal):.4f}" if full_needed else "")
                            + (f" kl={float(kl):.4f}" if a.teacher != 'none' else "")
                            + f" ({(time.time()-t0)/step:.2f}s/step)")
                if a.max_steps and step >= a.max_steps:
                    self.save(hist); return
                if a.entry_norm != "none" and step % a.norm_every == 0:
                    st = self.refresh_entry_norm()
                    lg.info(f"[{a.name}] entry-norm refreshed at step {step}: {st}")
                if a.variant == "V2" and step % a.refresh_every == 0:
                    n = self.refresh_entries(step, frac=a.refresh_frac)
                    lg.info(f"[{a.name}] refreshed {n} entries at step {step} "
                            f"(mean cos to previous rows {self.last_drift:.4f})")
        self.save(hist)

    def save(self, hist):
        a = self.a
        if a.variant == "V2":
            with Timer("final full re-encode of the entry matrix (V2 insertion)", lg):
                n = self.refresh_entries(10 ** 9, frac=1.0, over_all=True)
            lg.info(f"[{a.name}] re-encoded {n} entries with the trained encoder")
        if a.entry_norm != "none":
            with Timer("final entry-norm statistics (the map insertion will use)", lg):
                st = self.refresh_entry_norm()
            lg.info(f"[{a.name}] final entry-norm: {st}")
        d = paths.CKPT / a.name
        d.mkdir(parents=True, exist_ok=True)
        E_out = self.E_all
        if self.E_param is not None:      # trained rows in place, held-out rows as inserted
            E_out = self.E_all.clone()
            E_out[self.seen_idx] = F.normalize(self.E_param.detach(), dim=-1)
        if self.E_head is not None:
            # apply the shared map to EVERY entry, held-out included: that application is
            # exactly what inserting an entry means under this variant
            with torch.no_grad():
                E_out = torch.cat([self.E_head(E_out[i:i + 8192])
                                   for i in range(0, E_out.shape[0], 8192)], 0)
        torch.save(dict(head=self.head.state_dict(),
                        muV=self.muV, WV=self.WV,
                        E_all=E_out.cpu(),
                        AB=None if self.AB is None else self.AB.cpu()), d / "head.pt")
        if a.variant == "V2":
            self.enc.model.save_pretrained(str(d / "lora"))
        elif a.variant == "V3":
            self.enc.model.save_pretrained(str(d / "encoder"))
        extra = {}
        if self.vd_labels is not None:
            extra["cluster_labels_seen"] = self.vd_labels.cpu().numpy()
        if self.meta is not None:
            extra["meta_seen"] = self.meta.cpu().numpy()
        if extra:
            np.savez(d / "clusters.npz", seen_idx=self.seen_idx.cpu().numpy(), **extra)
        save_json(dict(vars(a), n_seen=int(self.seen.sum()), n_held=int(self.held.sum()),
                       amp=str(precision.autocast_dtype())),
                  d / "config.json")
        save_json(hist, d / "history.json")
        (d / "RUNNING").unlink(missing_ok=True)
        lg.info(f"saved {d}")


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="V1", choices=["V1", "V2", "V3"])
    ap.add_argument("--split", default="random",
                    choices=["random", "cluster", "rare", "phrase", "none"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--crand", action="store_true")
    ap.add_argument("--vd", action="store_true")
    ap.add_argument("--vd-mode", default="entry", choices=["entry", "cluster"])
    ap.add_argument("--vd-rate", type=float, default=0.3)
    ap.add_argument("--vd-clusters", type=int, default=100)
    ap.add_argument("--meta-frac", type=float, default=0.0)
    ap.add_argument("--meta-weight", type=float, default=1.0)
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--teacher", default="none", choices=["none", "dense"])
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--teacher-temp", type=float, default=0.02)
    ap.add_argument("--vocab-tag", default="")
    ap.add_argument("--entry-tf", default="own", choices=["own", "shared"])
    ap.add_argument("--entry-norm", default="none", choices=["none", "z", "q"],
                    help="Pilot F: per-entry background normalisation inside the score. "
                         "q = match the upper tail (a shift); z = match mean and sd")
    ap.add_argument("--norm-every", type=int, default=250)
    ap.add_argument("--norm-bank", type=int, default=32768)
    ap.add_argument("--norm-q", type=float, default=0.999)
    ap.add_argument("--norm-alpha", type=float, default=1.0,
                    help="how far to move each entry toward its decile target (1 = fully)")
    ap.add_argument("--min-free-gb", type=float, default=9.0,
                    help="wait for this much free GPU memory before starting (0 disables)")
    ap.add_argument("--entry-head", action="store_true",
                    help="learn a shared residual map on the entry side (Pilot I)")
    ap.add_argument("--entry-param", action="store_true",
                    help="control: train the seen entry rows as free parameters")
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
    ap.add_argument("--enc-chunk", type=int, default=512)
    ap.add_argument("--rep-chunk", type=int, default=256)
    ap.add_argument("--grad-ckpt", type=int, default=1,
                    help="gradient checkpointing inside the encoder for V2/V3 (1=on)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refresh-every", type=int, default=3000)
    ap.add_argument("--refresh-frac", type=float, default=1.0)
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
