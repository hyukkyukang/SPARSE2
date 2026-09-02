"""Encoder wrappers: pooled embeddings and word-unit hidden states (§0.1, §0.4).

A *word unit* is a pre-tokenizer word; its state is the mean of its wordpiece
states. Special tokens, padding, prompt-prefix words and pure-punctuation units
are dropped (see the §Appendix pitfall about [CLS]/[SEP] hubs).
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass

from . import paths


@dataclass
class WordUnits:
    """Word units of one batch, flattened.

    row:    index of the source text within the batch
    word:   surface string (as it appeared, not lowercased)
    states: (n_units, n_layers, d) float16 — mean over the word's pieces
    """
    row: np.ndarray
    word: list[str]
    states: torch.Tensor


class Encoder:
    def __init__(self, name: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModel, AutoTokenizer
        cfg = paths.ENCODERS[name]
        self.name, self.cfg = name, cfg
        self.tok = AutoTokenizer.from_pretrained(cfg["hf"])
        self.model = AutoModel.from_pretrained(cfg["hf"], dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        self.dim = cfg["dim"]

    # ------------------------------------------------------------------ helpers
    def prefix(self, is_query: bool) -> str:
        return self.cfg["q_prefix"] if is_query else self.cfg["d_prefix"]

    def _tok(self, texts, is_query, maxlen, offsets=False):
        pre = self.prefix(is_query)
        full = [pre + t for t in texts] if pre else list(texts)
        enc = self.tok(full, padding=True, truncation=True, max_length=maxlen,
                       return_tensors="pt", return_offsets_mapping=offsets)
        return full, enc, len(pre)

    # ------------------------------------------------------------------ pooled
    @torch.no_grad()
    def encode_pooled(self, texts, is_query=False, maxlen=None, batch_size=256) -> np.ndarray:
        maxlen = maxlen or (paths.MAXLEN_QRY if is_query else paths.MAXLEN_DOC)
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            _, enc, _ = self._tok(chunk, is_query, maxlen)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.model(**enc).last_hidden_state
            if self.cfg["pooling"] == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out[i:i + len(chunk)] = v.to(torch.float16).cpu().numpy()
        return out

    # ------------------------------------------------------------------ word units
    def encode_word_units(self, texts, layers, is_query=False, maxlen=None,
                          keep=None, grad=False) -> WordUnits:
        """Word-unit states for one batch of texts at the requested hidden layers.

        keep: optional callable(word_lower) -> bool, applied before states are
              gathered so we never materialise units we do not need.
        grad: keep the graph (Pilot D's V2/V3, which train the encoder).
        """
        import contextlib
        with (contextlib.nullcontext() if grad else torch.no_grad()):
            return self._word_units(texts, layers, is_query, maxlen, keep)

    def _word_units(self, texts, layers, is_query, maxlen, keep) -> WordUnits:
        maxlen = maxlen or (paths.MAXLEN_QRY if is_query else paths.MAXLEN_DOC)
        full, enc, plen = self._tok(texts, is_query, maxlen, offsets=True)
        offs = enc.pop("offset_mapping").numpy()
        B, L = offs.shape[:2]
        wid = np.full((B, L), -1, dtype=np.int32)
        for b in range(B):
            w = enc.word_ids(b)
            wid[b] = [-1 if x is None else x for x in w]

        # group contiguous pieces of the same word into units
        flat_wid = wid.reshape(-1)
        row_of = np.repeat(np.arange(B), L)
        valid = flat_wid >= 0
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return WordUnits(np.zeros(0, np.int32), [], torch.zeros(0, len(layers), self.dim))
        w_here = flat_wid[idx]
        r_here = row_of[idx]
        new = np.empty(len(idx), dtype=bool)
        new[0] = True
        new[1:] = (w_here[1:] != w_here[:-1]) | (r_here[1:] != r_here[:-1])
        gid = np.cumsum(new) - 1                      # group id per valid token
        n_units = int(gid[-1]) + 1
        starts = np.flatnonzero(new)
        ends = np.r_[starts[1:], len(idx)]
        u_row = r_here[starts]
        c0 = offs.reshape(-1, 2)[idx, 0][starts]
        c1 = np.maximum.reduceat(offs.reshape(-1, 2)[idx, 1], starts)

        words = [full[int(r)][int(a):int(b)] for r, a, b in zip(u_row, c0, c1)]
        ok = np.array([(c0[i] >= plen) and any(ch.isalnum() for ch in words[i])
                       for i in range(n_units)], dtype=bool)
        if keep is not None:
            ok &= np.array([keep(w.lower()) if o else False
                            for w, o in zip(words, ok)], dtype=bool)
        if not ok.any():
            return WordUnits(np.zeros(0, np.int32), [], torch.zeros(0, len(layers), self.dim))

        enc = {k: v.to(self.device) for k, v in enc.items()}
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        gid_t = torch.from_numpy(gid.astype(np.int64)).to(self.device)
        idx_t = torch.from_numpy(idx.astype(np.int64)).to(self.device)
        cnt = torch.zeros(n_units, device=self.device, dtype=torch.float32)
        cnt.index_add_(0, gid_t, torch.ones_like(gid_t, dtype=torch.float32))
        keep_t = torch.from_numpy(np.flatnonzero(ok)).to(self.device)
        st = torch.empty(int(ok.sum()), len(layers), self.dim,
                         device=self.device, dtype=torch.float16)
        for li, l in enumerate(layers):
            h = hs[l].reshape(-1, self.dim).index_select(0, idx_t).float()
            acc = torch.zeros(n_units, self.dim, device=self.device, dtype=torch.float32)
            acc.index_add_(0, gid_t, h)
            acc /= cnt.unsqueeze(1)
            st[:, li] = acc.index_select(0, keep_t).to(torch.float16)
        sel = np.flatnonzero(ok)
        return WordUnits(u_row[sel].astype(np.int32), [words[i] for i in sel], st)


class SpladeEncoder:
    """SPLADE++ (CoCondenser-EnsembleDistil) — used for expansion terms and as the LSR reference."""

    def __init__(self, device="cuda", dtype=torch.float16):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(paths.SPLADE)
        self.model = AutoModelForMaskedLM.from_pretrained(paths.SPLADE, dtype=dtype).to(device).eval()
        self.device = device
        self.vocab = self.tok.convert_ids_to_tokens(list(range(self.tok.vocab_size)))

    @torch.no_grad()
    def encode(self, texts, maxlen=paths.MAXLEN_DOC, batch_size=128):
        """Returns list of (indices, weights) sparse vectors (log(1+relu) max-pooled)."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                           max_length=maxlen, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            w = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            v = w.max(dim=1).values.float()
            for r in v:
                nz = torch.nonzero(r, as_tuple=True)[0]
                out.append((nz.cpu().numpy().astype(np.int32),
                            r[nz].cpu().numpy().astype(np.float32)))
        return out


class ColbertEncoder(Encoder):
    """ColBERTv2 token embeddings (128-d, L2-normalised) -- Pilot A's reference ceiling.

    Presented through the same interface as the candidates: one "layer" (the final
    one, after the 128-d projection), [Q]/[D] marker tokens as in the original, and
    a pooled object for R1 (the mean of the word's own normalised token embeddings).
    """

    def __init__(self, name: str = "colbert", device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModel, AutoTokenizer
        import torch.nn as nn
        cfg = paths.ENCODERS["colbert"]
        self.name, self.cfg = name, cfg
        self.tok = AutoTokenizer.from_pretrained(cfg["hf"])
        self.model = AutoModel.from_pretrained(cfg["hf"], dtype=dtype).to(device).eval()
        self.device, self.dtype, self.dim = device, dtype, cfg["dim"]
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st
        try:
            sd = st.load_file(hf_hub_download(cfg["hf"], "model.safetensors"))
        except Exception:
            sd = torch.load(hf_hub_download(cfg["hf"], "pytorch_model.bin"),
                            map_location="cpu")
        key = [k for k in sd if k.endswith("linear.weight")][0]
        self.proj = nn.Linear(768, 128, bias=False).to(device).to(dtype)
        self.proj.weight.data.copy_(sd[key].to(dtype))

    def _hidden(self, enc):
        h = self.model(**enc).last_hidden_state
        return torch.nn.functional.normalize(self.proj(h), dim=-1)

    def _word_units(self, texts, layers, is_query, maxlen, keep) -> WordUnits:
        maxlen = maxlen or (paths.MAXLEN_QRY if is_query else paths.MAXLEN_DOC)
        full, enc, plen = self._tok(texts, is_query, maxlen, offsets=True)
        offs = enc.pop("offset_mapping").numpy()
        B, L = offs.shape[:2]
        wid = np.full((B, L), -1, dtype=np.int32)
        for b in range(B):
            w = enc.word_ids(b)
            wid[b] = [-1 if x is None else x for x in w]
        flat = wid.reshape(-1)
        row_of = np.repeat(np.arange(B), L)
        idx = np.flatnonzero(flat >= 0)
        if len(idx) == 0:
            return WordUnits(np.zeros(0, np.int32), [], torch.zeros(0, 1, self.dim))
        w_here, r_here = flat[idx], row_of[idx]
        new = np.empty(len(idx), bool); new[0] = True
        new[1:] = (w_here[1:] != w_here[:-1]) | (r_here[1:] != r_here[:-1])
        gid = np.cumsum(new) - 1
        n_units = int(gid[-1]) + 1
        starts = np.flatnonzero(new)
        u_row = r_here[starts]
        c0 = offs.reshape(-1, 2)[idx, 0][starts]
        c1 = np.maximum.reduceat(offs.reshape(-1, 2)[idx, 1], starts)
        words = [full[int(r)][int(a):int(b)] for r, a, b in zip(u_row, c0, c1)]
        ok = np.array([(c0[i] >= plen) and any(ch.isalnum() for ch in words[i])
                       for i in range(n_units)], bool)
        if keep is not None:
            ok &= np.array([keep(w.lower()) if o else False
                            for w, o in zip(words, ok)], bool)
        if not ok.any():
            return WordUnits(np.zeros(0, np.int32), [], torch.zeros(0, 1, self.dim))
        enc = {k: v.to(self.device) for k, v in enc.items()}
        h = self._hidden(enc).reshape(-1, self.dim)
        gid_t = torch.from_numpy(gid.astype(np.int64)).to(self.device)
        idx_t = torch.from_numpy(idx.astype(np.int64)).to(self.device)
        cnt = torch.zeros(n_units, device=self.device)
        cnt.index_add_(0, gid_t, torch.ones_like(gid_t, dtype=torch.float32))
        acc = torch.zeros(n_units, self.dim, device=self.device)
        acc.index_add_(0, gid_t, h.index_select(0, idx_t).float())
        acc /= cnt.unsqueeze(1)
        sel = np.flatnonzero(ok)
        st = acc[torch.from_numpy(sel).to(self.device)].unsqueeze(1).to(torch.float16)
        return WordUnits(u_row[sel].astype(np.int32), [words[i] for i in sel], st)

    @torch.no_grad()
    def encode_pooled(self, texts, is_query=False, maxlen=None, batch_size=256):
        """R1 for ColBERT: the mean of the word's own normalised token embeddings."""
        maxlen = maxlen or (paths.MAXLEN_QRY if is_query else paths.MAXLEN_DOC)
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            _, enc, plen = self._tok(chunk, is_query, maxlen)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self._hidden(enc)
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out[i:i + len(chunk)] = v.to(torch.float16).cpu().numpy()
        return out


def get_encoder(name, **kw):
    return ColbertEncoder(name, **kw) if name == "colbert" else Encoder(name, **kw)
