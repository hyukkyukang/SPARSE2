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
    pooled: torch.Tensor | None = None      # (B, d) L2-normalised pooled embedding, if requested


def _trim(w: str) -> str:
    """Strip whitespace and any leading/trailing non-alphanumerics from a unit surface."""
    i, j = 0, len(w)
    while i < j and not w[i].isalnum():
        i += 1
    while j > i and not w[j - 1].isalnum():
        j -= 1
    return w[i:j]


class Encoder:
    def __init__(self, name: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModel, AutoTokenizer
        cfg = paths.ENCODERS[name]
        self.name, self.cfg = name, cfg
        trc = bool(cfg.get("trust_remote_code", False))
        self.tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=trc)
        self.model = AutoModel.from_pretrained(cfg["hf"], dtype=dtype,
                                               trust_remote_code=trc).to(device).eval()
        if cfg.get("adapter"):
            # jina-embeddings-v5 is a PEFT model with one LoRA adapter per task
            self.model.set_adapter([cfg["adapter"]])
            if cfg.get("merge_adapter"):
                # fold the active adapter into the weights: outputs identical (pooled cos
                # 1.0000, unit states 0.9999 in fp16) and no adapter matmuls -> 1.8x faster
                self.model = self.model.merge_and_unload().eval()
        self.device, self.dtype = device, dtype
        self.dim = cfg["dim"]
        # decoders keep a KV cache by default; we never generate, so never allocate one
        self._fwd_kw = {"use_cache": False} if cfg.get("pooling") == "last" else {}
        self.truncated = None

    # ------------------------------------------------------------------ layer surgery
    def _decoder_stack(self):
        """(parent module, its ModuleList of transformer layers) or (None, None).
        Qwen3: model.layers; under PEFT: base_model.model.layers; BERT names its list
        'layer', which we leave alone -- e5 runs at its final layer anyway."""
        import torch.nn as nn
        for n, m in self.model.named_modules():
            if n.split(".")[-1] == "layers" and isinstance(m, nn.ModuleList) and len(m):
                parent = self.model.get_submodule(n.rsplit(".", 1)[0]) if "." in n else self.model
                return parent, m
        return None, None

    def truncate_to(self, L: int) -> bool:
        """Drop every transformer layer above L.

        hidden_states[L] is a function of layers 1..L only, so when a single layer is
        all that is needed this is exact and saves (n-L)/n of the forward pass. The stack's
        final norm is replaced by the identity, because HF applies it to the *last* entry
        of hidden_states and the artifacts (prototypes, banks, whitening) were built from
        the un-normed layer-L state of the full model. Pooled embeddings are unavailable
        afterwards; callers that need them must not truncate."""
        import torch.nn as nn
        parent, layers = self._decoder_stack()
        if layers is None or L >= len(layers):
            return False
        del layers[L:]
        if hasattr(parent, "norm"):
            parent.norm = nn.Identity()
        for m in (self.model, parent):
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "num_hidden_layers"):
                c.num_hidden_layers = L
        self.truncated = L
        return True

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
        assert self.truncated is None, "pooled embeddings need the full stack"
        maxlen = maxlen or (paths.MAXLEN_QRY if is_query else paths.MAXLEN_DOC)
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            _, enc, _ = self._tok(chunk, is_query, maxlen)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.model(**enc, **self._fwd_kw).last_hidden_state
            v = self._pool(h, enc)
            out[i:i + len(chunk)] = v.to(torch.float16).cpu().numpy()
        return out

    # ------------------------------------------------------------------ word units
    def encode_word_units(self, texts, layers, is_query=False, maxlen=None,
                          keep=None, grad=False, pooled=False) -> WordUnits:
        """Word-unit states for one batch of texts at the requested hidden layers.

        keep:   optional callable(word_lower) -> bool, applied before states are
                gathered so we never materialise units we do not need.
        grad:   keep the graph (Pilot D's V2/V3, which train the encoder).
        pooled: also return the encoder's own pooled embedding (its dense retrieval
                vector) for every text, computed from the same forward pass -- the
                teacher for the distillation arm.
        """
        import contextlib
        with (contextlib.nullcontext() if grad else torch.no_grad()):
            return self._word_units(texts, layers, is_query, maxlen, keep, pooled)

    def _pool(self, hs_last, enc):
        pooling = self.cfg["pooling"]
        if pooling == "cls":
            v = hs_last[:, 0]
        elif pooling == "last":
            # right padding: the last real token, which for Octen is the <|endoftext|>
            # its tokenizer appends and for jina-v5 the final text token -- exactly what
            # each model's own encode() pools
            idx = enc["attention_mask"].sum(1) - 1
            v = hs_last[torch.arange(hs_last.shape[0], device=hs_last.device), idx]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(hs_last.dtype)
            v = (hs_last * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(v.float(), dim=-1)

    def _word_units(self, texts, layers, is_query, maxlen, keep, pooled=False) -> WordUnits:
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

        # Prompt handling at the PIECE level. Byte-level BPE offsets include the space that
        # precedes a word, so the first text word starts one character inside the prompt
        # ("Document: The" -> "ĠThe" spans (9,13), plen=10); and Qwen's pre-tokenizer glues
        # a leading punctuation mark onto the next word, so "Query:what" is ONE word whose
        # first piece ":" belongs to the prompt. A piece is text iff it ends after the
        # prompt; a unit is text iff it has such a piece; its state averages only those
        # pieces and its surface starts at the prompt boundary, then is stripped. For
        # WordPiece (offsets exclude spaces; no piece straddles the boundary) this is
        # identical to the old rule -- e5 units and states are bit-identical (D15).
        # Qwen's pre-tokenizer also glues ONE leading non-letter onto a word ("brown-fox"
        # -> "brown", "-fox"; ",leading" -> ",leading"), so a unit's surface is trimmed of
        # non-alphanumerics at both ends for the vocabulary lookup and pieces that carry no
        # alphanumeric character do not enter the unit's mean. WordPiece pre-tokens are
        # either all-punctuation or punctuation-free, so e5 is unaffected (D15).
        # Gated on the encoder (`trim_punct`): BERT's tokenizer keeps symbol characters such
        # as ° or € inside a word, which str.isalnum rejects, so applying this to e5 would
        # alter 0.1% of its units and break reproducibility of the frozen e5 artifacts.
        o2 = offs.reshape(-1, 2)[idx]
        piece_ok = o2[:, 1] > plen
        if self.cfg.get("trim_punct"):
            piece_txt = [full[int(r)][int(a):int(b)] for r, (a, b) in zip(r_here, o2)]
            piece_ok &= np.array([any(ch.isalnum() for ch in t) for t in piece_txt], dtype=bool)
            words = [_trim(full[int(r)][max(int(a), plen):int(b)])
                     for r, a, b in zip(u_row, c0, c1)]
        else:
            words = [full[int(r)][max(int(a), plen):int(b)].strip()
                     for r, a, b in zip(u_row, c0, c1)]
        has_piece = np.zeros(n_units, dtype=bool)
        has_piece[gid[piece_ok]] = True
        ok = has_piece & np.array([any(ch.isalnum() for ch in w) for w in words], dtype=bool)
        if keep is not None:
            ok &= np.array([keep(w.lower()) if o else False
                            for w, o in zip(words, ok)], dtype=bool)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if not ok.any():
            pv = None
            if pooled:
                assert self.truncated is None, "pooled embeddings need the full stack"
                pv = self._pool(self.model(**enc, **self._fwd_kw).last_hidden_state, enc)
            return WordUnits(np.zeros(0, np.int32), [], torch.zeros(0, len(layers), self.dim), pv)

        if pooled:
            assert self.truncated is None, "pooled embeddings need the full stack"
        hs = self.model(**enc, output_hidden_states=True, **self._fwd_kw).hidden_states
        pv = self._pool(hs[-1], enc) if pooled else None
        # only pieces outside the prompt contribute to a unit's mean
        gid_t = torch.from_numpy(gid[piece_ok].astype(np.int64)).to(self.device)
        idx_t = torch.from_numpy(idx[piece_ok].astype(np.int64)).to(self.device)
        cnt = torch.zeros(n_units, device=self.device, dtype=torch.float32)
        cnt.index_add_(0, gid_t, torch.ones_like(gid_t, dtype=torch.float32))
        cnt.clamp_(min=1.0)                    # units without a contributing piece are never selected
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
        return WordUnits(u_row[sel].astype(np.int32), [words[i] for i in sel], st, pv)


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

    def _word_units(self, texts, layers, is_query, maxlen, keep, pooled=False) -> WordUnits:
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
        words = [full[int(r)][int(a):int(b)].strip() for r, a, b in zip(u_row, c0, c1)]
        ok = np.array([(c1[i] > plen) and any(ch.isalnum() for ch in words[i])
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
