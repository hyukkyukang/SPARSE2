"""Domain-shift evaluation, step 2: the vocabulary a live deployment would actually add.

The entries are the words the *target* corpus uses that the trained vocabulary does not
have. That is the real version of the cluster split: not a k-means region carved out of
MS MARCO, but terminology the model has genuinely never had a dimension for.

Selection, deliberately simple so it cannot be tuned into a result:
  * lowercase alphabetic words in the target corpus, >= --min-occ occurrences there
  * NOT already in the trained 30k vocabulary
  * top --n-add by target-corpus frequency

Prototypes are built the same way as everywhere else (§A.1): the mean word-unit state over
up to k occurrences, here drawn from the *target* corpus, because that is where the term
lives. Decile is assigned by target-corpus frequency so the insertion-time calibration of
Pilot F has the frequency band it needs.

Writes an extended vocabulary and prototype file under the tag `dom_<name>`, so every
downstream stage takes --vocab-tag dom_<name> and behaves exactly as it does for words.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.encoders import get_encoder
from dvlsr.util import get_logger, save_json, Timer, rng

lg = get_logger("domvocab", "91_domain_vocab.log")
WORD_RE = re.compile(r"[A-Za-z0-9]+")


class DomainCollection:
    """Same interface as dvlsr.data.Collection, over a domain corpus."""

    def __init__(self, name):
        d = paths.DATA / "domain" / name
        self.off = np.load(d / "offsets.npy", mmap_mode="r")
        self._buf = np.memmap(d / "passages.bin", dtype=np.uint8, mode="r")
        self.ids = json.load(open(d / "ids.json"))
        self.n = len(self.off) - 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a, b = self.off[i], self.off[i + 1]
        return self._buf[a:b].tobytes().decode("utf-8", "replace")

    def texts(self, idx):
        return [self[int(i)] for i in idx]


def main(a):
    col = DomainCollection(a.name)
    base = np.load(paths.vocab_file(""))
    known = {str(w) for w in base["words"]}

    cnt = Counter()      # document frequency: passages containing the word
    tot = Counter()      # term frequency: total occurrences
    occ = {}
    for pid in range(len(col)):
        seen = set()
        for w in WORD_RE.findall(col[pid]):
            if not w.isalpha():
                continue
            wl = w.lower()
            if wl in known:
                continue
            tot[wl] += 1
            if wl in seen:
                continue
            seen.add(wl)
            cnt[wl] += 1
            occ.setdefault(wl, []).append(pid)
    N = len(col)
    lg.info(f"{a.name}: {len(cnt):,} word types absent from the trained vocabulary")

    # How entries are CHOSEN. Ranking by raw document frequency is blind to whether a term
    # discriminates *within* the corpus, so on a single-topic collection it selects the
    # corpus-defining terms first -- which is how covid/coronavirus/wuhan became the top
    # trec-covid entries, and why insertion there cost 0.257 MRR@10.
    pool = [w for w, c in cnt.items() if c >= a.min_occ]
    if a.select == "df":
        score = {w: cnt[w] for w in pool}
    elif a.select == "tfidf":
        # tf x idf: total occurrences weighted by how *few* documents contain the term, so
        # a word in most of the corpus is penalised however often it occurs
        score = {w: tot[w] * np.log(N / cnt[w]) for w in pool}
    elif a.select == "idf":
        # the heaviest possible IDF weighting: rank purely by rarity among the candidates
        # that clear the occurrence floor (tests whether lexically rarer terms are safer)
        score = {w: np.log(N / cnt[w]) for w in pool}
    elif a.select == "dfcap":
        # document frequency, but nothing above a share of the corpus: the per-corpus
        # analogue of a stopword list, computable at insertion time with no labels
        pool = [w for w in pool if cnt[w] <= a.max_df * N]
        score = {w: cnt[w] for w in pool}
    else:
        raise SystemExit(f"unknown --select {a.select}")
    cand = sorted(pool, key=lambda w: (-score[w], w))
    words = cand[: a.n_add]
    freq = np.asarray([cnt[w] for w in words], np.int64)
    share = freq / N
    lg.info(f"{a.name}/{a.select}: {len(words)} entries | document share "
            f"min {share.min():.4f} median {np.median(share):.4f} max {share.max():.4f} | "
            f"examples: {', '.join(words[:25])}")

    # prototypes from the target corpus, k occurrences per entry (§A.1)
    enc = get_encoder(a.encoder)
    L = paths.layers_for(a.encoder)
    inv = {w: i for i, w in enumerate(words)}
    g = rng("domain-proto", a.name)
    pick = {w: (occ[w] if len(occ[w]) <= a.k else
                list(g.choice(occ[w], size=a.k, replace=False))) for w in words}
    want = {}
    for w, pids in pick.items():
        for p in pids:
            want.setdefault(int(p), set()).add(w)
    acc = torch.zeros(len(words), len(L), enc.dim, device="cuda")
    n = torch.zeros(len(words), device="cuda")
    pids = sorted(want)
    with Timer(f"{a.name}: prototypes for {len(words)} entries over {len(pids)} passages", lg):
        for s in range(0, len(pids), a.bs):
            sel = pids[s:s + a.bs]
            wu = enc.encode_word_units(col.texts(sel), L, is_query=False,
                                       keep=lambda w: w in inv)
            if len(wu.word) == 0:
                continue
            for u, w in enumerate(wu.word):
                wl = w.lower()
                p = sel[wu.row[u]]
                if wl in inv and wl in want.get(p, ()):
                    j = inv[wl]
                    acc[j] += wu.states[u].float()
                    n[j] += 1
    ok = (n > 0).cpu().numpy()
    P = (acc / n.clamp(min=1).unsqueeze(1).unsqueeze(2)).cpu().numpy()
    lg.info(f"{a.name}: entries with no prototype: {int((~ok).sum())}; "
            f"median occurrences used {float(n.median()):.0f}")

    # Decile must be comparable ACROSS corpora, or the per-decile calibration of Pilot F
    # compares a scifact term seen 5 times in 5k passages against an MS MARCO term seen 500
    # times in 8.8M -- which is 965 vs 56 occurrences per million, i.e. the domain term is
    # relatively COMMON, not rare. Map by rate, not by raw count.
    base_rate = base["freq"] / paths.N_PASSAGES * 1e6
    dom_rate = freq / len(col) * 1e6
    edges = np.array([base_rate[base["decile"] == d].min() for d in range(10)])   # descending
    dec_new = np.full(len(words), 9, np.int8)
    for d in range(10):
        dec_new[dom_rate >= edges[d]] = np.minimum(dec_new[dom_rate >= edges[d]], d)
    lg.info(f"{a.name}: occurrences per million {dom_rate.min():.0f}–{dom_rate.max():.0f} "
            f"(trained vocabulary spans {base_rate.min():.0f}–{base_rate.max():.0f}); "
            f"inserted entries by decile: "
            f"{ {int(d): int((dec_new == d).sum()) for d in range(10) if (dec_new == d).any()} }")

    tag = f"dom_{a.name}" + ("" if a.select == "df" else f"_{a.select}")
    nb = len(base["words"])
    # The word list depends only on the corpus and the base vocabulary, so the vocab and
    # split files are shared across encoders and written once; a second encoder must
    # reproduce the same list exactly. Prototypes are per encoder and always written.
    if paths.vocab_file(tag).exists():
        prev = [str(w) for w in np.load(paths.vocab_file(tag))["words"][nb:]]
        if prev != words:
            raise SystemExit(f"{tag}: existing word list differs from this selection "
                             f"({len(prev)} vs {len(words)}); refusing to overwrite")
        lg.info(f"{tag}: vocabulary/splits already exist with an identical word list")
    else:
      np.savez(paths.vocab_file(tag),
             words=np.concatenate([base["words"], np.asarray(words)]),
             freq=np.concatenate([base["freq"], freq]),
             decile=np.concatenate([base["decile"].astype(np.int8), dec_new]),
             is_stop=np.concatenate([base["is_stop"], np.zeros(len(words), bool)]),
             occ_pid=np.concatenate([base["occ_pid"],
                                     np.full((len(words), base["occ_pid"].shape[1]), -1, np.int32)]),
             occ_n=np.concatenate([base["occ_n"], freq.astype(np.int32)]),
             is_domain=np.concatenate([np.zeros(nb, bool), np.ones(len(words), bool)]))
      sp = np.load(paths.vsplits_file(""))
      pad = np.zeros(len(words), bool)
      np.savez(paths.vsplits_file(tag),
               domain=np.concatenate([np.zeros(nb, bool), np.ones(len(words), bool)]),
               random=np.concatenate([sp["random"], pad]),
               cluster=np.concatenate([sp["cluster"], pad]),
               rare=np.concatenate([sp["rare"], pad]),
               cluster_labels=np.concatenate([sp["cluster_labels"], np.full(len(words), -1)]))
    bp = np.load(paths.proto_file(a.encoder))
    np.savez(paths.proto_file(a.encoder, tag),
             protoA=np.concatenate([bp["protoA"], P.astype(np.float16)]),
             protoB=np.concatenate([bp["protoB"], P.astype(np.float16)]),
             cntA=np.concatenate([bp["cntA"], n.cpu().numpy()]),
             cntB=np.concatenate([bp["cntB"], n.cpu().numpy()]),
             layers=bp["layers"])
    sfx = "" if a.select == "df" else "_" + a.select
    vj = paths.RESULTS / f"91_domain_vocab_{a.name}{sfx}.json"
    if not vj.exists():
        save_json(dict(name=a.name, select=a.select, n_types_absent=len(cnt), n_added=len(words),
                       doc_share_min=float(share.min()), doc_share_median=float(np.median(share)),
                       doc_share_max=float(share.max()), freq_min=int(freq.min()),
                       freq_max=int(freq.max()), examples=words[:80]), vj)
    save_json(dict(name=a.name, select=a.select, encoder=a.encoder, n_added=len(words),
                   n_usable=int(ok.sum()), median_occurrences=float(n.median())),
              paths.RESULTS / f"91_domain_proto_{a.name}{sfx}_{a.encoder}.json")
    lg.info(f"{a.name}: wrote vocabulary and prototypes under tag {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--encoder", default="e5")
    ap.add_argument("--n-add", type=int, default=3000)
    ap.add_argument("--min-occ", type=int, default=5)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--select", default="df", choices=["df", "tfidf", "dfcap", "idf"])
    ap.add_argument("--max-df", type=float, default=0.10,
                    help="dfcap only: drop terms appearing in more than this share of passages")
    main(ap.parse_args())
