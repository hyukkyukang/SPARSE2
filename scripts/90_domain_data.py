"""Domain-shift evaluation, step 1: fetch a target corpus and put it in our formats.

Every held-out split in the study so far is a synthetic ablation of MS MARCO: hold out
random words, or a k-means region of the entry space, and call the region "a semantically
new domain". The motivating story is new entities and evolving jargon arriving in a live
corpus, and none of that evidence comes from a genuinely different collection.

This builds the real version. A BEIR-format corpus (scifact = scientific claims,
nfcorpus = nutrition and medicine, trec-covid = COVID literature) is downloaded and
converted into the same mmap'd blob + offsets layout `dvlsr.data.Collection` expects, so
every later stage reads it exactly as it reads MS MARCO.

Artifacts, all under $DVLSR_DATA/domain/<name>/:
  passages.bin / offsets.npy   the corpus, in Collection's layout
  ids.json                     row index -> the corpus's own document id
  queries.tsv                  test queries, qid<TAB>text
  qrels.tsv                    qid 0 docid rel, restricted to judged test queries
"""
from __future__ import annotations
import argparse, csv, io, json, os, sys, urllib.request, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json, Timer

lg = get_logger("domdata", "90_domain_data.log")
BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def root(name):
    d = paths.DATA / "domain" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(name, force=False):
    d = root(name)
    raw = d / "raw"
    # presence of the directory is not proof of a complete download: an interrupted fetch
    # leaves it there without a corpus, and skipping on existence alone then fails later
    if raw.exists() and not force and any(raw.rglob("corpus.jsonl")):
        return raw
    if raw.exists():
        import shutil
        shutil.rmtree(raw)
        lg.info(f"{name}: discarding an incomplete download")
    raw.mkdir(exist_ok=True)
    with Timer(f"download {name}", lg):
        with urllib.request.urlopen(BEIR.format(name), timeout=300) as r:
            blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(raw)
    lg.info(f"{name}: extracted to {raw}")
    return raw


def build(name, split="test", force=False):
    d = root(name)
    if (d / "offsets.npy").exists() and not force:
        lg.info(f"{name}: already built"); return
    raw = fetch(name, force)
    base = next(p for p in raw.rglob("corpus.jsonl")).parent

    ids, offs, pos = [], [0], 0
    with open(base / "corpus.jsonl") as fi, open(d / "passages.bin", "wb") as fo:
        for line in fi:
            o = json.loads(line)
            # title and body together, as BEIR's own evaluation does
            text = ((o.get("title", "") + " ") if o.get("title") else "") + o.get("text", "")
            b = text.strip().encode("utf-8")
            fo.write(b); pos += len(b); offs.append(pos); ids.append(str(o["_id"]))
    np.save(d / "offsets.npy", np.asarray(offs, np.int64))
    json.dump(ids, open(d / "ids.json", "w"))

    qrels = {}
    with open(base / "qrels" / f"{split}.tsv") as f:
        rd = csv.reader(f, delimiter="\t"); next(rd, None)
        for q, doc, rel in rd:
            if int(rel) > 0:
                qrels.setdefault(str(q), set()).add(str(doc))
    qtexts = {}
    with open(base / "queries.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if str(o["_id"]) in qrels:
                qtexts[str(o["_id"])] = o["text"].strip()
    qrels = {q: v for q, v in qrels.items() if q in qtexts}

    with open(d / "queries.tsv", "w") as f:
        for q, t in qtexts.items():
            f.write(f"{q}\t{t}\n")
    with open(d / "qrels.tsv", "w") as f:
        for q, docs in qrels.items():
            for doc in docs:
                f.write(f"{q} 0 {doc} 1\n")

    stats = dict(name=name, split=split, n_passages=len(ids), n_queries=len(qtexts),
                 n_qrels=sum(len(v) for v in qrels.values()),
                 mean_rel_per_query=float(np.mean([len(v) for v in qrels.values()])),
                 bytes=pos)
    save_json(stats, paths.RESULTS / f"90_domain_{name}.json")
    lg.info(f"{name}: {stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scifact")
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.name, a.split, a.force)
