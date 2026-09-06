"""Central paths and global constants.

Code lives in the git repo; all large artifacts live on the scratch mount.
"""
from __future__ import annotations
import os, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("DVLSR_DATA", "/workspace/SPARSE/dvlsr"))

RAW = DATA / "data" / "raw"
PREP = DATA / "data" / "prep"          # collection binary, splits
ART = DATA / "artifacts"               # banks, whitening, prototypes, vocab
EMB = DATA / "emb"                     # corpus embeddings / sparse reps
RUNS = DATA / "runs"                   # retrieval runs (trec format / npz)
CKPT = DATA / "ckpt"                   # pilot D checkpoints

# Small JSON results and reports are versioned in git. A rebuild on another machine
# writes to its own directories (DVLSR_RESULTS_DIR etc.) so the study's record is
# never overwritten (notes/deviations.md D12).
RESULTS = REPO / os.environ.get("DVLSR_RESULTS_DIR", "results")
REPORTS = REPO / os.environ.get("DVLSR_REPORTS_DIR", "reports")
LOGS = REPO / os.environ.get("DVLSR_LOGS_DIR", "logs")

for _p in (PREP, ART, EMB, RUNS, CKPT, RESULTS, REPORTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- raw files ----
COLLECTION_TSV = RAW / "collection.tsv"
QUERIES_DEV_SMALL = RAW / "queries.dev.small.tsv"
QUERIES_TRAIN = RAW / "queries.train.tsv"
QRELS_DEV_SMALL = RAW / "qrels.dev.small.tsv"
QRELS_TRAIN = RAW / "qrels.train.tsv"

# ---- global constants (§0) ----
SEED = 20260902
N_PASSAGES = 8841823

ENCODERS = {
    "e5": dict(
        hf="intfloat/e5-base-v2", pooling="mean",
        q_prefix="query: ", d_prefix="passage: ", dim=768),
    "bge": dict(
        hf="BAAI/bge-base-en-v1.5", pooling="cls",
        q_prefix="Represent this sentence for searching relevant passages: ",
        d_prefix="", dim=768),
    "colbert": dict(
        hf="colbert-ir/colbertv2.0", pooling="token",
        q_prefix="[unused0] ", d_prefix="[unused1] ", dim=128),
}
COLBERT = "colbert-ir/colbertv2.0"
SPLADE = "naver/splade-cocondenser-ensembledistil"

LAYERS = list(range(4, 13))            # §0.1 sweep indices 4..12


def layers_for(enc: str) -> list[int]:
    """ColBERTv2 is the Pilot A ceiling and is used at its final layer only (§0.1)."""
    return [12] if enc == "colbert" else LAYERS


MAXLEN_DOC = 192
MAXLEN_QRY = 64

# §0.2 sample sizes
N_Q = 40_000                           # probe / bank pool
N_S = 100_000                          # statistics sample
N_BANK = 200_000                       # background token bank
N_RANDOM_C1 = 500_000
N_TRAIN_QUERIES = 200_000
N_QUERY_BANK_QUERIES = 50_000

# §0.3 vocabulary
V_SIZE = 30_000
V_MIN_OCC = 100
PROTO_MAX_OCC = 100                    # occurrences stored per entry (k=50 disjoint pairs)
PROTO_K = 50

# §0.5 whitening
WHITEN_EPS = 0.01

# §0.6 operating point
NNZ_DOC_TARGET = 120
NNZ_QRY_TARGET = 30

JAVA_HOME = os.environ.get("JAVA_HOME", "/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11")


# ---- vocabulary variants (the phrase/entity pilot appends entries to V) ----
def vocab_file(tag: str = ""):
    return ART / (f"vocab_{tag}.npz" if tag else "vocab.npz")


def proto_file(enc: str, tag: str = ""):
    return ART / (f"proto_{enc}_{tag}.npz" if tag else f"proto_{enc}.npz")


def vsplits_file(tag: str = ""):
    return PREP / (f"vocab_splits_{tag}.npz" if tag else "vocab_splits.npz")
