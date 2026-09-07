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
    # ---- Pilot M backbones (notes/backbones.md): 28-layer Qwen3-0.6B decoders ----
    # Prompts follow each model card's own reference code (D17): Octen = the Qwen3-Embedding
    # format, an instruction before queries and bare documents, with the <|endoftext|> the
    # tokenizer appends as the pooled token; jina = "Query: " / "Document: " with the
    # retrieval LoRA merged into the weights (identical outputs, 1.8x faster). Both pool the
    # last token; both ship in bf16 and are finite at every layer in fp16 on Turing.
    # `layers` are the single-layer candidates scripts/95_layer_probe.py chooses among.
    "octen": dict(
        hf="Octen/Octen-Embedding-0.6B", pooling="last",
        q_prefix="Instruct: Given a web search query, retrieve relevant passages that "
                 "answer the query\nQuery:",
        d_prefix="", dim=1024, layers=[12, 16, 20, 24, 28], fp16_weights=True,
        trim_punct=True),
    "jina5s": dict(
        hf="jinaai/jina-embeddings-v5-text-small", pooling="last",
        q_prefix="Query: ", d_prefix="Document: ", dim=1024, layers=[12, 16, 20, 24, 28],
        trust_remote_code=True, adapter="retrieval", merge_adapter=True, fp16_weights=True,
        trim_punct=True),
}
# The layer probe (D14) found retrieval improving monotonically toward the shallowest
# candidate for jina5s (layer 12 of 28). These keys extend the sweep to {6, 8, 10} as
# separate encoders -- identical models and prompts, different artifact names -- so the
# extension can run beside a live pipeline without touching its files.
ENCODERS["octen_lo"] = dict(ENCODERS["octen"], layers=[6, 8, 10])
ENCODERS["jina5s_lo"] = dict(ENCODERS["jina5s"], layers=[6, 8, 10])
COLBERT = "colbert-ir/colbertv2.0"
SPLADE = "naver/splade-cocondenser-ensembledistil"

LAYERS = list(range(4, 13))            # §0.1 sweep indices 4..12


def layers_for(enc: str) -> list[int]:
    """The hidden layers stored for an encoder: §0.1's sweep for the BERT-sized models,
    ColBERTv2 at its final layer only, and an explicit candidate list for the Pilot M
    decoders. Every artifact indexed by layer must go through this, never LAYERS."""
    if enc == "colbert":
        return [12]
    return list(ENCODERS.get(enc, {}).get("layers", LAYERS))


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
