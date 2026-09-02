# Dynamic-Vocabulary Learned Sparse Retrieval — pilot study

A sparse retriever whose output dimensions are **defined by text**, not by rows of a
learned matrix: vocabulary entry *j* is a text *t_j*, and its vector is `f(t_j)` for
the same encoder `f` that embeds the input. A new entry is added by encoding its
text — no new parameters, no retraining.

This repository runs the pilot protocol (`notes/protocol.md`) that decides whether
that premise holds, in order **A → B → C → D (→ E)**.

| Pilot | Question | Status |
|---|---|---|
| A | Do token states carry term-level semantics against text-defined entries? | see `results/14_pilotA_*.json` |
| B | Is an entry a bare string or a contextual prototype? | `results/20_pilotB_*.json` |
| C | Does the untrained projection already retrieve? | `results/30_pilotC*.json` |
| D | Do held-out entries behave like seen ones? | `results/40_pilotD*.json` |
| E | Can insertion-time calibration close a systematic gap? | conditional on D |

## Layout

```
dvlsr/         library: paths, data, encoders, whitening, sparse rule, metrics
scripts/       numbered stages; each writes a JSON artifact into results/
configs/       polysemy set and run configs
results/       small JSON results (versioned)
reports/       written findings per pilot
notes/         protocol, deviations, running log
```

Large artifacts (collection binary, corpus embeddings, banks, prototypes,
checkpoints) live outside the repo under `$DVLSR_DATA` (default
`/workspace/SPARSE/dvlsr`), keyed by the same names.

## Reproducing

```bash
export PYTHONPATH=.
python3 scripts/01_vocab.py                       # freeze V (30k) + occurrence sample
python3 scripts/02_encode_corpus.py --encoder e5  # full-collection dense vectors
python3 scripts/03_reference_runs.py --what all   # BM25 + SPLADE++ (Pyserini)
python3 scripts/04_dense_retrieval.py --encoder e5
python3 scripts/10_probes_banks.py --encoder plan # probes / banks / tau samples
python3 scripts/11_prototypes.py --encoder e5     # R2 contextual prototypes
python3 scripts/13_whitening.py --encoder e5      # freeze every §0.5 transform
python3 scripts/14_pilotA.py --encoder e5
```

Run everything with `./run_all.sh` (it fixes the OpenBLAS thread cap and the
JDK path Pyserini needs). Findings land in `reports/FINDINGS.md`; every deviation
from the written protocol is in `notes/deviations.md`, and `notes/log.md` is the
running record.

## Validation of the pipeline (§Appendix "sanity before A")

| System | full-collection MRR@10 | published |
|---|---|---|
| BM25 (k1=0.82, b=0.68) | 0.1874 | 0.1875 |
| SPLADE++ CoCondenser-EnsembleDistil | 0.3827 | 0.383 |
| e5-base-v2 | 0.3542 | ~0.35 |
| bge-base-en-v1.5 | 0.3498 | ~0.35 |

All four reproduce, so the prefix / pooling / normalisation conventions are right
and the pilots do not inherit a pipeline bug.
