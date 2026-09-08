# Refactor plan: SPARSE2 → the GenZ architecture

Status: **plan for review, nothing implemented** (2026-09-08).
Source of truth for the target: `/home/hkkang/GenZ` (commit 6130c1e, 218 Python files).
Source of truth for the science: `SUMMARY.md`, `notes/main_experiment.md`.

## 0. Goals, non-goals, principles

**Goal.** Rebuild this project as a GenZ-shaped codebase: Hydra config groups, a `src/`
package with registries and abstract bases (dataset / model / index / representation /
tokenization / task / metric / utils), Lightning-driven `scripts/evaluate.py` and
`scripts/index.py`, `tests/` with pytest, Sphinx docs, one pinned `requirements.txt`. On top
of that shape, add what GenZ does not have and the main experiment needs: a **training
stack**, an **annotation pipeline** (negative mining, teacher scoring), the **vocabulary
artifacts** (prototypes, whitening, layer probe), **dynamic entry insertion with the
per-query gate**, and the **evaluation suites** decided in `notes/main_experiment.md`.

**Non-goals.** No remote inference servers (vLLM, DeepSpeed-MII, FastAPI), no LLM
expanders (doc2query/query2doc), no GenZ/LASSER/JANUS/PromptReps models. Those GenZ
subsystems are out of scope and are not copied. The pilot scripts (`scripts/01_*`–`99_*`)
are not ported one by one: the pilots are finished and their record stays in git.

**Principles.**
1. *Same shape as GenZ, not a fork of GenZ.* Framework modules are copied with their tests
   where they are generic (registry, dataset base, index, representation, metric, utils,
   task, Lightning glue), trimmed of the remote/expander hooks we do not use.
2. *Parity before deletion.* Every ported component reproduces a pilot number to a stated
   tolerance before the legacy code that produced that number is removed.
3. *Configuration over code.* Backbone, layer, vocabulary, insertion rule, gate, training
   recipe, dataset and suite are Hydra groups; a run is a command line, not a script edit.
4. *Artifacts are addressed by content, not by history.* Prototypes, whitening, entry
   matrices and checkpoints live under a deterministic path derived from (backbone,
   vocabulary, layer, recipe) and carry a manifest, so nothing is keyed on "gpu10".
5. *Frozen encoder discipline is enforced by type.* The encoder never receives gradients
   unless the (ablation-only) trained-encoder variant is explicitly selected.

## 1. What we adopt from GenZ (observed)

| GenZ piece | path | what it gives us |
|---|---|---|
| Hydra entry | `config/config.yaml` + groups `model/ dataset/ prompt/ testing/ index/ server/`; `config/path.py` | one config tree; `python scripts/evaluate.py model=… dataset=…` |
| Registries | `src/utils/registry.py` `Registry`; `src/model/registry.py`, `src/dataset/registry.py`, `src/task/registry.py` | name → class, wired by `cfg.name` |
| Retriever bases | `src/model/retriever/base.py` `BaseRetriever` (torch.nn.Module, `retrieve`/`rerank`, index load/save); `sparse/base.py` `SparseRetriever`; `sparse/neural/base.py` `NeuralSparseRetriever` (`build_index` via `SparseVectorIndexBuilder`, `encode_docs`/`encode_queries`) | our model is one `NeuralSparseRetriever` subclass |
| Representations | `src/representation/{base,sparse,dense,hybrid}.py`: `SparseCOORepresentation` (ids + values over a `vocab_size`) | our sparse vectors over the entry vocabulary |
| Index | `src/index/sparse.py` `SparseVectorIndex`, `InvertedIndex` (BM25); `src/index/builder/*` shard-and-combine builders; numba scoring | full-corpus retrieval, parallel indexing with `torchrun` |
| Datasets | `src/dataset/datasets/base.py` `BaseDataset` (HF-hosted queries/corpus/qrels, cached data instances, retrieval/reranking items), `beir.py`, `lotte.py`; `src/dataset/corpus.py` `CorpusDataset`; `src/dataset/pl_module.py` `DataModule` | BEIR corpora for free; one pattern for new suites |
| Evaluation | `src/model/pl_module.py` `LightningModule` (test-only), `src/task/{retrieval,reranking}.py`, `src/metric/retrieval.py` `RetrievalMetrics` (torchmetrics nDCG/MRR/MAP/Recall/Hit) | multi-GPU evaluation, standard metric names |
| Tokenization | `src/tokenization/tokenizer/{base,huggingface,word_level}.py`, stopword filter, `vocab.py` | the place for our word-unit rule |
| Utilities | `src/utils/{logging,distributed,misc,string,tensor}.py` | rank-zero logging, seeding, tqdm-safe logging |
| Provenance | `src/benchmark/{retrieval_quality,sparse_scoring}.py`, `benchmarks/baselines/benchmark-result-template.json`: result JSON with `benchmark_config` + sha256, manifests (git commit, dirty flag, package versions, GPU) | adopted for every result file we write |
| Tooling | `tests/` (39 files, `unittest.TestCase` with hand-rolled dummies and `unittest.mock`; no conftest/fixtures; `test_imports.py` guards importability; GPU suites gated by env vars), `.github/workflows/unittest.yaml` (3.12–3.14), Sphinx `docs/` (apidoc), `requirements.txt` pinned, ruff used ad hoc without a config file | CI and docs |

Conventions to keep exactly (from the survey): discovery is by **decorator registries**
(`Registry.register("name")`), never Hydra `_target_`; the dataset registry is filled
imperatively from name lists (`"beir/scifact" -> BEIRDataset`); lookup key is always
`cfg.<group>.name`; every concrete config ends with `defaults: [_base, _self_]` (datasets:
`[base, <family>/_base, _self_]` with `# @package dataset`); stdlib `logging` only
(`logging.getLogger("ClassName")` in `src/`, `get_logger(__name__, __file__)` in scripts,
`log_if_rank_zero` in distributed code); Google-style docstrings with `Args/Returns/Raises`;
explicit `ValueError/RuntimeError` with messages; `hydra.run.dir: logs` with no per-run
directories (Lightning writes `logs/lightning_logs/version_N`); index paths derived from
config (`{model}_{dataset}_{hf_name}_len={max_input_length}`); `data/<name>/data_instances.pkl`
caches built once by `prepare_data`. Known defect to fix on copy: `src/utils/__init__.py`
re-exports `is_main_process` from `.misc` although it lives in `distributed.py`.

GenZ has **no training code** (no `training_step`, no `scripts/train.py`) and its
`DenseRetriever` is a stub (`NotImplementedError`). Both are built new here.

## 2. Target layout

```
SPARSE2/
├── config/
│   ├── config.yaml                  # defaults: model=dvlsr/jina dataset=beir/scifact testing=retrieval …
│   ├── path.py
│   ├── backbone/  {jina5s,gemma,arctic,e5,octen}.yaml      # NEW group: encoder identity, pooling, prompts, layer list, dtype, unit rule
│   ├── model/     _base.yaml dvlsr.yaml bm25.yaml splade.yaml dense.yaml   # dvlsr = ours
│   ├── vocabulary/ _base.yaml msmarco30k.yaml               # NEW: V construction, prototype k, whitening eps, layer probe
│   ├── insertion/ none.yaml df.yaml tfidf.yaml dfcap.yaml  # NEW: evaluation-time entry insertion rule
│   ├── gate/      none.yaml fallback.yaml dfdrop.yaml topk.yaml   # NEW: per-query gate (K, mode, two stores)
│   ├── training/  _base.yaml msmarco_only.yaml mixture.yaml  # NEW: losses, steps, batch, caps, stratified sampler
│   ├── annotation/ _base.yaml qwen3.yaml                    # NEW: miner, teacher, filter thresholds, candidate budget
│   ├── dataset/   base.yaml beir/ rteb/ shift/ train/       # rteb/ and shift/ NEW; train/ = mixture sources
│   ├── testing/   retrieval.yaml insertion.yaml             # insertion = paired with/without + bootstrap
│   ├── index/     base.yaml
│   └── hydra/job_logging/colorlog_custom.yaml
├── src/
│   ├── utils/            (copied: logging, registry, distributed, misc, string, tensor, + precision.py from dvlsr)
│   ├── representation/   (copied)
│   ├── index/            (copied) + two_store.py  # NEW: trained/inserted posting stores, gate-aware search
│   ├── tokenization/     (copied) + word_units.py # NEW: text-defined alnum units, piece→unit overlap (D19)
│   ├── dataset/          (copied base/beir/corpus/collator/pl_module/types/registry) + datasets/rteb.py, unified.py, training/{mixture.py, sampler.py, annotation_store.py}
│   ├── encoder/          NEW: backbone wrappers (from dvlsr/encoders.py): Encoder, truncate_to, pooling, post-load fixes, state_scale
│   ├── vocabulary/       NEW: builder.py (V from text, alnum), prototypes.py, banks.py, whitening.py (from dvlsr/whitening.py), layer_probe.py, domain_vocab.py (from 91), artifacts.py (manifest + paths)
│   ├── model/
│   │   ├── registry.py, pl_module.py (copied, eval)
│   │   ├── retriever/base.py, sparse/…, lexical/bm25.py, neural/splade.py (copied)
│   │   ├── retriever/sparse/neural/dvlsr.py     # NEW: our retriever (encode_docs/queries, insert_entries, gate)
│   │   ├── retriever/dense/{base,hf_dense.py}   # NEW implementation (GenZ's is a stub): encode, FAISS/GPU brute force, index
│   │   ├── head/                                # NEW: Head, EntryHead, scalars t/b, entry_norm_affine (from dvlsr/model.py)
│   │   └── reranker/{base,qwen3.py}             # NEW: cross-encoder teacher for annotation
│   ├── training/         NEW: pl_module.py (training LightningModule), losses.py (CE, KL, margin-MSE, FLOPS), sampler.py, checkpoint.py
│   ├── annotation/       NEW: store.py (unified passage store), mine.py (BM25 + dense), score.py (teacher), filter.py (positive-aware)
│   ├── metric/           (copied) + bootstrap.py (paired bootstrap, ratio CI from dvlsr/metrics.py)
│   └── task/             (copied) + insertion.py  # NEW task: with/without entries, paired delta
├── scripts/
│   ├── evaluate.py, index.py           (copied, trimmed of remote/expander branches)
│   ├── train.py                        NEW
│   ├── preprocess/vocabulary/{build_vocab.py, build_prototypes.py, build_banks.py, estimate_whitening.py, layer_probe.py}
│   ├── preprocess/annotation/{build_store.py, mine_negatives.py, score_teacher.py, filter_pairs.py, upload.py}
│   ├── preprocess/dataset/{migrate_beir.py, upload_*.py}   (copied) + convert_{litsearch,tripclick,cure}.py
│   └── analysis/{insertion_report.py, gate_report.py, diagnose.py, geometry.py, backbone_survey.py}
├── tests/   (copied framework tests that still apply + ours: word_units, whitening, head, scoring parity, two-store, gate, bootstrap, datasets, training smoke)
├── docs/    (Sphinx, apidoc)
├── benchmarks/  (result templates)
├── notes/   (kept as is: protocol, deviations, pilot notes, main_experiment, this plan)
├── results/ reports/ (kept, read-only record of the pilots)
├── scripts/pilot_study/  legacy/ (pilot package + scripts, self-contained) + reproduce_{pilotC,pilotD,pilotF,domain_shift,query_gate}.sh
├── .github/workflows/unittest.yaml, deploy-doc.yml
├── Dockerfile, docker-compose.yml, requirements.txt, README.md, LICENSE
```

Import style follows GenZ: `from src.model.retriever…`, `PYTHONPATH=.`; no installed package.

## 3. Component mapping (current → target)

| current | target | action |
|---|---|---|
| `dvlsr/paths.py` (ENCODERS dict, constants, paths) | `config/backbone/*.yaml`, `config/vocabulary/_base.yaml`, `src/vocabulary/artifacts.py` (path scheme), `config/path.py` | rewrite: constants become config; paths become derived, manifest-carrying |
| `dvlsr/encoders.py` (`Encoder`, `_word_units*`, `truncate_to`, `rematerialize_gte_buffers`, pooling, post-pool dense head, `state_scale`) | `src/encoder/backbone.py` (model load, layer truncation, pooling, post-load hooks), `src/tokenization/word_units.py` (unit rule D19 + legacy rule for e5), `src/encoder/units.py` (unit-state gather) | port, split; keep both unit rules behind `backbone.unit_rule` |
| `dvlsr/encoders.py::SpladeEncoder`, `ColbertEncoder` | SPLADE → GenZ `splade.py`; ColBERT dropped (pilot A only) | replace / drop |
| `dvlsr/whitening.py` | `src/vocabulary/whitening.py` | port (same math, add manifest) |
| `dvlsr/sparse.py` (training-free rule, τ search, `to_csr`) | `src/model/retriever/sparse/neural/dvlsr.py` (`encode_*`), `src/vocabulary/layer_probe.py` (τ search) | port |
| `dvlsr/model.py` (`Head`, `EntryHead`, `segment_max`, `sparse_rep`, `flops`, `info_nce`, `calibration_loss`, `entry_norm_affine`, `kmeans_labels`) | `src/model/head/{mlp.py,entry.py}`, `src/training/losses.py`, `src/model/retriever/sparse/neural/dvlsr.py` (scoring), `src/vocabulary/calibration.py` (Pilot F affine) | port |
| `dvlsr/retrieval.py` (GPU inverted index) | GenZ `src/index/sparse.py` `SparseVectorIndex` + `src/index/two_store.py` | replace; keep our exact GPU scorer only if the GenZ numba path is slower on 8.8M docs (benchmark in Phase 3) |
| `dvlsr/metrics.py` (MRR/recall, bootstrap) | GenZ `RetrievalMetrics` + `src/metric/bootstrap.py` | port bootstrap; metrics via torchmetrics |
| `dvlsr/data.py` (`Collection` mmap blob, MS MARCO splits) | `src/annotation/store.py` (unified passage store: MS MARCO + Wikipedia + others, same mmap layout) and HF datasets for evaluation corpora | port the blob layout; evaluation corpora go through `BaseDataset` |
| `dvlsr/precision.py` | `src/utils/precision.py` + `backbone.dtype` / `no_autocast` | port |
| `dvlsr/util.py` (logger, JSON, Timer, `run_queue` multi-GPU shard runner) | GenZ `src/utils/logging.py`; shard runner → `torchrun` + Lightning DDP (`index.parallel`) | replace |
| `dvlsr/probes.py` (Pilot A/B probe metrics) | `legacy/` | archive |
| `scripts/01_vocab.py`, `11_prototypes.py`, `10_probes_banks.py`, `12_r1.py`, `13_whitening.py`, `83_proto_k100.py` | `scripts/preprocess/vocabulary/*` over `src/vocabulary/*` | port (R1 dropped: text-defined prototypes only) |
| `scripts/95_layer_probe.py` | `src/vocabulary/layer_probe.py`, `scripts/preprocess/vocabulary/layer_probe.py` | port |
| `scripts/40_mine_negatives.py`, `41_vocab_splits.py` | `src/annotation/mine.py`; `src/vocabulary/splits.py` (rare/random/cluster/phrase held-out splits) | rewrite (mixture, BM25 + dense, positive-aware filter) / port |
| `scripts/42_train.py` (688 lines: V1/V2/V3, VD, meta-frac, teacher, oracle, crand) | `src/training/pl_module.py` + `losses.py` + `config/training/*`; V2/V3/VD/meta-frac kept as config switches (`training.variant`, `training.vocab_dropout`) | rewrite around Lightning; keep the loss math verbatim |
| `scripts/43_encode_eval.py`, `44_pilotD_eval.py`, `46_pilotD_report.py` (recovery ratio ρ, Q_H, C-alias) | `src/task/insertion.py` + `scripts/analysis/insertion_report.py` (recovery ratio at full scale against the retrained oracle) | port the ρ computation; drop pilot-only tables |
| `scripts/90_domain_data.py` | `scripts/preprocess/dataset/convert_*.py` + HF upload (unified schema) | replace |
| `scripts/91_domain_vocab.py` (df / tfidf / dfcap / idf selection) | `src/vocabulary/domain_vocab.py` + `config/insertion/*` | port |
| `scripts/92_domain_eval.py` (with/without, controls, `--norm`, `--qgate`, `--two-store`, caches) | `src/model/retriever/sparse/neural/dvlsr.py::insert_entries`, `src/index/two_store.py`, `src/task/insertion.py`, `config/gate/*`, `testing=insertion` | port; the paired with/without run becomes a task, not a script |
| `scripts/93_domain_report.py`, `99_query_gate_report.py` | `scripts/analysis/{insertion_report,gate_report}.py` (write `reports/*.md` from result JSON) | port |
| `scripts/94_domain_baselines.py` | `model=bm25`, `model=splade`, `model=dense backbone=…` through `scripts/evaluate.py` | replace (GenZ BM25/SPLADE; new dense) |
| `scripts/96_domain_diagnose.py`, `98_geometry_analysis.py`, `97_qf_filter.py` | `scripts/analysis/{diagnose,geometry}.py`; qf filter kept only as the documented upper bound (`insertion.filter=query_sample`, flagged non-compliant) | port (analysis only) |
| `scripts/60_full_corpus.py`, `02–06_*`, `30–33_*` (C1 sub-corpus machinery) | dropped: the main experiment indexes full corpora with `scripts/index.py` | drop |
| `scripts/14–21, 45, 47, 48, 50, 51, 70, 80–82` (pilots A/B/D/E/F, phrase pilot, synthesis) | `legacy/` | archive |
| `run_backbone.sh`, `run_backbone_prep.sh`, `run_all.sh`, `run_followups.sh` | `scripts/pipelines/{prepare_backbone.sh, train_backbone.sh}` (thin wrappers over Hydra commands) | rewrite |
| `env.sh` / `env.example.sh` | `.env` (HF_TOKEN, DATA_ROOT) via `python-dotenv` as GenZ does | replace |
| `results_gpu10/`, `reports_gpu10/`, `notes/`, `SUMMARY.md`, `PROTOCOL.md` | unchanged; new results go to `results/main/…` | keep |

## 4. New components (in neither codebase)

1. **`DVLSR` retriever** (`src/model/retriever/sparse/neural/dvlsr.py`), registered as
   `model=dvlsr`. Holds: backbone (`src/encoder`), unit rule, whitening transforms (H, Q),
   entry matrix E (trained rows; inserted rows appended), head g + scalars (t, b), Pilot F
   affine (optional). `encode_docs`/`encode_queries` → `SparseCOORepresentation` over the
   entry vocabulary (`vocab_size = |E|`), top-k truncation, `log1p(relu(t·cos − b))`.
   `insert_entries(texts, selection)` builds prototypes for new entries from the *target
   corpus* (reusing `src/vocabulary/prototypes.py`), appends rows, returns the inserted id
   range. Loads a checkpoint written by the training stack (head + scalars + E manifest).
2. **Two-store index and gate** (`src/index/two_store.py`): trained and inserted posting
   stores; `search(query_rep, gate)` counts inserted entries firing on the query and applies
   `none | all(fallback) | dfdrop | topk` with K; df statistics of inserted entries are
   computed at indexing time. This is the deployable rule of `reports_gpu10/QUERY_GATE.md`.
3. **Insertion task** (`src/task/insertion.py`, `testing=insertion`): one evaluation =
   {baseline store} vs {store + inserted entries (+ gate)} on the same queries; emits
   per-query metrics for both, paired bootstrap CIs (`src/metric/bootstrap.py`), controls
   (`insertion.control=random|other_corpus`). Replaces `92_domain_eval.py`.
4. **Training stack** (`src/training/`, `scripts/train.py`): Lightning module with
   `training_step` over batches of (query, positive, candidates, teacher scores); losses =
   CE over candidates + KL / margin-MSE to the teacher + FLOPS regularisers (λ_d, λ_q);
   variants V1 (frozen encoder, default), V2 (LoRA), V3 (full) and vocabulary dropout kept
   as config; source-stratified batch sampler with per-source caps; checkpoints =
   {head, t, b, E-manifest, config}. Encoder chunking (`enc-chunk`, `rep-chunk`) ported.
5. **Annotation pipeline** (`src/annotation/`): unified passage store (mmap blob, ids,
   source tags); BM25 (GenZ `InvertedIndex`) + dense mining (Qwen3-Embedding-0.6B through
   the new `HFDenseRetriever`); candidate sampling per `notes/main_experiment.md`; teacher
   scoring with `Qwen3Reranker` (`src/model/reranker/qwen3.py`, logit(yes)−logit(no),
   P(yes) for filtering, per-source instruction); positive-aware filter; output = one
   parquet per source {qid, pos, cands, scores, source}.
6. **Dense retriever implementation** (`src/model/retriever/dense/hf_dense.py`): GenZ's is
   a stub. Encode with the backbone's own pooling + prompts (our `Encoder.encode_pooled`),
   GPU brute-force or FAISS flat index, save/load. Serves as (a) the dense baseline row per
   backbone, (b) the miner.
7. **Datasets**: `datasets/rteb.py` (the `Hyukkyu/rteb-*` unified schema), a
   `UnifiedRetrievalDataset` for every new corpus converted to that schema (LitSearch,
   TripClick/TripJudge, CUREv1 English), `training/mixture.py` (sources, caps, annotation
   store). MS MARCO dev in-domain = `beir/msmarco` as in GenZ.
8. **Vocabulary artifacts** (`src/vocabulary/`): V builder (alnum words, ≥100 occurrences,
   top 30k), occurrence sampling, prototypes (k=50 mean states per layer), banks (200k
   states H/Q), whitening (ZCA, ε = 0.01·tr/d), layer probe (training-free MRR per
   candidate layer → picks L and τ), held-out splits. Artifact directory
   `data/artifacts/<backbone>/<vocab>/<layer>/…` with `manifest.json` (hashes of inputs).
9. **Bootstrap metrics** (`src/metric/bootstrap.py`): paired and ratio bootstrap (10,000
   resamples), CI and two-sided p, as in `dvlsr/metrics.py`.

## 5. Configuration design (examples)

```
# artifacts for one backbone
python scripts/preprocess/vocabulary/build_vocab.py vocabulary=msmarco30k
torchrun --nproc_per_node=4 scripts/preprocess/vocabulary/build_prototypes.py backbone=jina5s vocabulary=msmarco30k
python scripts/preprocess/vocabulary/estimate_whitening.py backbone=jina5s
python scripts/preprocess/vocabulary/layer_probe.py backbone=jina5s            # writes layer + tau into the manifest

# annotation (once, shared by all backbones)
python scripts/preprocess/annotation/build_store.py dataset=train/mixture
torchrun --nproc_per_node=4 scripts/preprocess/annotation/mine_negatives.py annotation=qwen3
torchrun --nproc_per_node=4 scripts/preprocess/annotation/score_teacher.py annotation=qwen3

# training
python scripts/train.py model=dvlsr backbone=jina5s training=msmarco_only
python scripts/train.py model=dvlsr backbone=gemma  training=mixture training.max_steps=12500

# indexing + evaluation (GenZ style)
torchrun --nproc_per_node=4 scripts/index.py model=dvlsr backbone=jina5s dataset=beir/msmarco index.parallel.enabled=true
python scripts/evaluate.py model=dvlsr backbone=jina5s dataset=beir/msmarco testing=retrieval
python scripts/evaluate.py model=dvlsr backbone=jina5s dataset=rteb/CUREv1 testing=insertion insertion=df gate=dfdrop gate.k=20
python scripts/evaluate.py model=bm25 dataset=shift/tripclick
python scripts/evaluate.py model=dense backbone=arctic dataset=shift/litsearch
```

Config groups and their keys:

* `backbone/*`: `hf_name, model_class, pooling {mean,cls,last}, q_prefix, d_prefix, dtype,
  no_autocast, layers (probe candidates), unit_rule {alnum,legacy}, trust_remote_code,
  adapter, merge_adapter, config_kwargs, post_load, state_scale, st_dense, fp16_weights`.
* `vocabulary/*`: `size, min_occurrences, proto_k, bank_size, whitening_eps, nnz_targets,
  pool (which corpus)`.
* `model/dvlsr.yaml`: `checkpoint (or "training-free"), topk_doc, topk_query, log1p,
  calibration (pilot F affine on|off)`; `model/dense.yaml`: `backbone-driven`.
* `insertion/*`: `select {df,tfidf,dfcap,idf}, n_add, min_occ, control {none,random,other}`.
* `gate/*`: `mode {none,all,dfdrop,topk}, k, df_threshold, two_store`.
* `training/*`: `variant {V1,V2,V3}, steps, batch_queries, candidates_per_query, lr_head,
  warmup, lambda_d, lambda_q, teacher {none,ce_scores}, loss {ce,kl,margin_mse}, weights,
  vocab_dropout {none,entry,cluster}, caps per source, stratified true, seed`.
* `annotation/*`: `miner {bm25,dense}, dense_backbone, candidates_per_query, rank windows,
  teacher, instruction per source, pos_filter_pct 0.95, pos_floor`.
* `dataset/rteb/*`, `dataset/shift/*`, `dataset/train/*`.
* `testing/insertion.yaml`: `bootstrap_resamples 10000, k_list, report_per_query`.

## 6. Data and artifacts on disk

```
$DATA_ROOT/                        (from .env; the scratch mount, never in git)
├── hf/                            HF cache (models, datasets)
├── store/                         unified passage store: passages.bin, offsets.npy, ids.parquet (source, id)
├── annotation/<recipe>/<source>.parquet   mined candidates + teacher scores + filter flags
├── artifacts/<backbone>/<vocab>/  vocab.parquet, occurrences.npz, protos_L{l}.npy, bankH.npy, bankQ.npy, whitening_{H,Q}_L{l}.npz, manifest.json, layer_probe.json
├── checkpoints/<backbone>/<recipe>/<run>/  head.pt, scalars.json, E_manifest.json, config.yaml, history.json
├── indexes/<model>/<dataset>/     GenZ index files (+ two_store/ for inserted entries)
└── cache/                         extended-encode caches keyed by manifest hashes
repo/results/main/<suite>/<backbone>/<run>.json   small JSON, versioned
repo/reports/main/*.md                           rendered tables
```

Evaluation corpora live on the HF Hub in the unified schema (`Hyukkyu/rteb-*` already;
BEIR via `Hyukkyu/beir-*` as GenZ; LitSearch, TripClick, CUREv1 to be uploaded with
`scripts/preprocess/dataset/upload_unified.py`).

## 7. Migration phases and verification gates

Each phase ends with a gate; legacy code for that phase is deleted only after the gate.

| phase | work | gate (parity / test) | est. |
|---|---|---|---|
| 0. Freeze | new repository on `/mnt/sdc`; copy the pilot package + scripts to `scripts/pilot_study/legacy/`; ruff/pytest exclude it; SPARSE2 tagged `pilot-final` | one pilot command runs from the copy | 0.5 d |
| 1. Skeleton | copy GenZ framework (`src/utils, representation, index, metric, task, tokenization, dataset base+beir+corpus+collator+pl_module+types, model/registry+pl_module+retriever/base+sparse/base+neural/base+lexical/bm25+neural/splade`), `scripts/{evaluate,index}.py`, `config/`, `tests/` for those, `requirements.txt`, CI, docs skeleton; strip remote/expander hooks | `pytest` green; `evaluate.py model=bm25 dataset=beir/scifact` reproduces GenZ's BM25 nDCG@10 on scifact; SPLADE-v3 scifact 0.686 as in `reports_gpu10/DOMAIN.md` | 2 d |
| 2. Encoder + units + artifacts | `src/encoder`, `src/tokenization/word_units.py`, `src/vocabulary/*`, preprocess scripts, backbone configs | (a) unit states bit-identical to `legacy` for e5 (legacy rule) and cosine ≥ 0.9999 for jina/arctic/gemma (alnum rule) on 1k passages; (b) prototypes/banks/whitening for jina reproduce the pilot artifacts (cos 1.0000); (c) layer probe reproduces jina 0.2981 @12, arctic 0.315 @24 | 3 d |
| 3. Retriever + index + evaluate | `dvlsr.py`, head, checkpoint loader (reads pilot `head.pt`), GenZ index build, `testing=retrieval` | training-free and V1oracle_jina5s on scifact/nfcorpus/trec-covid reproduce `reports_gpu10/DOMAIN.md` baselines to ±0.001 MRR@10; index build ≥ pilot throughput (417/259/142 p/s) | 3 d |
| 4. Insertion + gate | `insert_entries`, two-store index, gate modes, `testing=insertion`, bootstrap, controls, reports | the nine-cell table of `QUERY_GATE.md` (two stores + dfdrop K=20) reproduced to ±0.002 | 3 d |
| 5. Datasets | `rteb.py`, unified dataset, converters + uploads (LitSearch, TripClick, CUREv1), `dataset/shift/*`, `dataset/rteb/*` | loaders pass schema tests; BM25 on each corpus matches published numbers where they exist (e.g. CUREv1, FiQA) | 2 d |
| 6. Training stack | `src/training`, `scripts/train.py`, `training/msmarco_only.yaml` on the pilot negatives file | retraining V1oracle_jina5s (3,124 steps, seed 1) lands within ±0.005 MRR@10 of the pilot checkpoint on scifact/nfcorpus; loss curves overlay | 3 d |
| 7. Annotation pipeline | store, dense retriever, mining, Qwen3 reranker scoring, filter, `training/mixture.yaml` | MS MARCO subset: our mined negatives overlap the public package's top-50 ≥ 60%; teacher scores rank MS MARCO dev top-20 with MRR@10 ≥ 0.40 (sanity, in-domain only); throughput within 20% of estimates | 4 d + GPU (12 h store encode, 18 h scoring) |
| 8. Baselines | dense rows per backbone via `HFDenseRetriever`; SPLADE++/v3 via GenZ; our-own-SPLADE training (mixture arm) if kept | dense scifact/nfcorpus/trec-covid match `94_domain_baselines` (jina 0.744/0.391/0.828 nDCG@10) | 2 d |
| 9. Analysis + docs + CI | `scripts/analysis/*`, README (GenZ style), Sphinx, workflows, `benchmarks/` templates | docs build; CI green on 3.12; README commands all run | 2 d |
| 10. Pilot drivers | `scripts/pilot_study/reproduce_*.sh` for C, D, F, L/M, gate with expected numbers | each driver runs end to end on one cell | 1 d |

Total: about 25 working days of implementation plus the GPU time listed. Phases 2–4 are
the critical path; 5 and 7 can proceed in parallel with them.

## 8. Tests, CI, docs, environment

* **Tests** (GenZ style: `unittest.TestCase` files run under pytest, hand-rolled dummies
  and `unittest.mock`, no fixtures, `tests/test_imports.py` kept):
  `test_word_units.py` (alnum vs legacy on fixtures incl. `covid-19`, `5°C`, CJK),
  `test_whitening.py` (ZCA identity/scale invariance), `test_head.py`,
  `test_dvlsr_scoring.py` (scoring formula against a scalar oracle, like GenZ's SPLADE test),
  `test_two_store.py` (eviction impossible; gate modes), `test_bootstrap.py` (seeded CI),
  `test_insertion_task.py` (with/without on a toy corpus), `test_training_step.py` (CPU,
  tiny backbone stub), `test_datasets_unified.py` (schema), `test_annotation_filter.py`.
  Parity gates of §7 run as opt-in GPU tests (`SPARSE2_RUN_GPU_PARITY=1`), mirroring
  GenZ's `GENZ_RUN_VLLM_INTEGRATION`.
* **CI**: `unittest.yaml` on Python 3.12/3.13 (CPU torch); GPU parity is manual. One
  addition over GenZ: a checked-in `ruff.toml` and a lint step (GenZ runs ruff ad hoc).
* **Result files**: every evaluation writes GenZ's benchmark-artifact JSON (config hash,
  git commit and dirty flag, package versions, GPU) plus our per-query arrays for the
  bootstrap; reports are rendered from these files only.
* **Docs**: Sphinx apidoc as GenZ; README sections: Getting started, Artifacts, Annotation,
  Training, Indexing, Evaluation (retrieval / insertion / gate), Adding a backbone / dataset.
* **Environment**: GenZ's `requirements.txt` as the base (transformers 5.16.1, lightning
  2.6.5, hydra 1.3.5, numba, bm25s, torchmetrics, faiss-cpu) plus `peft` (jina adapters),
  `sentence-transformers` (reference validation only), `pyyaml`; Python 3.12 venv (the
  current venv is 3.10; GenZ targets 3.12–3.14). Lucene/JAVA dependency dropped (BM25 via
  bm25s/GenZ). Dockerfile inherited from GenZ.

## 9. Owner's decisions (2026-09-08)

1. **New repository, no git history.** Working name `SPARSE` (the acronym is fixed; its
   expansion is still being chosen). Lives on the scratch mount
   (`/mnt/sdc/hkkang/SPARSE`, symlinked from `~/SPARSE`) because the root filesystem is
   full. SPARSE2 stays as the read-only pilot record.
2. **Copy scope**: the framework layer plus `src/telemetry` (index.py depends on it) and
   the benchmark-artifact schema; `src/prompt` reduced to the base/constructor/registry and
   the `raw` prompt, since our backbones use plain prefixes. Nothing from `src/remote`,
   `src/model/expander`, `src/model/embedder`, the LLM prompt templates, or GenZ's own models.
3. **Pilot code** is kept under `scripts/pilot_study/` in the new repository: a
   self-contained copy of the pilot package and scripts (`scripts/pilot_study/legacy/`) so
   the numbers can be re-run exactly, plus thin reproduction drivers for the important
   pilots only: C (training-free retrieval), D (recovery ratio on the held-out splits),
   F (insertion-time calibration), L/M (domain shift on three backbones) and the query-time
   gate. Each driver states the expected numbers from `SUMMARY.md`. Phases 4 and 6 also
   reproduce L/M and D on the new stack, which is the long-term reproduction path.
4. **Datasets on the HF Hub** under the owner's account (`Hyukkyu/*`) in the RTEB unified
   schema, including LitSearch, TripClick and CUREv1.
5. **Python 3.14 and the newest packages** that resolve together at creation time, then
   pinned (GenZ's discipline: torch unpinned, everything else pinned). Environment via `uv`
   on the scratch mount.

## 10. Risks

1. **Root filesystem full** (726 MB free on `/`): every path of the new project, the venv,
   pip/uv caches and HF caches must be on `/mnt/sdc`; a stray cache on `/` will fail writes.
2. **Python 3.14 wheel coverage**: torch, numba, faiss-cpu, peft and sentence-transformers
   must all have cp314 wheels; GenZ's Dockerfile shows torch/numba/faiss do. If peft or
   sentence-transformers lag, jina's adapter merge is done once and the merged weights
   saved, and the reference-pipeline validation runs from the 3.10 venv.
3. **GPU index scorer.** GenZ scores sparse indexes with numba on CPU; the pilots scored on
   GPU. Phase 3 benchmarks both on the 8.8M-passage MS MARCO index; if numba is >3x slower,
   our GPU scorer is kept as `index.backend=gpu`.
4. **transformers remote-code fixes** (arctic-m's GTE buffers, Gemma's fp32-only path)
   stay as backbone `post_load` hooks; they are already written.
5. **Timeline.** ~5 weeks for one engineer with the GPU runs interleaved; the main
   experiment's artifact and annotation runs (Phases 2, 7) can start on this machine before
   the training stack (Phase 6) is finished.
