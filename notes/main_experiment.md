# Main experiment — settings (living document, started 2026-09-08)

The pilot study (SUMMARY.md) is closed. This file records the main experiment's setting as
it is decided, with the reasoning, so that every later choice can be traced.

## Backbones (decided)

jina-embeddings-v5-text-small (Qwen3-0.6B decoder, last token, layer 12),
embeddinggemma-300m (Gemma 3 bidirectional, mean pooling + dense head, fp32, layer by
probe), snowflake-arctic-embed-l-v2.0 (XLM-R large, CLS, layer 24). Three pretraining
families, three pooling rules. Wiring and validation: `notes/backbones.md`; survey and
rationale: `notes/backbone_survey.md`. e5-base-v2 is the pilot contrast only.

## Training data (decided in outline)

Goal stated by the owner: use every public retrieval training set that does not overlap
the test suite, in particular RTEB-open (`Hyukkyu/rteb-open`).

| tier | sources | status |
|---|---|---|
| A. supervised retrieval | MS MARCO passage (502k queries, full), NQ, TriviaQA, SQuAD, HotpotQA, FEVER, ELI5, Quora duplicates, AllNLI | in |
| B. weakly supervised pairs | Yahoo Answers, PAQ, WikiAnswers, AG News, NPR, CC-News, WikiHow, SimpleWiki, SearchQA | in, capped per source |
| B, excluded | StackExchange (LoTTE, CQADupStack, BRIGHT), S2ORC and SPECTER (SCIDOCS, LitSearch), GooAQ (LoTTE search queries), Amazon QA/reviews (ESCI) | out: each collides with a shift test set |
| C. code | CodeSearchNet, CoSQA, StackOverflow QA (StaQC/ProCQA) | in; nothing from Apps, MBPP, WikiSQL train splits, HumanEval or DS-1000 |
| excluded | MIRACL, Mr.TyDi (RTEB beta set), medical-QA instruction data (ChatDoctor), non-English sets | out |

Recipe, from the literature on multi-source training (Arctic-Embed, Nomic, NV-Retriever,
the task-conflict/model-merging paper):
* **source-stratified batches**: every batch from one source, so in-batch negatives are not
  trivially cross-domain;
* **per-source caps** (E5 style): MS MARCO in full, other sources capped so PAQ/WikiAnswers
  cannot swamp; symmetric sets (Quora, NLI, WikiAnswers) kept small; the backbone's own
  query/document prefixes remain the task signal;
* **one teacher for every source** so distillation scores share a scale;
* **positive-aware negative filtering** (NV-Retriever TopK-PercPos): a mined candidate whose
  teacher score exceeds 95% of the positive's is dropped as a probable false negative.

Sequencing: **MS MARCO-only first** (a day with the existing pipeline; like-for-like with
SPLADE++ / SPLADE-v3; the control that says whether the mixture buys anything for a frozen
encoder plus a 1.2M-parameter head), then the mixture. With the mixture, the fair sparse
baseline is our own SPLADE trained on the same data; public SPLADE-v3 is a reference.

## Annotations (decided: regenerate for every source; models open)

What exists publicly: MS MARCO has 50 mined negatives per query and 160M MiniLM-L6
cross-encoder scores; NQ/TriviaQA/SQuAD have DPR's BM25 negatives; Quora has triplets;
everything else is pairs. Nothing but MS MARCO has teacher scores, so both hard negatives
and teacher scores are regenerated uniformly.

* **Miner (full retrieval over the unified passage store)**: candidates BM25 (always) plus
  a strong external dense model; not one of the three students alone, whose negatives
  would be its own confusions. Decision pending (see discussion in the session log).
* **Teacher (candidate scoring)**: one cross-encoder for all sources. Candidates
  considered: Qwen3-Reranker-0.6B (strongest per parameter, data undisclosed),
  Ettin-reranker-1B/400M (May 2026, documented public data, MSE-distilled from
  mxbai-rerank-large-v2, Apache), bge-reranker-v2-m3 (documented, weaker),
  ms-marco-MiniLM-L6 (MS MARCO only, fastest, weakest). Larger rerankers (4B, 8B) are 7–13x
  slower and out of budget on this machine. Decision pending.
* Budget: ~1.5M queries x (1 positive + 8–15 filtered candidates) = 12–24M pairs; a 0.6B–1B
  reranker scores ~100–200 pairs/s per TITAN RTX, i.e. 8–17 h on four cards.

## Test suites (decided in outline)

* **In-domain**: MS MARCO dev (full 8.8M corpus, not C1).
* **RTEB-open** (`Hyukkyu/rteb-open`): all 15 leaderboard sets including the six code sets
  (owner's decision: code shows generalisation; identifiers are units under D19), reported
  as text and code groups; CUREv1 English queries as a main shift corpus.
* **Terminology shift** (English, corpus >= 10k documents, >= 300 queries, embeddings known
  weak): scifact, nfcorpus, trec-covid (as in the pilots), FiQA-2018, SCIDOCS, LitSearch,
  TripClick (TripJudge), CUREv1; optionally LoTTE or Amazon ESCI as a long-tail entity test
  (their training sources are excluded above). BRIGHT left out (reasoning, not vocabulary).
* **Controlled insertion**: the rare / random / cluster / phrase splits at full scale for the
  recovery ratio against a retrained oracle.
* Nothing Wikipedia-QA based (NQ, HotpotQA, FEVER, Climate-FEVER, DBPedia) is tested once
  those sources are in training.

## Baselines

BM25; SPLADE++ and SPLADE-v3 (public); our own SPLADE on the same training data (mixture
arm); each backbone's dense retrieval; SPLADE fine-tuned with the domain terms added
(the practitioner's alternative to insertion), if GPU days allow. Index size and latency
reported alongside effectiveness.

## Open items

1. Miner model and teacher model (this discussion).
2. Per-source caps and the exact mixture sizes.
3. Which of LoTTE / ESCI, if either.
4. Whether cross-language insertion (CUREv1 es/fr) is in scope.
