# Main experiment — settings (living document, started 2026-09-08)

The pilot study (SUMMARY.md) is closed. This file records the main experiment's setting as
it is decided, with the reasoning, so that every later choice can be traced.

## Backbones (decided)

jina-embeddings-v5-text-small (Qwen3-0.6B decoder, last token, layer 12),
embeddinggemma-300m (Gemma 3 bidirectional, mean pooling + dense head, fp32, layer by
probe), snowflake-arctic-embed-l-v2.0 (XLM-R large, CLS, layer 24). Three pretraining
families, three pooling rules. Wiring and validation: `notes/backbones.md`; survey and
rationale: `notes/backbone_survey.md`. e5-base-v2 is the pilot contrast only.

## Training data (decided: tiers A and B)

Goal stated by the owner: use every public retrieval training set that does not overlap
the test suite, in particular RTEB-open (`Hyukkyu/rteb-open`).

| tier | sources | status |
|---|---|---|
| A. supervised retrieval | MS MARCO passage (502k queries, full), NQ, TriviaQA, SQuAD, HotpotQA, FEVER, ELI5, Quora duplicates, AllNLI | in |
| B. weakly supervised pairs | Yahoo Answers, PAQ, WikiAnswers, AG News, NPR, CC-News, WikiHow, SimpleWiki, SearchQA | in, capped per source |
| B, excluded | StackExchange (LoTTE, CQADupStack, BRIGHT), S2ORC and SPECTER (SCIDOCS, LitSearch), GooAQ (LoTTE search queries), Amazon QA/reviews (ESCI) | out: each collides with a shift test set |
| C. code | CodeSearchNet, CoSQA, StackOverflow QA (StaQC/ProCQA) | **out** (owner's decision 2026-09-08): tiers A and B only; the RTEB code sets are tested zero-shot with no code training data |
| excluded | MIRACL, Mr.TyDi (RTEB beta set), medical-QA instruction data (ChatDoctor), non-English sets | out |

The unified training set (counts from the sentence-transformers cards and BEIR train splits;
`documents` is the positive side unless a shared corpus exists; Wikipedia-derived sets share
one Wikipedia store, DPR's 21.0M passages, in the unified index):

| task | repo | languages | domain | license | queries | documents | qrels | public neg. |
|---|---|---|---|---|---:|---:|---:|:---:|
| MS MARCO passage | sentence-transformers/msmarco-hard-negatives | eng | web search | MS MARCO (non-commercial) | 502,939 | 8,841,823 | 532,761 | ✓ 50/query + CE scores |
| Natural Questions | sentence-transformers/natural-questions | eng | Wikipedia QA | cc-by-sa-3.0 | 100,231 | Wikipedia | 100,231 | ✓ DPR BM25 |
| TriviaQA | sentence-transformers/trivia-qa | eng | trivia QA | apache-2.0 | 73,346 | Wikipedia | 73,346 | ✓ DPR BM25 |
| SQuAD | sentence-transformers/squad | eng | Wikipedia RC | cc-by-sa-4.0 | 87,599 | 18,891 paragraphs | 87,599 | ✓ DPR BM25 |
| HotpotQA | sentence-transformers/hotpotqa, BeIR train | eng | multi-hop Wikipedia QA | cc-by-sa-4.0 | 85,000 | 5,233,329 | 170,000 | ✓ 20/query |
| FEVER | BeIR fever train | eng | fact verification | cc-by-sa-4.0 | 109,810 | 5,416,568 | 140,085 | ✗ |
| ELI5 | sentence-transformers/eli5 | eng | long-form QA (Reddit) | unspecified (Reddit) | 325,475 | 325,475 | 325,475 | ✗ |
| Quora duplicates | sentence-transformers/quora-duplicates | eng | duplicate questions, symmetric | other (Quora) | 149,263 | ~537k questions | 149,263 | ✓ 101,762 triplets |
| AllNLI | sentence-transformers/all-nli | eng | NLI, symmetric | cc-by-sa-4.0 / cc-by-3.0 | 314,315 | sentences | 314,315 | ✓ contradictions |
| Yahoo Answers | sentence-transformers/yahoo-answers | eng | community QA | Yahoo Webscope (non-commercial) | 1,198,260 | 1,198,260 | 1,198,260 | ✗ |
| PAQ | sentence-transformers/paq | eng | synthetic Wikipedia QA | cc-by-sa | 64,371,441 | Wikipedia | 64,371,441 | ✗ |
| WikiAnswers duplicates | sentence-transformers/wikianswers-duplicates | eng | duplicate questions, symmetric | unspecified | 761,379,586 pairs | questions | same | ✗ |
| AG News | sentence-transformers/agnews | eng | news title→description | unspecified | 1,157,745 | 1,157,745 | 1,157,745 | ✗ |
| NPR | sentence-transformers/npr | eng | news title→body | unspecified | 594,384 | 594,384 | 594,384 | ✗ |
| CC-News | sentence-transformers/ccnews | eng | news title→article | Common Crawl terms | 614,664 | 614,664 | 614,664 | ✗ |
| WikiHow | sentence-transformers/embedding-training-data | eng | how-to summary→text | cc-by-nc-sa-3.0 | 128,542 | 128,542 | 128,542 | ✗ |
| SimpleWiki | sentence-transformers/embedding-training-data | eng | Wikipedia↔Simple Wikipedia | cc-by-sa | 102,225 | 102,225 | 102,225 | ✗ |
| SearchQA | sentence-transformers/embedding-training-data | eng | Jeopardy QA with web snippets | unspecified | 117,220 | top-5 snippets | 117,220 | ✗ |

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

* **Miner (decided 2026-09-08)**: BM25 plus Qwen3-Embedding-0.6B over the unified passage
  store. Not one of the three students alone, whose negatives would be its own confusions.
  Per query: BM25 top-50 and dense top-100 with known positives removed, then a fixed
  sample of ~16 candidates (dense ranks 1–30 and 30–100, BM25 top-30) so the scoring budget
  is bounded; all of it goes to the teacher.
* **Teacher (decided 2026-09-08)**: Qwen3-Reranker-0.6B, a pointwise cross-encoder, for
  every source. Alternatives considered: Ettin-reranker-1B/400M (May 2026, documented
  public data, Apache, faster), bge-reranker-v2-m3, ms-marco-MiniLM-L6; 4B/8B rerankers are
  7–13x slower and out of budget here. Caveat to state in the paper: Qwen3-Reranker's
  training data is undisclosed, so contamination of the test suite through the teacher
  cannot be ruled out; Ettin-1B is the documented-data fallback if a reviewer objects.
  Details: score = logit(yes) − logit(no) for the loss (unbounded, no saturation);
  P(yes) for filtering; per-source retrieval instruction in the prompt (within-query
  losses are unaffected by scale shifts between sources); positive-aware filter in
  probability space (drop a candidate with P(yes) >= 0.95 x the positive's); drop queries
  whose positive scores below a floor (label noise in weakly supervised sources).
* Budget: ~1.5M queries x 17 pairs = ~25M pairs; ~100 pairs/s per TITAN RTX -> ~18 h on
  four cards; encoding a ~30M-passage store with Qwen3-Embedding-0.6B ~12 h on four cards.

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

1. Per-source caps and the exact mixture sizes.
2. Which of LoTTE / ESCI, if either.
3. Whether cross-language insertion (CUREv1 es/fr) is in scope.
