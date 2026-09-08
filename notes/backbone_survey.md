# Backbone candidates for the main experiment: open dense retrievers under 1B parameters

Surveyed 2026-09-08 from model cards, papers and the MTEB/RTEB leaderboards. Scores are as
published by each model's authors; suites differ, so compare within a column only.
`MTEB(eng v2)` is the 41-task English mean; `MTEB-v2 R` its 10-task retrieval mean
(nDCG@10); `BEIR-15` the classic 15-dataset nDCG@10 mean; `MMTEB` the multilingual v2
mean; `RTEB` the retrieval-only public leaderboard mean (has private sets; 0–100 scale).

## Table 1 — candidates

| # | model | org, release | params | base / attention / layers × width | pooling; query prompt | MTEB(eng v2) | retrieval score (suite) | MMTEB | license |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-Embedding-0.6B | Alibaba, Jun 2025 | 595M | Qwen3-0.6B, causal decoder, 28 × 1024 | last token; `Instruct: … Query:` | 70.70 | 61.83 (MTEB-v2 R); 55.5 (BEIR-15, per Jina paper) | 64.34 | Apache-2.0 |
| 2 | Octen-Embedding-0.6B | Octen, Jan 2026 | 595M | LoRA fine-tune of #1, same stack | last token; same as #1 | not published | 72.4 (RTEB public; = Nemotron-3-1B 72.4, Qwen3-8B 73.1) | not published | Apache-2.0 |
| 3 | jina-embeddings-v5-text-small | Jina, Feb 2026 | 677M | Qwen3-0.6B-Base + task LoRAs, causal, 28 × 1024 | last token; `Query:` / `Document:` | 71.7 | 56.67 (BEIR-15); 66.8 (RTEB subset in paper) | 67.0–67.7 | CC BY-NC 4.0 |
| 4 | harrier-oss-v1-0.6b | Microsoft Bing, Apr 2026 | 596M | Qwen3 stack (28 × 1024, Qwen3 vocab), causal | last token + L2; `Instruct: … Query:` | not on card | not on card | 69.0 (#1 sub-1B) | MIT |
| 5 | KaLM-embedding-multilingual-mini-instruct-v2.5 | HIT/Tencent, Nov 2025 | 494M | Qwen2-0.5B, **bidirectional**, 24 × 896 | mean; `Instruct: … Query:` | 70.1–71.3 | 55.0 (BEIR-15, per Jina paper) | 60.1 | Apache-2.0 |
| 6 | embeddinggemma-300m | Google, Sep 2025 | 308M | Gemma 3, **bidirectional** encoder (T5Gemma init), 24 × 768 | mean; `task: … \| query:` | 69.67 | 62.5 (MMTEB retrieval); BEIR not published | 61.15 | Gemma terms |
| 7 | snowflake-arctic-embed-l-v2.0 | Snowflake, Dec 2024 | 568M | XLM-R large, bidirectional, 24 × 1024 | CLS; `query:` | n/a | 55.65 (BEIR-15) | n/a | Apache-2.0 |
| 8 | snowflake-arctic-embed-m-v2.0 | Snowflake, Dec 2024 | 305M | gte-multilingual-base (BERT-like), 12 × 768 | CLS; `query:` | n/a | 58.4 (MTEB-v2 R, per Granite card); 55.4 (BEIR-15) | n/a | Apache-2.0 |
| 9 | gte-modernbert-base | Alibaba, Jan 2025 | 149M | ModernBERT-base, bidirectional, 22 × 768 | CLS; none | 64.38 (old MTEB) | 57.0 (MTEB-v2 R); 55.33 (BEIR-15) | English only | Apache-2.0 |
| 10 | granite-embedding-english-r2 | IBM, Aug 2025 | 149M | ModernBERT-base, bidirectional, 22 × 768 | CLS; none | 62.8 (old MTEB) | 56.4 (MTEB-v2 R); 53.1 (BEIR-15) | English only | Apache-2.0 |
| 11 | nomic-embed-text-v2-moe | Nomic, Feb 2025 | 475M (305M active) | XLM-R base + MoE, bidirectional, 12 × 768 | mean; `search_query:` | n/a | 52.86 (BEIR-15) | n/a | Apache-2.0 |
| 12 | bge-m3 (dense) | BAAI, Jan 2024 | 568M | XLM-R large, bidirectional, 24 × 1024 | CLS; none | n/a | 48.8 (BEIR-15, dense only) | 59.56 | MIT |
| 13 | multilingual-e5-large-instruct | Microsoft, Feb 2024 | 560M | XLM-R large, bidirectional, 24 × 1024 | mean; `Instruct: … Query:` | 65.53 | 57.1 (MMTEB retrieval) | 63.22 | MIT |
| 14 | stella_en_400M_v5 | NovaSearch, Jul 2024 | 435M | gte-large-en-v1.5 (BERT + RoPE), 24 × 1024 | mean; `Instruct: … Query:` | n/a (old MTEB 70.1) | 58.97 (old MTEB retrieval) | English only | MIT |
| 15 | e5-base-v2 (pilot backbone) | Microsoft, May 2023 | 109M | BERT-base, bidirectional, 12 × 768 | mean; `query:` | n/a | 49.7 (MTEB-v2 R); 50.3 (BEIR-15) | English only | MIT |

Just over the limit, for reference: **Nemotron-3-Embed-1B** (NVIDIA, Jul 2026; 1.14B; pruned
Ministral-3-3B made bidirectional; mean pooling; 2048-d; RTEB 72.4, MMTEB retrieval 71.0;
OpenMDW), gte-Qwen2-1.5B-instruct (1.5B), stella_en_1.5B_v5 (1.5B), inf-retriever-v1-1.5b.
Closed weights (unusable here): gemini-embedding-001, voyage-3.5, text-embedding-3-large,
Seed1.6-embedding.

## Table 2 — the three we have already run, dense nDCG@10 on the domain corpora (`scripts/94_domain_baselines.py`)

| system | scifact | nfcorpus | trec-covid |
|---|---|---|---|
| BM25 | 0.662 | 0.303 | 0.627 |
| SPLADE++ | 0.679 | 0.342 | 0.803 |
| SPLADE-v3 | 0.686 | 0.351 | 0.826 |
| e5-base-v2 dense | 0.701 | 0.357 | 0.790 |
| Octen-0.6B dense | 0.708 | 0.365 | 0.830 |
| jina-v5-small dense | 0.744 | 0.391 | 0.828 |

## What matters for our method, beyond the leaderboard

* **The strongest sub-1B retrievers are one architecture.** #1–#4 are all the Qwen3-0.6B
  stack (28 layers, 1024 wide, byte-level BPE, last-token pooling); they differ in training
  data, adapters and license. Choosing three of them tests one backbone three times.
* **Bidirectional alternatives that are competitive**: EmbeddingGemma (#6, Gemma 3 made
  bidirectional, 768 wide), KaLM v2.5 (#5, Qwen2-0.5B made bidirectional, 896 wide),
  arctic-embed-l-v2.0 (#7, XLM-R large), and the ModernBERT pair (#9, #10, 149M).
* **Pilot evidence on architecture**: the one BERT-family backbone we ran (e5) turned
  inserted entries into hubs; the two Qwen3 decoders did not. One observation each, so
  spanning families in the main experiment is what would make that a finding.
* **Width sets the head size** (768 / 896 / 1024 / 2048) and **the tokenizer sets the
  word-unit rule** (WordPiece: BERT; byte-level BPE: Qwen, ModernBERT; SentencePiece:
  Gemma, XLM-R). All are handled by `dvlsr/encoders.py` except SentencePiece, which needs
  the `trim_punct`-style boundary check verified.
* **Licenses**: jina v5 is non-commercial (fine for research, not for a demo release);
  EmbeddingGemma is under Gemma terms; the rest are Apache/MIT.
* **Layer choice is per backbone** (Pilot M): Octen peaked at its last layer, jina at
  8–12, so the layer probe (`scripts/95_layer_probe.py`) must be re-run for every pick.

## A selection that spans families (recommendation, not a decision)

1. One Qwen3-0.6B decoder: **Harrier-0.6b** (MIT, top sub-1B on MMTEB) or
   **Qwen3-Embedding-0.6B** (the canonical, Apache) or keep **jina-v5-small** (already
   wired, best on our corpora, non-commercial).
2. **EmbeddingGemma-300m**: the strongest bidirectional model under 500M, a different
   family and tokenizer, and the cheapest to train.
3. **snowflake-arctic-embed-l-v2.0** (XLM-R, 1024 wide, Apache) or **KaLM v2.5**
   (bidirectional Qwen2) as the third family; e5-base-v2 stays as the pilot contrast only
   if a fourth run is affordable.
