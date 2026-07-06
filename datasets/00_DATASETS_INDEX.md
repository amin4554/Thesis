# Datasets — Index & Access

Every dataset in the thesis, what it's for, where to get it, and what's already in this folder.

> **What's included here vs. fetched on demand.** The `samples/` subfolder holds small samples/listings already pulled for inspection. The **full RAGTruth and FaithBench datasets ship in `../code_and_models/`** (cloned with their repos). The largest sets (full Trivia+, RAGBench) are downloaded on your machine with `get_datasets.py` (the Cowork sandbox blocks HuggingFace, so they can't be pulled from here).

## The four datasets that matter

### 1. RAGTruth — *training + in-domain* (QA used for training; summarization = task-shift target)
- **What:** ~18,000 RAG responses with response-level and word-level (span) hallucination labels, across QA, summarization, and data-to-text.
- **Already on disk:** full dataset at `../code_and_models/RAGTruth/dataset/` (`response.jsonl` ≈ 21 MB, `source_info.jsonl` ≈ 15 MB) + baselines at `../code_and_models/RAGTruth/baseline/`.
- **Sample/listing here:** `samples/ragtruth_dataset_listing.json`.
- **Source:** GitHub `ParticleMedia/RAGTruth` · paper [arXiv:2401.00396](https://arxiv.org/abs/2401.00396).
- **Use:** train the QA-only detector (QA split) and the all-task detector (full set); summarization split is the **task-shift** evaluation. Span labels exist here → span-level analysis lives on RAGTruth only.

### 2. Trivia+ — *domain-shift target* (primary)
- **What:** public, human-verified retrieval-augmented QA benchmark; the domain-shift evaluation at fixed task.
- **Already on disk:** `samples/triviaplus_dataset.parquet` (≈ 15 MB sample) + `samples/TRIVIA_PLUS_DATA_DETAILS.md`.
- **Source:** Chen et al. (2026), [arXiv:2605.11330](https://arxiv.org/abs/2605.11330). Paper source tar is in `../docs/`.
- **Use:** zero-shot domain-shift gap + the Trivia+ adaptation track. Flag
  contexts over 8,192 tokens and evaluate them with the registered
  chunk-and-aggregate procedure; also report the single-window diagnostic
  subset. Response-level labels only.

### 3. RAGBench (QA subset) — *fallback domain-shift target*
- **What:** large explainable RAG benchmark; labels are **model-generated** → used only as a robustness check if Trivia+ proves infeasible.
- **Get it:** HuggingFace `rungalileo/ragbench` (run `get_datasets.py`).
- **Source:** Friel, Belyi & Sanyal (2024), [arXiv:2407.11005](https://arxiv.org/abs/2407.11005).

### 4. FaithBench — *optional secondary summarization target*
- **What:** diverse, deliberately challenging summarization hallucination benchmark (the "~50% off-distribution" stat).
- **Already on disk:** `../code_and_models/FaithBench/` (`FaithBench.csv`, `data_for_release/`, `annot/`).
- **Sample/listing here:** `samples/faithbench_batch_1.json`, `samples/faithbench_data_listing.json`.
- **Source:** GitHub `vectara/FaithBench` · paper [arXiv:2410.13210](https://arxiv.org/abs/2410.13210).

### (also) TofuEval — MeetingBank — *optional secondary summarization target*
- **What:** topic-focused dialogue-summarization faithfulness; hard for LLM evaluators.
- **Sample here:** `samples/meetingbank_factual_eval_dev.csv` + `samples/tofueval_factual_consistency_listing.json`.
- **Get full:** GitHub `amazon-science/tofueval` · paper [arXiv:2402.13249](https://arxiv.org/abs/2402.13249).

## Label-mapping reminder (from the exposé)
RAGTruth word-level spans → aggregated to response level (hallucinated if ≥1 annotated span). Span-level analysis only where span labels exist (RAGTruth QA + summarization). Trivia+ / RAGBench = response level. Gray-area cases excluded from the primary analysis, reintroduced in a sensitivity analysis.

---
*Run `python get_datasets.py` on your own machine to pull the full Trivia+/RAGBench/TofuEval sets from HuggingFace/GitHub.*
