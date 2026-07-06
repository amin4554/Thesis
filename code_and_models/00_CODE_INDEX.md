# Code & Models — Index

Repositories and model checkpoints for the thesis. Repos that clone cleanly from GitHub are **included in this folder** (`.git` history stripped to save space). Model weights and the large ModernBERT repo live on HuggingFace/GitHub — pull them with `setup_environment.sh`.

> The Cowork sandbox blocks HuggingFace, so model weights couldn't be downloaded from here. Everything below is reproducible with the commands in `setup_environment.sh`.

## Included in this folder

### `LettuceDetect/` — the detector under study
- Full package (`lettucedetect/`: datasets, detectors, models, preprocess, prompts) + demo + docs.
- Source: GitHub `KRLabsOrg/LettuceDetect` · paper [arXiv:2502.17125](https://arxiv.org/abs/2502.17125).
- Install: `pip install lettucedetect` (PyPI, v0.2.0) **or** `pip install -e ./LettuceDetect`.
- Checkpoints (HuggingFace): `KRLabsOrg/lettucedect-base-modernbert-en-v1` (≈150M) and `KRLabsOrg/lettucedect-large-modernbert-en-v1` (≈396M). Your thesis trains **your own** QA-only and all-task models, but these are the reproduction reference (target: 79.22 example-F1 / 58.93 span-F1 on RAGTruth test).

### `RAGTruth/` — training + in-domain data and baselines
- `dataset/response.jsonl` + `dataset/source_info.jsonl` = the **full RAGTruth corpus**; `baseline/` = reference detection scripts.
- Source: GitHub `ParticleMedia/RAGTruth` · paper [arXiv:2401.00396](https://arxiv.org/abs/2401.00396).

### `FaithBench/` — optional secondary summarization target
- `FaithBench.csv`, `data_for_release/`, `annot/`, `scripts/`.
- Source: GitHub `vectara/FaithBench` · paper [arXiv:2410.13210](https://arxiv.org/abs/2410.13210).

### `faithjudge/` — LLM-as-a-judge baseline protocol (code only)
- `eval.py`, `generate_responses.py`, `prompt_templates.py`, `generate_table.py`, README. The repo's large precomputed `eval_results/` and `generated_outputs/` (~350 MB) were **omitted** — re-clone the full repo if you need them (see `_OMITTED_RESULTS_NOTE.txt`).
- Source: GitHub `vectara/faithjudge` · paper [arXiv:2505.04847](https://arxiv.org/abs/2505.04847).

## Pulled on your machine (in `setup_environment.sh`)

| Component | Where | Role |
|-----------|-------|------|
| **ModernBERT** encoder | GitHub `AnswerDotAI/ModernBERT`; weights `answerdotai/ModernBERT-base` / `-large` | Backbone of LettuceDetect (8k context). Repo not bundled (large). [arXiv:2412.13663](https://arxiv.org/abs/2412.13663) |
| **NLI baseline** | HuggingFace, e.g. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | DeBERTa-v3 MNLI/ANLI; windowed + least-entailed aggregation for long contexts. |
| **LettuceDetect weights** | `KRLabsOrg/lettucedect-{base,large}-modernbert-en-v1` | Reproduction reference checkpoints. |
| **LLM judge model** | Provider API (pinned at registration) | FaithJudge arm; price + sample cap fixed in advance. |

---
*See `setup_environment.sh` for a one-shot install + download script.*
