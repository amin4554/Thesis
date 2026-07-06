# Reference Papers — Annotated Index

All 16 works cited in the exposé *"Closing the Gap on a Budget: Few-Shot Domain Adaptation and Cost–Accuracy Trade-offs of Lightweight Hallucination Detectors for RAG."*

> **Note on the PDFs:** I could not fetch the PDFs from inside the Cowork sandbox — its network only allows a short allowlist (github.com, pypi) and blocks arXiv and every academic mirror. To get all the PDFs into this folder, run **`download_papers.sh`** (macOS/Linux) or **`download_papers.py`** (any OS with Python) from this folder; each pulls all 16 PDFs from arXiv in one go. A ready-to-use **`references.bib`** is also here for your thesis.

Papers are grouped by the role they play in the thesis.

## Load-bearing — the core of the method
| # | Paper | arXiv / DOI | Role |
|---|-------|-------------|------|
| 1 | **Kovács & Recski (2025)** — LettuceDetect: A Hallucination Detection Framework for RAG | [2502.17125](https://arxiv.org/abs/2502.17125) | The detector under study (ModernBERT token classifier; 79.22% F1 on RAGTruth). |
| 2 | **Niu et al. (2024)** — RAGTruth: A Hallucination Corpus for Trustworthy RAG | [2401.00396](https://arxiv.org/abs/2401.00396) | Training + in-domain dataset (QA, summarization, data-to-text; word-level labels). |
| 3 | **Chen et al. (2026)** — Rethinking Evaluation for LLM Hallucination Detection (Trivia+) | [2605.11330](https://arxiv.org/abs/2605.11330) | Source of **Trivia+**, the domain-shift target; benchmark-desiderata argument. |
| 4 | **Tamber et al. (2025)** — Benchmarking LLM Faithfulness in RAG (FaithJudge) | [2505.04847](https://arxiv.org/abs/2505.04847) | The **FaithJudge** LLM-as-a-judge protocol used as a baseline. |
| 5 | **Warner et al. (2024)** — ModernBERT | [2412.13663](https://arxiv.org/abs/2412.13663) | The encoder backbone (8k context) LettuceDetect is built on. |
| 6 | **Hu et al. (2021)** — LoRA: Low-Rank Adaptation | [2106.09685](https://arxiv.org/abs/2106.09685) | Rung 3 of the adaptation ladder. |

## The research gap — why this study is needed
| # | Paper | arXiv / DOI | Role |
|---|-------|-------------|------|
| 7 | **Dubanowska et al. (2025)** — Representation-based detectors fail OOD | [2509.19372](https://arxiv.org/abs/2509.19372) | Cross-task failure + spurious-correlation finding; motivates the shortcut probe. |
| 8 | **Janiak et al. (2025)** — The Illusion of Progress | [2508.08285](https://arxiv.org/abs/2508.08285) | Length-heuristic baseline; "no ROUGE" decision; evaluation-artifact warning. |
| 9 | **Bao et al. (2025)** — FaithBench | [2410.13210](https://arxiv.org/abs/2410.13210) | The "~50% / near-chance off-distribution" stat; optional secondary target. |
| 10 | **Tang et al. (2024)** — TofuEval | [2402.13249](https://arxiv.org/abs/2402.13249) | Dialogue-summary difficulty; optional secondary target (MeetingBank). |
| 11 | **Karbasi et al. (2025)** — (Im)possibility of Automated Hallucination Detection | [2504.17004](https://arxiv.org/abs/2504.17004) | Theory: labeled incorrect examples matter → motivates the adaptation question. |
| 12 | **Valentin et al. (2024)** — Cost-Effective Hallucination Detection | [2407.21424](https://arxiv.org/abs/2407.21424) | Prior cost-aware work this thesis extends to the lightweight-encoder class. |

## Foundations & supporting
| # | Paper | arXiv / DOI | Role |
|---|-------|-------------|------|
| 13 | **Lewis et al. (2020)** — Retrieval-Augmented Generation | [2005.11401](https://arxiv.org/abs/2005.11401) | Defines RAG. |
| 14 | **Li et al. (2023)** — HaluEval | [2305.11747](https://arxiv.org/abs/2305.11747) | Early evidence LLMs hallucinate. |
| 15 | **Huang et al. (2025)** — Survey on Hallucination in LLMs | [10.1145/3703155](https://doi.org/10.1145/3703155) · arXiv [2311.05232](https://arxiv.org/abs/2311.05232) | Risk framing (law/medicine/finance). ACM TOIS; arXiv version is the downloadable copy. |
| 16 | **Friel, Belyi & Sanyal (2024)** — RAGBench | [2407.11005](https://arxiv.org/abs/2407.11005) | Pre-specified fallback domain-shift target. |

---
*Generated for Amin Niaziardekani's B.Sc. thesis proposal. 16 references total.*
