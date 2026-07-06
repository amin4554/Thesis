# Closing the Gap on a Budget — Lightweight Hallucination Detectors for RAG

Bachelor thesis code & experimental protocol.
**Author:** Amin Niaziardekani (3119828), B.Sc. Computer Science, SRH Berlin
**Supervisors:** Prakash Shreyaasri (1st), Joel Dokmegang (2nd)

## Research questions

- **RQ1** — How large is the zero-shot generalization gap when a RAGTruth-trained
  LettuceDetect detector is applied to a new domain (Trivia+) and a new task
  (summarization)?
- **RQ2** — How much of that gap does the registered adaptation ladder
  (threshold recalibration, frozen linear probe, LoRA; n = 50–500) close?
- **RQ3** — At a matched USD/compute budget, does the adapted lightweight detector
  beat an LLM-as-judge and an NLI baseline?

Full plan: [`THESIS_PLAN.md`](THESIS_PLAN.md) · Locked protocol:
[`FROZEN_PROTOCOL.md`](FROZEN_PROTOCOL.md).
Current implementation status and disclosed open decisions:
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Current status — 6 July 2026

- Week 1 is complete: common-schema data preparation, label mapping,
  gray-area handling, long-context flags, experiment logging, and unit tests.
- RAGTruth loads 17,790 responses. Trivia+ contains 645 test examples:
  **618 non-gray primary cases**, of which **574 fit one ModernBERT window**;
  longer primary cases are retained for the registered chunk-and-aggregate run.
- Week 2 training is ready to run on Kaggle P100 using the exact notebook recipe
  in [`code_and_models/REPRODUCE_TRAINING.md`](code_and_models/REPRODUCE_TRAINING.md).
- The LLM judge model/prices and numeric reliability operating point remain open
  and are explicitly blocked from execution until recorded.

## Repository layout

```
src/         protocol.py (frozen constants) · data_prep.py · tracking.py
tests/       unit tests for the label-mapping / gray-area / token filter
datasets/    dataset index + get_datasets.py (data itself is NOT in git)
code_and_models/  code index + setup_environment.sh (weights/repos NOT in git)
docs/        references.bib + paper index (PDFs NOT in git)
```

The **code** is versioned here; **data, model weights, and papers are not**
(see `.gitignore`). Fetch them with the scripts below or attach a Kaggle Dataset.

## Setup

```bash
pip install -r requirements.txt          # data-prep / analysis deps
# GPU machine (training/inference): also install a CUDA-matched torch, then:
bash code_and_models/setup_environment.sh   # pulls ModernBERT / LettuceDetect / NLI weights
python datasets/get_datasets.py             # pulls full Trivia+ / RAGBench / TofuEval
```

## Reproduce the Week-2 detector training (GPU)

The QA-only and all-task ModernBERT detectors are trained with the bundled
LettuceDetect code on a **Kaggle P100** (free tier; the local 8 GB GPU cannot
fit an fp32 full fine-tune of ModernBERT-large) — training compute is
intentionally outside the RQ3 cost comparison, which prices inference only.
The complete start-from-zero recipe (dataset upload, notebook cells, both
runs, targets 79.22 example-F1 / 58.93 span-F1, OOM/time-cap fallbacks):
[`code_and_models/REPRODUCE_TRAINING.md`](code_and_models/REPRODUCE_TRAINING.md).

## Reproduce the Week-1 data prep

```bash
python src/protocol.py          # prints where it found the data + output dirs
python -m pytest tests/ -q      # 8 tests: labels, views, gray areas, token flag
python src/data_prep.py         # writes output/prepared/*.jsonl + stats.json
```

### Running on Kaggle / Colab

Data is attached as a Kaggle Dataset, output goes to the writable working dir:

```python
import os
os.environ["THESIS_DATA_DIR"]   = "/kaggle/input/thesis-data"
os.environ["THESIS_OUTPUT_DIR"] = "/kaggle/working/output"
```

`src/protocol.py` then resolves all paths automatically — the same code runs on
the laptop and in the notebook.

## Reproducibility conventions

- Secondary grids use seeds **{13, 42, 123}**; the exposé-registered primary
  n=100 comparison uses **{13, 42, 123, 2026, 3119828}**.
- One results log (`output/results_log.{csv,sqlite}`) — one row per
  (method, train-set, eval-set, n, seed, metrics, cost).
- Each figure/table is regenerated from that log.

## Before committing or pushing

```bash
python scripts/check_repo.py
git status --short
```

The checker rejects tracked datasets, generated outputs, Office/PDF files,
model weights, caches, and private assistant/IDE state.
