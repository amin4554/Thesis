# Frozen Protocol — locked Week 1 (30 Jun 2026)

The experimental protocol the thesis commits to. Everything here is fixed before
any model runs. **Changing any value below is a protocol change** — record the
date and reason in the changelog at the bottom. Machine-readable mirror:
[`src/protocol.py`](src/protocol.py).

---

## 1. Datasets & roles

| Dataset | Role | On disk | Loader |
|---|---|---|---|
| **RAGTruth** | Training + in-domain reference; **summarization** split = task-shift | `code_and_models/RAGTruth/dataset/` (full) | `load_ragtruth` |
| **Trivia+** | **Domain-shift** target (primary), fixed task = QA | `datasets/samples/triviaplus_dataset.parquet` (complete, 3,224 rows) | `load_triviaplus` |
| RAGBench QA | Domain-shift **fallback** (model-generated labels) | pull via `datasets/get_datasets.py` | (later) |
| FaithBench, TofuEval | **Optional** secondary summarization | FaithBench full on disk; TofuEval sample on disk | (optional) |

All datasets load into **one common schema** (see `src/data_prep.py` docstring):
`id, dataset, task_type, split, question, context, answer, prompt, spans,
response_label, is_gray, n_tokens, over_token_limit, model, meta`.

Prepared data is materialized to `output/prepared/{ragtruth,triviaplus}.jsonl`
(+ `stats.json`) by `python src/data_prep.py`.

## 2. Splits

- **RAGTruth:** native `train` (15,090) / `test` (2,700). No native dev — a dev
  slice for early stopping is carved from train with seed 13 when training (Week 2).
- **Trivia+:** native `train` (2,263) / `valid` (316) / `test` (645).
- Few-shot adaptation (Week 5) samples n ∈ {50, 100, 200, 500} from the Trivia+
  **train** primary set; evaluation is always on the held-out **test** primary set.
- The same adaptation ladder is evaluated for the RAGTruth summarization
  task-shift target, using training/development data only for adaptation and a
  held-out test set for final evaluation.

## 3. Label mapping (LOCKED)

- **Response-level (primary metric):** a response is **hallucinated (1)** iff
  **≥ 1 annotated span** (RAGTruth) / iff the binary annotator-aggregated label
  is 1 (Trivia+). This is exactly LettuceDetect's *example-level* definition
  (`models/evaluator.py`), so our response-level numbers are directly comparable
  to the reproduction reference.
- **Span-level:** only where char-offset span labels exist → **RAGTruth only**
  (QA + summarization). Trivia+ / RAGBench are **response-level only**.

## 4. Gray-area exclusion (LOCKED)

Gray-area cases are **excluded from the primary analysis** and **re-included in a
sensitivity analysis** (Week 3).

- **RAGTruth:** `quality != "good"` → 173 records (truncated / incorrect_refusal).
- **Trivia+:** `response_level_label == -1` → 139 records (no annotator majority).
  Note the dataset's `response_level_label_binary` already folds these into the
  positive class; we keep that binary label but flag the row `is_gray` so primary
  vs. sensitivity is a one-line switch (`primary_view()`).

## 5. Long-context handling (EXPOSÉ-ALIGNED) — measured Week 1

Flag inputs over ModernBERT's **8,192-token** window, counted on the full
`(prompt, answer)` pair with the real `answerdotai/ModernBERT-base` tokenizer
(no truncation). The registered primary analysis retains non-gray over-window
cases and evaluates them with a documented chunk-and-aggregate procedure. A
single-window subset is reported as a diagnostic, not substituted for the
primary set.

| Dataset | Total | > 8,192 tok (flagged) | % | max tokens |
|---|---:|---:|---:|---:|
| RAGTruth (all) | 17,790 | **0** | 0.0 % | 2,632 |
| **Trivia+ (all)** | 3,224 | **314** | **9.74 %** | 45,452 |

Trivia+ views by split (`primary` excludes gray only; `single-window` also
excludes over-limit rows):

| split | total | gray | > 8k | **primary** | **single-window** |
|---|---:|---:|---:|---:|---:|
| train | 2,263 | 98 | 255 | **2,165** | **1,927** |
| valid | 316 | 14 | 14 | **302** | **288** |
| **test** | 645 | 27 | 45 | **618** | **574** |

→ The **RQ1 domain-shift primary test set is 618 Trivia+ examples**. The 574
single-window cases remain a diagnostic subset. Encoder aggregation is locked
before final evaluation; NLI uses windowed + least-entailed aggregation.

## 6. Metrics

- **Response level (all datasets):** F1 (primary), precision, recall, balanced
  accuracy, AUROC, and true-positive rate at 5% false-positive rate. Report at
  the default threshold and a target-calibrated threshold (Week 3).
- **Span level (RAGTruth only):** span F1.
- **Cost (for RQ3):** USD per 1,000 checked responses, wall-clock seconds,
  GPU-seconds, and tokens — logged via `src/tracking.py`. Encoder/NLI inference
  uses a documented T4-class reference configuration; training cost is outside
  the matched-inference comparison.
- Secondary grids use **seeds {13, 42, 123}**. The pre-registered primary n=100
  adapted-detector-vs-judge comparison uses **five seeds
  {13, 42, 123, 2026, 3119828}**, with bootstrap 95% CIs.

**Reproduction target (anchor):** lettucedect-large on RAGTruth test →
**79.22 example-F1 / 58.93 span-F1**, tolerance ≈ ±1–2 F1 (Week 2).

## 7. LLM-as-judge — ⚠️ OPEN REGISTRATION DEVIATION

The exposé says the judge model, prices, and numeric reliability operating point
are pinned at registration. They are currently **UNPINNED**. This is disclosed
rather than silently filled with invented values; `src/protocol.py` blocks the
judge harness until the model and prices are recorded. The reliability threshold
must also be fixed before evaluation code is finalized.

| Field | Value | Status |
|---|---|---|
| Judge model | _TBD — decide before Week 4_ (`JUDGE_MODEL`) | ☐ pin |
| Sample cap (per arm) | **500** (`JUDGE_SAMPLE_CAP`) | ☑ decided 2026-06-30 |
| Price USD/1k input · output | _TBD — set with model and dated source_ | ☐ pin |
| Protocol | FaithJudge (`code_and_models/faithjudge/`) | fixed |
| Caching | all judge outputs cached, never re-run | fixed |
| Reliability minimum response F1 | _TBD — anchor to reproduction_ | ☐ pin |
| Reliability minimum TPR at 5% FPR | _TBD — anchor to reproduction_ | ☐ pin |

## 8. Reproducibility conventions

- Seeds **{13, 42, 123}** for secondary grids and five fixed seeds for the
  primary comparison; pinned packages in `requirements.txt`.
- One results log; every figure/table regenerated from it.
- Commit after every experiment; tag the snapshot used for final numbers.

---

## Changelog
- **2026-06-30** — Protocol frozen (Week 1). Datasets, splits, label mapping,
  gray-area rule, token lengths (measured: 314 Trivia+ over-window), metrics, seeds,
  results-log schema all locked. LLM judge **sample cap pinned at 500/arm**;
  **judge model deferred** (decide before Week 4 — guard blocks running until set).
- **2026-07-06** — Exposé-alignment correction before GitHub publication:
  restored five seeds for the pre-registered primary comparison, USD cost units,
  TPR@5% FPR, the threshold/linear-probe/LoRA ladder, and chunk-and-aggregate
  handling of long Trivia+ cases. The earlier 574-case filtered view is retained
  as a single-window diagnostic; the registered non-gray primary test set is
  618. The still-unpinned judge and reliability threshold are explicitly logged
  as open deviations.
