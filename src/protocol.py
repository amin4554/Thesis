"""Frozen experimental protocol — single source of truth for the thesis.

Everything here is *locked on day 1* (Week 1 of THESIS_PLAN.md) and must not
drift afterwards. If a value here changes, it is a protocol change and must be
recorded in FROZEN_PROTOCOL.md with a date and reason.

Imported by data_prep.py, tracking.py, and (later) train.py / evaluate.py so
that every script shares the same seeds, paths, label rules, and token budget.

--- Portable paths (laptop <-> Kaggle <-> fresh clone) ---------------------- #
Code lives in this repo (``GitHubRepo/``). The large data does NOT — it is kept
out of git. ``DATA_ROOT`` is resolved at import time so the *same* code runs
unchanged in three places:

  1. Kaggle / Colab : set env ``THESIS_DATA_DIR=/kaggle/input/<your-dataset>``
                      and ``THESIS_OUTPUT_DIR=/kaggle/working/output``.
  2. Your laptop    : data sits in the parent ``thesis/`` folder (auto-detected).
  3. Fresh clone    : after running get_datasets.py / setup_environment.sh the
                      data lands inside the repo (auto-detected).
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)  # the GitHubRepo/ folder


def _has_ragtruth(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "code_and_models", "RAGTruth", "dataset"))


def _resolve_data_root() -> str:
    """Where the large datasets live. Override with THESIS_DATA_DIR."""
    env = os.environ.get("THESIS_DATA_DIR")
    if env:
        return os.path.abspath(env)
    if _has_ragtruth(REPO_ROOT):            # data fetched into the clone
        return REPO_ROOT
    parent = os.path.dirname(REPO_ROOT)     # laptop: code in thesis/GitHubRepo, data in thesis/
    if _has_ragtruth(parent):
        return parent
    return REPO_ROOT                        # fallback (scripts will error clearly if data is missing)


def _resolve_output_dir() -> str:
    """Where to write prepared data / results / figures. Override with THESIS_OUTPUT_DIR.
    On Kaggle this MUST point under /kaggle/working (the repo dir is read-only)."""
    env = os.environ.get("THESIS_OUTPUT_DIR")
    return os.path.abspath(env) if env else os.path.join(REPO_ROOT, "output")


DATA_ROOT = _resolve_data_root()
OUTPUT_DIR = _resolve_output_dir()

RAGTRUTH_DIR = os.path.join(DATA_ROOT, "code_and_models", "RAGTruth", "dataset")
RAGTRUTH_RESPONSE = os.path.join(RAGTRUTH_DIR, "response.jsonl")
RAGTRUTH_SOURCE_INFO = os.path.join(RAGTRUTH_DIR, "source_info.jsonl")

# The sample parquet IS the complete Trivia+ benchmark (3,224 rows). The
# get_datasets.py header documents this; a clone to code_and_models/TriviaPlus/
# would be byte-identical, so either location is read.
TRIVIAPLUS_PARQUET = os.path.join(DATA_ROOT, "datasets", "samples", "triviaplus_dataset.parquet")
TRIVIAPLUS_PARQUET_FALLBACK = os.path.join(
    DATA_ROOT, "code_and_models", "TriviaPlus", "triviaplus_dataset.parquet"
)

PREPARED_DIR = os.path.join(OUTPUT_DIR, "prepared")
RESULTS_LOG_CSV = os.path.join(OUTPUT_DIR, "results_log.csv")
RESULTS_LOG_DB = os.path.join(OUTPUT_DIR, "results_log.sqlite")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# Three seeds cover secondary/exploratory grids.  The exposé pre-registers at
# least five seeds for the primary n=100 adapted-detector-vs-judge comparison.
SEEDS = (13, 42, 123)
PRIMARY_COMPARISON_SEEDS = (13, 42, 123, 2026, 3119828)

# --------------------------------------------------------------------------- #
# Model / tokenizer
# --------------------------------------------------------------------------- #
# ModernBERT backbone window. The detector tokenizes (context, answer) with
# truncation="only_first", so the *answer* is never truncated; only the context
# is. The Trivia+ length filter below is applied to the full (prompt, answer)
# pair to mirror exactly what the model sees.
MODERNBERT_TOKENIZER = "answerdotai/ModernBERT-base"
MAX_TOKENS = 8192

# Reproduction-reference checkpoints (we still train our own QA-only / all-task
# models; these only anchor the in-domain reproduction target).
LETTUCEDETECT_BASE = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
LETTUCEDETECT_LARGE = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
# Reproduction targets on RAGTruth test (lettucedect-large), ±1-2 F1 tolerance.
REPRO_TARGET_EXAMPLE_F1 = 79.22
REPRO_TARGET_SPAN_F1 = 58.93

# NLI baseline (Week 4).
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# --------------------------------------------------------------------------- #
# Label mapping & gray-area policy  (LOCKED — see FROZEN_PROTOCOL.md)
# --------------------------------------------------------------------------- #
# Rule: a response is hallucinated (response_label = 1) iff >= 1 annotated span
# (RAGTruth) / iff the binary annotator-aggregated label is 1 (Trivia+). This is
# exactly LettuceDetect's "example-level" definition (evaluator.py:165), so our
# response-level numbers are directly comparable to the reproduction reference.
RESPONSE_POSITIVE_IF_ANY_SPAN = True

# Gray-area cases are EXCLUDED from the primary analysis and re-included in a
# sensitivity analysis (Week 3).
#   - RAGTruth : quality != "good"  (truncated / incorrect_refusal -> degenerate)
#   - Trivia+  : response_level_label == -1  (no annotator majority / tie)
RAGTRUTH_GOOD_QUALITY = "good"
TRIVIAPLUS_GRAY_LABEL = -1

# Canonical task-type names in the common schema.
TASK_QA = "qa"
TASK_SUMMARIZATION = "summarization"
TASK_DATA2TXT = "data2txt"

# RAGTruth source_info.task_type -> canonical task name.
RAGTRUTH_TASK_MAP = {
    "QA": TASK_QA,
    "Summary": TASK_SUMMARIZATION,
    "Data2txt": TASK_DATA2TXT,
}

# --------------------------------------------------------------------------- #
# LLM-as-judge — PINNED AT REGISTRATION (Week 1 must-do).
# Sample cap DECIDED (Week 1) = 500/arm. Judge MODEL deferred (decide before
# Week 4). All judge outputs are cached and never re-run (risk register).
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "PINME"          # DECIDE LATER — e.g. a frontier or mid-tier judge
JUDGE_SAMPLE_CAP = 500         # DECIDED (Week 1): 500 evaluations per arm, cached
JUDGE_PRICE_PER_1K_INPUT = 0.0   # USD per 1k input tokens, set with the model
JUDGE_PRICE_PER_1K_OUTPUT = 0.0  # USD per 1k output tokens, set with the model
JUDGE_PINNED = False           # flip to True once the model + prices above are set

# The exposé also requires a numeric reliability operating point anchored to
# the reproduced in-domain result. Keep these unset until the supervisor-facing
# record contains the actual values; evaluation code must call the guard below.
RELIABILITY_MIN_RESPONSE_F1 = None
RELIABILITY_MIN_TPR_AT_5_FPR = None


def assert_judge_pinned() -> None:
    """Guard used by the Week-4 judge harness so an unpinned judge can't run."""
    if (
        not JUDGE_PINNED
        or JUDGE_MODEL == "PINME"
        or JUDGE_SAMPLE_CAP <= 0
        or JUDGE_PRICE_PER_1K_INPUT <= 0
        or JUDGE_PRICE_PER_1K_OUTPUT <= 0
    ):
        raise RuntimeError(
            "LLM judge is not pinned. Set JUDGE_MODEL, the prices, and "
            "JUDGE_PINNED=True in src/protocol.py before running the judge "
            "baseline (and record it in FROZEN_PROTOCOL.md)."
        )


def assert_reliability_pinned() -> None:
    """Block final evaluation while the registered operating point is unset."""
    if RELIABILITY_MIN_RESPONSE_F1 is None or RELIABILITY_MIN_TPR_AT_5_FPR is None:
        raise RuntimeError(
            "Reliability thresholds are not pinned. Set the minimum response "
            "F1 and TPR@5%FPR in src/protocol.py and record them in "
            "FROZEN_PROTOCOL.md before final evaluation."
        )


def describe_paths() -> str:
    """Handy one-liner for notebooks: confirms where code thinks data/output are."""
    return (
        f"REPO_ROOT   = {REPO_ROOT}\n"
        f"DATA_ROOT   = {DATA_ROOT}  (THESIS_DATA_DIR overrides)\n"
        f"OUTPUT_DIR  = {OUTPUT_DIR}  (THESIS_OUTPUT_DIR overrides)\n"
        f"RAGTruth on disk: {os.path.exists(RAGTRUTH_RESPONSE)}\n"
        f"Trivia+ on disk : {os.path.exists(TRIVIAPLUS_PARQUET) or os.path.exists(TRIVIAPLUS_PARQUET_FALLBACK)}"
    )


if __name__ == "__main__":
    print(describe_paths())
