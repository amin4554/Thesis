"""Experiment tracking — one results log for the whole thesis (Week 1 deliverable).

Every experiment (reproduction, zero-shot transfer, few-shot adaptation, every
baseline) appends ONE row per (model, train-set, eval-set, n, seed) with both
accuracy metrics and cost. Figures/tables are regenerated from this log, so the
schema is fixed once here and never edited ad hoc.

Two synchronized sinks:
  - CSV   (output/results_log.csv)   — easy to eyeball / load into pandas.
  - SQLite(output/results_log.sqlite)— easy to query for the Pareto plot.

Usage:
    from tracking import log_result, RUN_FIELDS
    log_result(
        method="lettucedetect-large", train_set="ragtruth-all", eval_set="ragtruth-test",
        task="qa", n_train=None, seed=42,
        example_f1=79.1, span_f1=58.8, balanced_acc=..., auroc=...,
        cost_eur=..., wall_clock_s=..., n_tokens=..., n_eval=2700,
        phase="P0", notes="reproduction",
    )
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import sqlite3
from typing import Optional

import protocol as P

# Fixed column order — adding a field is a schema change; append, never reorder.
RUN_FIELDS = [
    "timestamp",      # ISO8601, filled automatically
    "phase",          # P0..P6
    "method",         # detector / nli / judge / heuristic name
    "train_set",      # e.g. ragtruth-qa, ragtruth-all, triviaplus-n200, none
    "eval_set",       # e.g. ragtruth-test, triviaplus-test, ragtruth-summ
    "task",           # qa | summarization | data2txt
    "n_train",        # few-shot n (int) or empty for zero-shot/full
    "seed",           # 13 | 42 | 123
    # --- accuracy ---
    "example_f1",     # response-level F1 (primary)
    "span_f1",        # span-level F1 (RAGTruth only; empty elsewhere)
    "balanced_acc",
    "auroc",
    "precision",
    "recall",
    "threshold",      # decision threshold used
    "n_eval",         # number of evaluated examples
    # --- cost (for the RQ3 Pareto plot) ---
    "cost_eur",       # total € for this run's evaluation
    "wall_clock_s",   # total seconds
    "gpu_seconds",    # GPU-seconds (training+inference as applicable)
    "n_tokens",       # total tokens billed (judge / API methods)
    # --- provenance ---
    "git_commit",
    "notes",
]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _ensure_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RUN_FIELDS)


def _ensure_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cols = ", ".join(f'"{c}" TEXT' for c in RUN_FIELDS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS runs ({cols})")
    conn.commit()
    return conn


def log_result(
    *,
    method: str,
    eval_set: str,
    seed: int,
    phase: str = "",
    train_set: str = "",
    task: str = "",
    n_train: Optional[int] = None,
    example_f1: Optional[float] = None,
    span_f1: Optional[float] = None,
    balanced_acc: Optional[float] = None,
    auroc: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    threshold: Optional[float] = None,
    n_eval: Optional[int] = None,
    cost_eur: Optional[float] = None,
    wall_clock_s: Optional[float] = None,
    gpu_seconds: Optional[float] = None,
    n_tokens: Optional[int] = None,
    git_commit: str = "",
    notes: str = "",
    csv_path: str = P.RESULTS_LOG_CSV,
    db_path: str = P.RESULTS_LOG_DB,
) -> dict:
    """Append one experiment row to both the CSV and the SQLite log."""
    if seed not in P.SEEDS:
        # Not fatal, but warn: reported numbers must use the fixed seeds.
        print(f"[tracking] warning: seed {seed} is not one of {P.SEEDS}")

    row = {
        "timestamp": _now(),
        "phase": phase,
        "method": method,
        "train_set": train_set,
        "eval_set": eval_set,
        "task": task,
        "n_train": "" if n_train is None else n_train,
        "seed": seed,
        "example_f1": example_f1,
        "span_f1": span_f1,
        "balanced_acc": balanced_acc,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "n_eval": n_eval,
        "cost_eur": cost_eur,
        "wall_clock_s": wall_clock_s,
        "gpu_seconds": gpu_seconds,
        "n_tokens": n_tokens,
        "git_commit": git_commit,
        "notes": notes,
    }
    row = {k: ("" if v is None else v) for k, v in row.items()}

    _ensure_csv(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[c] for c in RUN_FIELDS])

    conn = _ensure_db(db_path)
    placeholders = ", ".join("?" for _ in RUN_FIELDS)
    conn.execute(
        f"INSERT INTO runs VALUES ({placeholders})", [str(row[c]) for c in RUN_FIELDS]
    )
    conn.commit()
    conn.close()
    return row


if __name__ == "__main__":
    # Smoke test: log a dummy row and read it back.
    log_result(
        phase="P0",
        method="_smoketest",
        train_set="ragtruth-all",
        eval_set="ragtruth-test",
        task="qa",
        seed=42,
        example_f1=0.0,
        notes="tracking.py smoke test row",
    )
    print(f"OK — wrote a smoke-test row to:\n  {P.RESULTS_LOG_CSV}\n  {P.RESULTS_LOG_DB}")
