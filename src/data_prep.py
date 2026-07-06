"""Common-schema data preparation for the thesis (Week 1 deliverable).

Loads every dataset into one record schema, applies the locked label-mapping
(span annotations -> response-level), flags gray-area and over-window records,
then writes prepared JSONL + a stats report. Over-window Trivia+ records stay in
the registered primary set and are handled later by chunk-and-aggregate
evaluation; ``single_window_view`` exposes the 8,192-token diagnostic subset.

Common record schema (one dict per generated response):

    id                 str   unique within (dataset, split)
    dataset            str   "ragtruth" | "triviaplus"
    task_type          str   "qa" | "summarization" | "data2txt"
    split              str   "train" | "valid" | "test"
    question           str|None   user request (None for summarization/data2txt)
    context            str   grounding evidence the answer must be faithful to
    answer             str   generated response being checked
    prompt             str   full detector input (question front, then context)
    spans              list|None   [{start,end,label}] char offsets in answer;
                                    None when no span labels exist (Trivia+)
    response_label     int   1 = hallucinated, 0 = supported  (LOCKED rule)
    is_gray            bool  gray-area -> excluded from primary, kept for sensitivity
    n_tokens           int|None   ModernBERT token count of (prompt, answer)
    over_token_limit   bool  n_tokens > MAX_TOKENS
    model              str   the LLM that generated `answer`
    meta               dict  source-specific provenance

Usage:
    python src/data_prep.py                 # prepare everything, write to output/prepared/
    python src/data_prep.py --no-tokens     # skip token counting (fast, no tokenizer)
    python src/data_prep.py --dataset ragtruth
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Iterable, Optional

import pandas as pd

import protocol as P

# --------------------------------------------------------------------------- #
# Tokenizer (lazy — only loaded if we actually count tokens, so the module
# imports fine on machines without `transformers`).
# --------------------------------------------------------------------------- #
_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(P.MODERNBERT_TOKENIZER)
    return _TOKENIZER


def count_tokens(prompt: str, answer: str) -> int:
    """Token length of the detector input, exactly as HallucinationDataset
    tokenizes it: tokenizer(prompt, answer) with special tokens. We do NOT
    truncate here so we can measure the *true* length and decide exclusion."""
    tok = get_tokenizer()
    enc = tok(prompt, answer, add_special_tokens=True)
    return len(enc["input_ids"])


# --------------------------------------------------------------------------- #
# Label mapping (LOCKED)
# --------------------------------------------------------------------------- #
def response_label_from_spans(spans: Optional[list]) -> int:
    """RAGTruth rule: hallucinated iff >= 1 annotated span. Matches
    LettuceDetect's example-level definition (evaluator.py)."""
    return 1 if spans else 0


# --------------------------------------------------------------------------- #
# RAGTruth
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ragtruth_prompt_and_context(source: dict) -> tuple[str, str, Optional[str]]:
    """Return (prompt, context, question) for a RAGTruth source record.

    RAGTruth ships a fully-built `prompt` string already (question + passages
    for QA, instruction + source for summary/data2txt). We keep that prompt as
    the detector input (so it matches LettuceDetect's preprocessing), and also
    expose `context` = the raw grounding evidence for NLI/judge baselines.
    `question` is only meaningful for QA; None otherwise.
    """
    prompt = source["prompt"]
    if isinstance(prompt, dict):  # defensive; RAGTruth prompts are strings
        prompt = json.dumps(prompt, ensure_ascii=False)
    src_info = source["source_info"]
    context = src_info if isinstance(src_info, str) else json.dumps(src_info, ensure_ascii=False)
    return prompt, context, None


def load_ragtruth(count_tok: bool = True) -> list[dict]:
    sources = {s["source_id"]: s for s in _read_jsonl(P.RAGTRUTH_SOURCE_INFO)}
    records: list[dict] = []
    for resp in _read_jsonl(P.RAGTRUTH_RESPONSE):
        source = sources[resp["source_id"]]
        prompt, context, question = _ragtruth_prompt_and_context(source)
        answer = resp["response"]

        spans = [
            {"start": lb["start"], "end": lb["end"], "label": lb["label_type"]}
            for lb in resp["labels"]
        ]
        response_label = response_label_from_spans(spans)
        is_gray = resp.get("quality") != P.RAGTRUTH_GOOD_QUALITY

        n_tokens = count_tokens(prompt, answer) if count_tok else None
        records.append(
            {
                "id": f"ragtruth-{resp['id']}",
                "dataset": "ragtruth",
                "task_type": P.RAGTRUTH_TASK_MAP.get(source["task_type"], source["task_type"]),
                "split": resp["split"],
                "question": question,
                "context": context,
                "answer": answer,
                "prompt": prompt,
                "spans": spans,
                "response_label": response_label,
                "is_gray": is_gray,
                "n_tokens": n_tokens,
                "over_token_limit": (n_tokens is not None and n_tokens > P.MAX_TOKENS),
                "model": resp.get("model"),
                "meta": {
                    "source_id": resp["source_id"],
                    "quality": resp.get("quality"),
                    "ragtruth_source": source.get("source"),
                },
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Trivia+
# --------------------------------------------------------------------------- #
def _triviaplus_path() -> str:
    if os.path.exists(P.TRIVIAPLUS_PARQUET):
        return P.TRIVIAPLUS_PARQUET
    if os.path.exists(P.TRIVIAPLUS_PARQUET_FALLBACK):
        return P.TRIVIAPLUS_PARQUET_FALLBACK
    raise FileNotFoundError(
        "Trivia+ parquet not found. Expected at "
        f"{P.TRIVIAPLUS_PARQUET} or {P.TRIVIAPLUS_PARQUET_FALLBACK}. "
        "Run datasets/get_datasets.py on your machine."
    )


def build_triviaplus_prompt(question: str, context: str) -> str:
    """Trivia+ has no pre-built prompt, so we construct the LettuceDetect-style
    input documented in HallucinationSample: question at the front so it is
    never lost to context truncation."""
    return f"User request: {question}\n\n{context}"


def load_triviaplus(count_tok: bool = True) -> list[dict]:
    df = pd.read_parquet(_triviaplus_path())
    records: list[dict] = []
    for i, row in df.iterrows():
        question = str(row["question"])
        context = str(row["article"])
        answer = str(row["answer"])
        prompt = build_triviaplus_prompt(question, context)

        raw_label = int(row["response_level_label"])  # 0 / 1 / -1
        # response_level_label_binary already folds the -1 gray cases into 1;
        # we keep that as the binary label and flag gray separately.
        response_label = int(row["response_level_label_binary"])
        is_gray = raw_label == P.TRIVIAPLUS_GRAY_LABEL

        n_tokens = count_tokens(prompt, answer) if count_tok else None
        records.append(
            {
                "id": f"triviaplus-{row['split']}-{i}",
                "dataset": "triviaplus",
                "task_type": P.TASK_QA,
                "split": str(row["split"]),
                "question": question,
                "context": context,
                "answer": answer,
                "prompt": prompt,
                "spans": None,  # response-level only (no char-offset span labels)
                "response_label": response_label,
                "is_gray": is_gray,
                "n_tokens": n_tokens,
                "over_token_limit": (n_tokens is not None and n_tokens > P.MAX_TOKENS),
                "model": str(row.get("model")),
                "meta": {
                    "source": str(row.get("source")),
                    "response_level_label_raw": raw_label,
                },
            }
        )
    return records


LOADERS = {"ragtruth": load_ragtruth, "triviaplus": load_triviaplus}


# --------------------------------------------------------------------------- #
# Filtering views (non-destructive: we keep every record + flags, and offer
# helpers to get the analysis subsets).
# --------------------------------------------------------------------------- #
def primary_view(records: list[dict]) -> list[dict]:
    """Registered primary analysis set: exclude gray-area labels only.

    Over-window contexts remain present and must be evaluated with the
    documented chunk-and-aggregate procedure required by the exposé.
    """
    return [r for r in records if not r["is_gray"]]


def single_window_view(records: list[dict]) -> list[dict]:
    """Diagnostic subset that fits ModernBERT's 8,192-token window directly."""
    return [r for r in primary_view(records) if not r["over_token_limit"]]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def dataset_stats(records: list[dict]) -> dict:
    by_split = Counter(r["split"] for r in records)
    by_task = Counter(r["task_type"] for r in records)
    pos = sum(r["response_label"] for r in records)
    gray = sum(r["is_gray"] for r in records)
    over = sum(bool(r["over_token_limit"]) for r in records)
    toks = [r["n_tokens"] for r in records if r["n_tokens"] is not None]
    prim = primary_view(records)
    single = single_window_view(records)
    stats = {
        "n_total": len(records),
        "by_split": dict(by_split),
        "by_task": dict(by_task),
        "n_positive": pos,
        "pct_positive": round(100 * pos / len(records), 2) if records else 0.0,
        "n_gray": gray,
        "n_over_token_limit": over,
        "n_primary": len(prim),
        "n_single_window": len(single),
    }
    if toks:
        toks_sorted = sorted(toks)
        stats["tokens"] = {
            "min": toks_sorted[0],
            "median": toks_sorted[len(toks_sorted) // 2],
            "max": toks_sorted[-1],
            "p95": toks_sorted[int(0.95 * (len(toks_sorted) - 1))],
            "n_over_8192": over,
            "pct_over_8192": round(100 * over / len(toks_sorted), 2),
        }
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare thesis datasets into a common schema.")
    ap.add_argument("--dataset", choices=list(LOADERS) + ["all"], default="all")
    ap.add_argument("--no-tokens", action="store_true", help="skip ModernBERT token counting")
    ap.add_argument("--out", default=P.PREPARED_DIR)
    args = ap.parse_args()

    count_tok = not args.no_tokens
    targets = list(LOADERS) if args.dataset == "all" else [args.dataset]

    all_stats = {}
    for name in targets:
        print(f"\n=== Preparing {name} (count_tokens={count_tok}) ===")
        records = LOADERS[name](count_tok=count_tok)
        out_path = os.path.join(args.out, f"{name}.jsonl")
        write_jsonl(records, out_path)
        st = dataset_stats(records)
        all_stats[name] = st
        print(json.dumps(st, indent=2))
        print(f"  wrote {len(records)} records -> {out_path}")

    stats_path = os.path.join(args.out, "stats.json")
    os.makedirs(args.out, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nWrote combined stats -> {stats_path}")


if __name__ == "__main__":
    main()
