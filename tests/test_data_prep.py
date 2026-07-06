"""Unit tests for the Week-1 label-mapping, gray-area, and token-filter logic.

Run from the thesis root:
    python -m pytest tests/ -q
or without pytest installed:
    python tests/test_data_prep.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_prep as dp  # noqa: E402
import protocol as P  # noqa: E402


# --------------------------------------------------------------------------- #
# Label mapping: span annotations -> response-level (the LOCKED rule)
# --------------------------------------------------------------------------- #
def test_response_label_no_spans_is_supported():
    assert dp.response_label_from_spans([]) == 0
    assert dp.response_label_from_spans(None) == 0


def test_response_label_any_span_is_hallucinated():
    assert dp.response_label_from_spans([{"start": 1, "end": 5, "label": "X"}]) == 1
    # Two spans -> still exactly 1 (binary).
    assert dp.response_label_from_spans(
        [{"start": 1, "end": 5, "label": "X"}, {"start": 9, "end": 12, "label": "Y"}]
    ) == 1


def test_response_label_matches_lettucedetect_example_level():
    """LettuceDetect evaluator: example label = 1 iff any token is hallucinated,
    i.e. iff >= 1 span. Our mapping must agree for both polarities."""
    assert dp.response_label_from_spans([{"start": 0, "end": 1, "label": "Z"}]) == 1
    assert dp.response_label_from_spans([]) == 0


# --------------------------------------------------------------------------- #
# Trivia+ prompt construction (question kept at the front)
# --------------------------------------------------------------------------- #
def test_triviaplus_prompt_puts_question_first():
    prompt = dp.build_triviaplus_prompt("Who wrote Hamlet?", "Hamlet was written by Shakespeare.")
    assert prompt.startswith("User request: Who wrote Hamlet?")
    assert "Shakespeare" in prompt


# --------------------------------------------------------------------------- #
# Token filter (exact ModernBERT tokenizer). Skipped if transformers/tokenizer
# is unavailable so the suite still runs on a bare machine.
# --------------------------------------------------------------------------- #
def _tokenizer_available() -> bool:
    try:
        dp.get_tokenizer()
        return True
    except Exception:
        return False


def test_token_filter_flags_overlong(monkeypatch=None):
    if not _tokenizer_available():
        print("  [skip] tokenizer unavailable")
        return
    short = dp.count_tokens("User request: hi", "ok")
    assert short < P.MAX_TOKENS
    # A context far exceeding the window must be flagged over-limit.
    long_ctx = "word " * 20000
    n = dp.count_tokens(f"User request: q\n\n{long_ctx}", "some answer")
    assert n > P.MAX_TOKENS


# --------------------------------------------------------------------------- #
# Integration: the gray-area exclusion + primary_view on REAL data (small reads)
# --------------------------------------------------------------------------- #
def test_ragtruth_loads_and_maps(small=300):
    """Load a slice of real RAGTruth (no token counting for speed) and check the
    schema + that every record carries a binary response_label and a gray flag."""
    if not os.path.exists(P.RAGTRUTH_RESPONSE):
        print("  [skip] RAGTruth not on disk")
        return
    records = dp.load_ragtruth(count_tok=False)
    assert len(records) > 1000
    sample = records[:small]
    for r in sample:
        assert r["dataset"] == "ragtruth"
        assert r["response_label"] in (0, 1)
        assert isinstance(r["is_gray"], bool)
        assert r["task_type"] in (P.TASK_QA, P.TASK_SUMMARIZATION, P.TASK_DATA2TXT)
        # Span-derived label must agree with the rule.
        assert r["response_label"] == dp.response_label_from_spans(r["spans"])
    # Primary analysis drops gray labels, but keeps over-window cases for the
    # exposé-required chunk-and-aggregate evaluation.
    prim = dp.primary_view(records)
    assert all(not r["is_gray"] for r in prim)
    assert len(prim) <= len(records)


def test_primary_and_single_window_views_follow_registered_scope():
    records = [
        {"id": "short", "is_gray": False, "over_token_limit": False},
        {"id": "long", "is_gray": False, "over_token_limit": True},
        {"id": "gray", "is_gray": True, "over_token_limit": False},
    ]
    assert [r["id"] for r in dp.primary_view(records)] == ["short", "long"]
    assert [r["id"] for r in dp.single_window_view(records)] == ["short"]


def test_triviaplus_loads_and_maps():
    if not (os.path.exists(P.TRIVIAPLUS_PARQUET) or os.path.exists(P.TRIVIAPLUS_PARQUET_FALLBACK)):
        print("  [skip] Trivia+ parquet not on disk")
        return
    records = dp.load_triviaplus(count_tok=False)
    assert len(records) == 3224  # the complete benchmark
    for r in records:
        assert r["dataset"] == "triviaplus"
        assert r["task_type"] == P.TASK_QA
        assert r["spans"] is None  # response-level only
        assert r["response_label"] in (0, 1)
    # Exactly the -1 raw-label rows are gray.
    n_gray = sum(r["is_gray"] for r in records)
    assert n_gray == 139, f"expected 139 gray Trivia+ rows, got {n_gray}"


# --------------------------------------------------------------------------- #
# Bare-bones runner so the file works without pytest.
# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
