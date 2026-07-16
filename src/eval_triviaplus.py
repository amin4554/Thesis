"""Zero-shot response-level evaluation of LettuceDetect checkpoints on Trivia+.

Week 2/3 harness (RQ1 domain shift). Reads the prepared common-schema
``triviaplus.jsonl`` (see src/data_prep.py), scores every example with a trained
token-classification detector, and reports response-level metrics per
FROZEN_PROTOCOL §6: F1 / precision / recall (hallucinated class), balanced
accuracy, AUROC, at the default threshold and (optionally) a threshold
calibrated on the Trivia+ valid split. Bootstrap 95% CIs on everything.

Scoring rule (mirrors LettuceDetect's example-level evaluator): a response's
hallucination score = max over its answer tokens of P(token = hallucinated).
At threshold 0.5 this is identical to "predicted positive iff any token argmax
is 1", i.e. directly comparable to the RAGTruth example-level numbers.

Primary view (protocol §4/§5): split == test, is_gray == False,
over_token_limit == False  ->  574 examples. --include-gray runs the
sensitivity variant.

Inference window: --max-length 8192 by default (protocol §5; ModernBERT native
window). Run again with --max-length 4096 for the truncation sensitivity check.

Usage (Kaggle, after `pip install -e LettuceDetect`):
    python eval_triviaplus.py \
        --model-path anfs4554/lettucedetect-full-large \
        --data-path /kaggle/input/.../triviaplus.jsonl \
        --calibrate --output-dir /kaggle/working/trivia_results

Outputs: metrics JSON + per-example CSV (id, generator model, score, pred,
label) for the Week-3 error-analysis sample.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- data


def load_examples(
    data_path: str | Path,
    split: str,
    include_gray: bool = False,
) -> list[dict]:
    """Load prepared Trivia+ rows for one split, applying the primary-view filters."""
    rows = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == split]
    n_split = len(rows)
    if not include_gray:
        rows = [r for r in rows if not r["is_gray"]]
    n_gray = n_split - len(rows)
    rows = [r for r in rows if not r["over_token_limit"]]
    print(
        f"[data] split={split}: {n_split} total, -{n_gray} gray, "
        f"-{n_split - n_gray - len(rows)} over token limit -> {len(rows)} evaluated"
    )
    return rows


# ---------------------------------------------------------------- scoring


def score_examples(
    rows: list[dict],
    model_path: str,
    max_length: int,
    batch_note: str = "",
) -> tuple[np.ndarray, float]:
    """Score each (prompt, answer) with max answer-token hallucination probability.

    Returns (scores array, wall-clock seconds for pure inference).
    """
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from lettucedetect.datasets.hallucination_dataset import HallucinationDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    scores = []
    t0 = time.time()
    with torch.no_grad():
        for r in tqdm(rows, desc=f"Scoring {batch_note}", leave=True):
            encoding, _labels, offsets, answer_start = (
                HallucinationDataset.prepare_tokenized_input(
                    tokenizer, r["prompt"], r["answer"], max_length
                )
            )
            logits = model(
                encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits[0]
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            seq_len = encoding["input_ids"].shape[1]
            answer_probs = [
                probs[i].item()
                for i in range(answer_start, seq_len)
                if offsets[i][1] > offsets[i][0]  # skip special/degenerate tokens
            ]
            scores.append(max(answer_probs) if answer_probs else 0.0)
    return np.asarray(scores), time.time() - t0


# ---------------------------------------------------------------- metrics


def metrics_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )

    preds = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "threshold": float(thr),
        "f1_hallucinated": float(f1[1]),
        "precision_hallucinated": float(p[1]),
        "recall_hallucinated": float(r[1]),
        "f1_supported": float(f1[0]),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def bootstrap_ci(
    fn, y: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 13
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over examples (protocol §6, seed 13)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y[b], scores[b]))
        except ValueError:  # resample with a single class
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing hallucinated-class F1 (for calibration on valid)."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", required=True, help="Local dir or HF repo id")
    ap.add_argument("--data-path", required=True, help="Prepared triviaplus.jsonl")
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Also report metrics at the F1-optimal threshold found on the "
        "Trivia+ VALID split (never on the eval split itself).",
    )
    ap.add_argument("--include-gray", action="store_true", help="Sensitivity view")
    ap.add_argument("--output-dir", default="trivia_results")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_path.rstrip("/").split("/")[-1]
    run_tag = (
        f"{model_tag}_triviaplus-{args.split}_maxlen{args.max_length}"
        + ("_withgray" if args.include_gray else "")
    )

    rows = load_examples(args.data_path, args.split, args.include_gray)
    y = np.asarray([int(r["response_label"]) for r in rows])

    scores, secs = score_examples(rows, args.model_path, args.max_length, run_tag)

    results: dict = {
        "model": args.model_path,
        "dataset": "triviaplus",
        "split": args.split,
        "max_length": args.max_length,
        "include_gray": args.include_gray,
        "inference_seconds_total": round(secs, 2),
        "inference_seconds_per_example": round(secs / len(rows), 4),
        "default": metrics_at_threshold(y, scores, args.threshold),
        "auroc": auroc(y, scores),
    }

    # Bootstrap CIs (default threshold)
    f1_fn = lambda yy, ss: metrics_at_threshold(yy, ss, args.threshold)["f1_hallucinated"]
    ba_fn = lambda yy, ss: metrics_at_threshold(yy, ss, args.threshold)["balanced_accuracy"]
    results["default"]["f1_ci95"] = bootstrap_ci(f1_fn, y, scores)
    results["default"]["balanced_accuracy_ci95"] = bootstrap_ci(ba_fn, y, scores)
    results["auroc_ci95"] = bootstrap_ci(auroc, y, scores)

    # Optional calibrated threshold from the valid split
    if args.calibrate:
        cal_rows = load_examples(args.data_path, "valid", args.include_gray)
        cal_y = np.asarray([int(r["response_label"]) for r in cal_rows])
        cal_scores, _ = score_examples(
            cal_rows, args.model_path, args.max_length, "calibration(valid)"
        )
        thr = best_f1_threshold(cal_y, cal_scores)
        results["calibrated"] = metrics_at_threshold(y, scores, thr)
        cal_f1_fn = lambda yy, ss: metrics_at_threshold(yy, ss, thr)["f1_hallucinated"]
        results["calibrated"]["f1_ci95"] = bootstrap_ci(cal_f1_fn, y, scores)
        results["calibrated"]["calibrated_on"] = "valid"

    # ---- report
    print(f"\n===== {run_tag} =====")
    d = results["default"]
    print(f"n={d['n']} (positives={d['n_positive']})")
    print(
        f"Default thr {d['threshold']:.2f}: "
        f"F1={d['f1_hallucinated']:.4f} CI{d['f1_ci95']}  "
        f"P={d['precision_hallucinated']:.4f} R={d['recall_hallucinated']:.4f}  "
        f"BalAcc={d['balanced_accuracy']:.4f} CI{d['balanced_accuracy_ci95']}"
    )
    print(f"AUROC={results['auroc']:.4f} CI{results['auroc_ci95']}")
    if "calibrated" in results:
        c = results["calibrated"]
        print(
            f"Calibrated thr {c['threshold']:.4f} (on valid): "
            f"F1={c['f1_hallucinated']:.4f} CI{c['f1_ci95']}  "
            f"P={c['precision_hallucinated']:.4f} R={c['recall_hallucinated']:.4f}  "
            f"BalAcc={c['balanced_accuracy']:.4f}"
        )
    print(
        f"Inference: {results['inference_seconds_total']}s total, "
        f"{results['inference_seconds_per_example']}s/example"
    )

    # ---- persist
    (out_dir / f"{run_tag}.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / f"{run_tag}_per_example.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "generator_model", "n_tokens", "score", "pred", "label"])
        for r, s in zip(rows, scores):
            w.writerow(
                [
                    r["id"],
                    r["model"],
                    r["n_tokens"],
                    f"{s:.6f}",
                    int(s >= args.threshold),
                    r["response_label"],
                ]
            )
    print(f"\nSaved: {out_dir / run_tag}.json + _per_example.csv")


if __name__ == "__main__":
    main()
