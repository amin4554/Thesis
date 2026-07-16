"""Bootstrap 95% CIs (F1_hallucinated, balanced accuracy, AUROC) for RAGTruth
example-level predictions dumped by scripts/evaluate.py --dump_dir.

Matches the Trivia+ harness: percentile bootstrap over examples, seed 13, 1000
resamples, threshold 0.5. Reads {task}_preds.csv files with columns label,prob.

Usage:
    python src/bootstrap_ragtruth.py --dump_dir output/evals/zeroshot/ragtruth_full_preds \
        --model full --out output/evals/zeroshot/ragtruth_full_cis.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

TASK_ORDER = ["QA", "Summary", "Data2txt", "whole"]


def f1_hall(y, scores, thr=0.5):
    preds = (scores >= thr).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        y, preds, labels=[0, 1], average=None, zero_division=0
    )
    return float(f1[1])


def balacc(y, scores, thr=0.5):
    return float(balanced_accuracy_score(y, (scores >= thr).astype(int)))


def auroc(y, scores):
    return float(roc_auc_score(y, scores))


def bootstrap_ci(fn, y, scores, n_boot=1000, seed=13):
    """Percentile bootstrap 95% CI over examples (protocol §6, seed 13)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y[b], scores[b]))
        except ValueError:  # single-class resample
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_preds(path):
    y, s = [], []
    with Path(path).open() as fh:
        for row in csv.DictReader(fh):
            y.append(int(row["label"]))
            s.append(float(row["prob"]))
    return np.array(y), np.array(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="Dir with {task}_preds.csv files")
    ap.add_argument("--model", default="", help="Label written into the model column")
    ap.add_argument("--out", default=None, help="Output CSV (default: stdout only)")
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    rows = []
    files = sorted(dump.glob("*_preds.csv"))
    order = {t: i for i, t in enumerate(TASK_ORDER)}
    files.sort(key=lambda p: order.get(p.name.replace("_preds.csv", ""), 99))

    header = ["model", "task", "n", "n_pos", "f1", "f1_lo", "f1_hi",
              "balacc", "ba_lo", "ba_hi", "auroc", "auroc_lo", "auroc_hi"]
    print("  ".join(f"{h:>8}" for h in header))
    for f in files:
        task = f.name.replace("_preds.csv", "")
        y, s = load_preds(f)
        rec = {
            "model": args.model, "task": task, "n": len(y), "n_pos": int(y.sum()),
            "f1": f1_hall(y, s), "balacc": balacc(y, s), "auroc": auroc(y, s),
        }
        rec["f1_lo"], rec["f1_hi"] = bootstrap_ci(f1_hall, y, s)
        rec["ba_lo"], rec["ba_hi"] = bootstrap_ci(balacc, y, s)
        rec["auroc_lo"], rec["auroc_hi"] = bootstrap_ci(auroc, y, s)
        rows.append(rec)
        print(f"{args.model:>8}  {task:>8}  {len(y):>8}  {int(y.sum()):>8}  "
              f"{rec['f1']:.4f}  {rec['f1_lo']:.3f}  {rec['f1_hi']:.3f}  "
              f"{rec['balacc']:.4f}  {rec['ba_lo']:.3f}  {rec['ba_hi']:.3f}  "
              f"{rec['auroc']:.4f}  {rec['auroc_lo']:.3f}  {rec['auroc_hi']:.3f}")

    if args.out:
        with Path(args.out).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
