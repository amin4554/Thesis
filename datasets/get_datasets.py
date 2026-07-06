#!/usr/bin/env python3
"""Fetch the full datasets used in the thesis. Run on your own machine.
    pip install datasets huggingface_hub pandas pyarrow
    python get_datasets.py
RAGTruth and FaithBench full data already ship in ../code_and_models/ ; this
script pulls Trivia+ (primary domain-shift target, GitHub) and the RAGBench
fallback (HuggingFace), and points you to the TofuEval repo.
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# Official Trivia+ release (Chen et al., 2026) -- GitHub, not HuggingFace.
# Single ~15 MB parquet = the COMPLETE benchmark (3,224 rows: 2,263 train /
# 316 valid / 645 test). License: CC BY-NC-ND 4.0 (academic / non-commercial).
TRIVIA_REPO = "https://github.com/amazon-science/hallucination-benchmark-trivialplus.git"
TRIVIA_DEST = os.path.normpath(os.path.join(HERE, "..", "code_and_models", "TriviaPlus"))


def try_hf(label, hf_id, **kw):
    try:
        from datasets import load_dataset
        print("\n=== {}  ({}) ===".format(label, hf_id))
        ds = load_dataset(hf_id, **kw)
        print(ds)
        return ds
    except Exception as e:
        print("  could not load {}: {}".format(hf_id, e))
        return None


def get_triviaplus():
    """Clone the official Trivia+ repo into ../code_and_models/TriviaPlus/.
    Idempotent: skips if the parquet is already present."""
    print("\n=== Trivia+ (primary domain-shift target) ===")
    parquet = os.path.join(TRIVIA_DEST, "triviaplus_dataset.parquet")
    if os.path.exists(parquet):
        print("  already present: {}".format(parquet))
        return TRIVIA_DEST
    tmp = os.path.join(HERE, "_triviaplus_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", TRIVIA_REPO, tmp], check=True)
        os.makedirs(TRIVIA_DEST, exist_ok=True)
        for fn in ("triviaplus_dataset.parquet", "verify_label_consistency.py",
                   "DATA_DETAILS.md", "README.md", "LICENSE", "NOTICE"):
            src = os.path.join(tmp, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(TRIVIA_DEST, fn))
        print("  installed to {}".format(TRIVIA_DEST))
        print("  verify: python verify_label_consistency.py triviaplus_dataset.parquet")
    except Exception as e:
        print("  could not clone Trivia+: {}".format(e))
        print("  manual: git clone {}".format(TRIVIA_REPO))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return TRIVIA_DEST


if __name__ == "__main__":
    # Trivia+ -- PRIMARY domain-shift target (human-verified, GitHub-hosted).
    get_triviaplus()
    # RAGBench -- pre-specified FALLBACK domain-shift target (model-generated
    # labels). Only needed if Trivia+ proves infeasible; otherwise skippable.
    try_hf("RAGBench (QA configs)", "rungalileo/ragbench", name="hotpotqa")
    # TofuEval (MeetingBank) -- optional secondary summarization target:
    print("\nTofuEval: git clone https://github.com/amazon-science/tofueval")
    print("\nRAGTruth & FaithBench full data are already in ../code_and_models/.")
    print("Model weights: see ../code_and_models/setup_environment.sh")
