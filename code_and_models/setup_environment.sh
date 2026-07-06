#!/usr/bin/env bash
# One-shot environment setup for the thesis. Run on your machine (ideally in a venv).
#   bash setup_environment.sh
set -e
cd "$(dirname "$0")"

echo "== Python deps =="
pip install -U "lettucedetect" transformers datasets accelerate peft scikit-learn \
               huggingface_hub evaluate sentencepiece pandas pyarrow

echo "== ModernBERT repo (not bundled; large) =="
[ -d ModernBERT ] || git clone https://github.com/AnswerDotAI/ModernBERT.git

cat <<'PY'

# --- Download model weights (run in Python) -------------------------------
from huggingface_hub import snapshot_download
# LettuceDetect reproduction-reference checkpoints
snapshot_download("KRLabsOrg/lettucedect-base-modernbert-en-v1")
snapshot_download("KRLabsOrg/lettucedect-large-modernbert-en-v1")
# ModernBERT backbone (for training your own QA-only / all-task detectors)
snapshot_download("answerdotai/ModernBERT-base")
snapshot_download("answerdotai/ModernBERT-large")
# NLI baseline (DeBERTa-v3 MNLI/ANLI)
snapshot_download("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
# --------------------------------------------------------------------------
PY
echo "Done. RAGTruth + FaithBench data are already in this folder."
