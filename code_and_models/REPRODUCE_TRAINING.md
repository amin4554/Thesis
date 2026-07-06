# Week 2 — Reproduce the QA-only and all-task detectors (100% on Kaggle)

Complete, self-contained recipe: every step — preprocessing, sanity check,
training, and evaluation — runs in one Kaggle notebook, using the **bundled**
`LettuceDetect/` code (pinned copy, not PyPI) so the reproduction is
attributable to an exact code state. Nothing needs to run locally; the local
8 GB GPU is not used (the bundled trainer is plain fp32 AdamW — ModernBERT-large
needs ~6.4 GB for weights+gradients+optimizer *before* activations at
up-to-4k-token contexts, so it cannot train on 8 GB).

The same notebook is run **twice**, changing one variable (`MODE`):

| Run | `MODE` | Training data | Role in thesis |
|---|---|---|---|
| **A — all-task** (full data) | `"all_task"` | full RAGTruth train split, 15,090 samples (QA + Summary + Data2txt) | in-domain reproduction anchor + RQ1 task-shift arm |
| **B — QA-only** | `"qa_only"` | train split filtered to task type `QA` (~5k samples) | RQ1 domain-shift arm (task fixed = QA → Trivia+) |

**Targets** (`lettucedect-large` reference, RAGTruth test, tolerance ±1–2 F1):
**79.22 example-level F1** (whole test set) / **58.93 span-level F1**.

> **Cost-analysis scope:** training compute is deliberately **outside** the
> RQ3 budget comparison — RQ3 prices *per-evaluation inference* only. Free
> Kaggle training therefore doesn't distort the cost results. Hardware,
> wall-clock, and hyperparameters are still recorded for the Experimental
> Setup chapter.

---

## Step 0 — Kaggle account prerequisites (once)

1. Kaggle account, **phone-verified** (required to enable Internet in
   notebooks — needed to pull `answerdotai/ModernBERT-large` and the
   reference checkpoint from HuggingFace).
2. Check your GPU quota under *Settings → Accelerators*: ~**30 GPU-h/week**.
   The full recipe (probe + two training versions) fits in one week's quota.

## Step 1 — Upload the data as a private Kaggle Dataset (once)

Create a **private** Dataset named `thesis-data` containing exactly these two
folders from the thesis workspace (both are deliberately *not* in git, so
cloning the GitHub repo is not enough):

```
thesis-data/
├── LettuceDetect/          ← code_and_models/LettuceDetect/  (the whole folder)
└── RAGTruth/
    └── dataset/            ← code_and_models/RAGTruth/dataset/  (response.jsonl + source_info.jsonl)
```

Easiest: zip the two folders, upload both zips to one new Dataset (Kaggle
auto-extracts). The notebook mount point varies by account/UI version — this
dataset mounts at `/kaggle/input/datasets/anfs2003/thesis-data` (set once as
`DATA` in Cell 1; find yours with `!ls -R /kaggle/input | head -30`). After
attaching, **verify these paths exist**:

- `{DATA}/LettuceDetect/pyproject.toml`
- `{DATA}/RAGTruth/dataset/response.jsonl`

If the zips extracted with an extra nesting level, adjust `DATA` (or the two
subpaths in Cells 2–3) accordingly.

## Step 2 — Create the notebook (once)

New Notebook → attach the `thesis-data` Dataset → settings:

- **Accelerator:** GPU **P100** (single GPU; the trainer is single-device)
- **Internet:** **ON**
- **Persistence:** No

## Step 3 — The notebook cells (paste all of them)

```python
# Cell 1 — run configuration.
# MODE: "qa_only" (run B) or "all_task" (run A) — the ONLY thing you change between runs.
# RUN_SANITY: evaluate the published reference checkpoint first (do this once,
#             in the interactive probe session; leave False in the Save&Run versions).
MODE = "qa_only"
RUN_SANITY = False
BATCH_SIZE = 2          # drop to 1 if Cell 6 hits CUDA OOM; record the value used
EPOCHS = 6              # repo default; cut to 3-4 only if the 12 h cap forces it — record it

# Where Kaggle mounted the thesis-data Dataset — check with:  !ls /kaggle/input -R | head
# (mount points vary: /kaggle/input/thesis-data or /kaggle/input/datasets/<user>/thesis-data)
DATA = "/kaggle/input/datasets/anfs2003/thesis-data"
```

```python
# Cell 2 — install the bundled (pinned) LettuceDetect copy
# (/kaggle/input is read-only, so copy it out first)
!cp -r {DATA}/LettuceDetect /kaggle/working/LettuceDetect
%cd /kaggle/working/LettuceDetect
%pip install -q -e .
```

```python
# Cell 3 — preprocess full RAGTruth into LettuceDetect's format
# → data/ragtruth/ragtruth_data.json : 17,790 samples, splits (train/test) and
#   task types (QA / Summary / Data2txt) preserved. Used for training AND evaluation.
!python lettucedetect/preprocess/preprocess_ragtruth.py \
    --input_dir {DATA}/RAGTruth/dataset \
    --output_dir data/ragtruth
```

```python
# Cell 4 — sanity check (once): the published checkpoint through this exact
# eval pipeline must land near 79.22 example-F1 on "whole dataset".
# If it doesn't, the pipeline is broken — fix before training anything.
if RUN_SANITY:
    !python scripts/evaluate.py \
        --model_path KRLabsOrg/lettucedect-large-modernbert-en-v1 \
        --data_path data/ragtruth/ragtruth_data.json \
        --evaluation_type example_level --batch_size 8 \
        2>&1 | tee /kaggle/working/sanity_reference_checkpoint.log
```

```python
# Cell 5 — build the training file
# all_task: the full file (train.py itself selects the 15,090 train-split samples)
# qa_only : filter the train split to task_type "QA" with LettuceDetect's own script
if MODE == "qa_only":
    !python scripts/filter_data.py \
        --input data/ragtruth/ragtruth_data.json \
        --output data/ragtruth/train_file.json \
        --split train --task-type QA
else:
    !cp data/ragtruth/ragtruth_data.json data/ragtruth/train_file.json
```

```python
# Cell 6 — TRAIN (repo-default recipe; seed 123 is hardcoded in train.py and is
# one of the protocol seeds {13, 42, 123}; best epoch kept by dev hallucinated-F1)
!python scripts/train.py \
    --ragtruth-path data/ragtruth/train_file.json \
    --model-name answerdotai/ModernBERT-large \
    --output-dir /kaggle/working/models/ragtruth_{MODE}_large \
    --batch-size {BATCH_SIZE} --epochs {EPOCHS} --learning-rate 1e-5 \
    2>&1 | tee /kaggle/working/train_{MODE}.log
```

```python
# Cell 7 — EVALUATE the freshly trained model on the RAGTruth test set.
# example_level → compare to 79.22 · char_level (span) → compare to 58.93.
# The all_task model is judged on the "Task type: whole dataset" block;
# the qa_only model on the "Task type: QA" block.
!python scripts/evaluate.py \
    --model_path /kaggle/working/models/ragtruth_{MODE}_large \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level --batch_size 8 \
    2>&1 | tee /kaggle/working/eval_{MODE}_example_level.log

!python scripts/evaluate.py \
    --model_path /kaggle/working/models/ragtruth_{MODE}_large \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type char_level \
    2>&1 | tee /kaggle/working/eval_{MODE}_char_level.log
```

```python
# Cell 8 — pack checkpoint + all logs for download from the Output tab
!cd /kaggle/working && zip -qr ragtruth_{MODE}_large.zip models/ragtruth_{MODE}_large *.log
!ls -lh /kaggle/working/*.zip /kaggle/working/*.log
```

## Step 4 — Execution order

1. **Probe session (interactive, ~1 h of quota).** Set `RUN_SANITY = True`,
   `MODE = "qa_only"`. Run Cells 1–5, check the sanity log lands near 79.22,
   then start Cell 6 and read the tqdm ETA for one epoch. Multiply:
   `6 × time/epoch + ~30 min eval` must fit the **12 h session cap**
   (remember the all-task run is ~3× the QA-only epoch time). Stop the
   session once you have the numbers.
2. **Run B (QA-only):** set `RUN_SANITY = False`, keep `MODE = "qa_only"` →
   **Save Version → Save & Run All**. Runs in the background, no browser
   needed; `ragtruth_qa_only_large.zip` appears under the notebook's
   **Output** tab when done.
3. **Run A (all-task, full data):** edit Cell 1 to `MODE = "all_task"` →
   **Save Version → Save & Run All** again. If the probe said 6 epochs won't
   fit in 12 h, lower `EPOCHS` to 3–4 and record the deviation.
4. **If Cell 6 OOMs:** `BATCH_SIZE = 1` and re-save. Batch ≠ 4 (repo default)
   is fine but is a documented deviation — note it next to the reproduction
   numbers.

**No-resume warning:** the trainer only saves the best checkpoint and cannot
resume — a version that dies mid-run restarts from zero. That is what the
probe session is for.

## Step 5 — Collect results, record, check exit criteria

1. Download both zips (checkpoint + logs) from the two versions' Output tabs;
   unpack under `code_and_models/LettuceDetect/output/models/` in the thesis
   workspace. Checkpoints stay out of git (`.gitignore`); only code, configs,
   and logged metrics are committed.
2. Log every run into the results log (`src/tracking.py` schema): model,
   train-set, eval-set, seed (123), metrics, wall-clock, hardware
   (`Kaggle P100 16 GB, fp32, batch <actual>, epochs <actual>`).
3. **Week-2 exit criteria:** all-task large within ±1–2 F1 of
   **79.22 / 58.93** on the whole test set. If off-target, debug tokenization
   / label alignment / threshold **before** any transfer runs — this is the
   credibility anchor of the thesis.

**Next after this:** unified `evaluate.py` with bootstrap CIs, then the first
zero-shot runs on Trivia+ (`output/prepared/triviaplus.jsonl`: 618 non-gray
primary test examples, including 574 single-window cases) and
RAGTruth-summarization — see `THESIS_PLAN.md`, Weeks 2–3.
