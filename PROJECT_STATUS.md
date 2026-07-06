# Project status — 6 July 2026

This is the reviewer-facing status record for the bachelor thesis repository.
It distinguishes completed work, reproducible next steps, and unresolved
protocol decisions. It is intentionally honest about work that has not run yet.

## Completed in Week 1

- Defined a common schema for RAGTruth and Trivia+ in `src/data_prep.py`.
- Implemented the registered span-to-response label rule and gray-area flags.
- Measured input lengths with the ModernBERT tokenizer. Trivia+ has 314 inputs
  over 8,192 tokens (9.74%); they are retained for chunked evaluation.
- Verified dataset counts: RAGTruth 17,790 responses; Trivia+ 3,224 responses.
- Defined portable laptop/Kaggle paths and fixed seeds in `src/protocol.py`.
- Added synchronized CSV/SQLite experiment logging in `src/tracking.py`.
- Added tests for labels, prompt construction, analysis views, token limits,
  and real-data loading.
- Wrote a start-to-finish Kaggle recipe for the Week-2 reproduction runs.

No model-training, zero-shot, adaptation, baseline, or cost result is claimed at
this point.

## Registered scope and implementation record

| Item | Repository implementation |
|---|---|
| Domain shift | RAGTruth QA → Trivia+ QA |
| Task shift | RAGTruth QA → RAGTruth summarization |
| Adaptation ladder | Threshold recalibration → frozen linear probe → LoRA |
| Adaptation sizes | 50, 100, 200, 500 labeled target examples |
| Primary comparison | Best parameter-light method at n=100 vs. LLM judge at matched cost |
| Seeds | 5 for the primary comparison; 3 for secondary grids |
| Primary metric | Response-level F1 |
| Cost unit | USD per 1,000 checked responses |
| Long contexts | Retain and evaluate by chunk-and-aggregate; report a single-window diagnostic |

The execution calendar was compressed to the actual 30 June–2 September 2026
window. This changes scheduling, not the scientific questions.

## Open decisions / deviations requiring resolution

1. **LLM judge model and dated USD prices are not pinned.** The exposé expected
   these at registration. The sample cap is fixed at 500 per arm, and code is
   guarded so the judge cannot run while the remaining values are placeholders.
2. **The numeric reliability operating point is not pinned.** The exposé calls
   for a minimum F1 and minimum TPR at 5% FPR anchored to in-domain reproduction.
   These values must be recorded before final evaluation.
3. **Long-context aggregation details remain to be implemented and frozen.**
   The primary Trivia+ test set has 618 non-gray cases; 574 fit one ModernBERT
   window. The other primary cases must not be silently dropped.
4. **No project-level license has been selected.** If the GitHub repository will
   be public rather than supervisor-only, choose a license before publication.

Every resolution should be dated in the changelog of `FROZEN_PROTOCOL.md`.

## Week 2 next step

Run the published checkpoint sanity check, then train/evaluate the QA-only and
all-task models using `code_and_models/REPRODUCE_TRAINING.md`. Commit only code,
configuration, and small metric logs. Dataset files, checkpoints, generated
JSONL, papers, Office documents, caches, and local assistant/IDE state remain
outside Git history.

## Repository boundary

This `GitHubRepo/` directory is a standalone Git repository and the only folder
intended for GitHub. The surrounding thesis workspace contains private drafts,
third-party repositories, raw data, papers, and generated outputs. Those files
are reproducible through the documented download/setup steps and should not be
copied into this repository.
