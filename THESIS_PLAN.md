# Bachelor Thesis — Full Execution Plan

**Title:** *Closing the Gap on a Budget: Few-Shot Domain Adaptation and Cost-Accuracy Trade-offs of Lightweight Hallucination Detectors for RAG*
**Student:** Amin Niaziardekani (3119828) · B.Sc. Computer Science, SRH Berlin
**Supervisors:** Prakash Shreyaasri (1st) · Joel Dokmegang (2nd)
**Window:** Tue 30 Jun 2026 → **submission Wed 2 Sep 2026** (~9 working weeks + 2-day buffer)
**This file lives at:** `thesis/THESIS_PLAN.md` — update the checkboxes as you go.

---

## 1. What the thesis proves (research questions)

| # | Question | How it's answered |
|---|----------|-------------------|
| **RQ1** | How large is the zero-shot generalization gap when a RAGTruth-trained LettuceDetect detector is applied to a new **domain** (Trivia+) and a new **task** (summarization)? | Phase 1 — measure in-domain vs. transfer performance. |
| **RQ2** | How much of that gap can the registered adaptation ladder — threshold recalibration, frozen linear probe, and LoRA — close at n = 50–500? | Phase 3 — learning curves for domain and task shift; five seeds for the primary n=100 comparison. |
| **RQ3** | At a **matched inference-cost budget in USD per 1,000 responses**, does the adapted lightweight detector beat an LLM-as-judge and an NLI baseline (and a trivial length heuristic)? | Phase 4 — cost-matched Pareto analysis. |

**Core contribution:** a cost-accuracy Pareto picture showing *when* a cheap fine-tuned encoder is preferable to an expensive LLM judge for RAG hallucination detection, plus a practical few-shot recipe.

---

## 2. Experimental design at a glance

**Detector under study:** LettuceDetect (ModernBERT backbone, token/span-level) — repo at `code_and_models/LettuceDetect/`.

**Models you train:**
- QA-only detector (RAGTruth QA split)
- All-task detector (full RAGTruth)

**Evaluation axes:**
- *In-domain reference* — RAGTruth test. **Reproduction targets: 79.22 example-F1 / 58.93 span-F1** (`lettucedect-large`).
- *Domain-shift (task fixed = QA)* — **Trivia+** (primary), RAGBench QA (fallback).
- *Task-shift* — RAGTruth **summarization** split.
- *Optional secondary summarization* — FaithBench, TofuEval–MeetingBank.

**Adaptation:** threshold recalibration, a frozen-encoder linear probe, and LoRA
at n ∈ {50, 100, 200, 500}. Secondary grids use 3 seeds; the pre-registered
primary n=100 adapted-detector-vs-judge comparison uses 5 fixed seeds.

**Baselines for the cost comparison:**
- Length / trivial heuristic (floor)
- NLI baseline — DeBERTa-v3 MNLI, windowed + least-entailed aggregation
- LLM-as-judge — **FaithJudge protocol** (`code_and_models/faithjudge/`), model + sample cap pinned in advance

**Metrics:** response-level F1 (primary), precision/recall, balanced accuracy,
AUROC, and TPR at 5% FPR; span-level F1 where labels exist; **USD per 1,000
responses** and wall-clock/GPU time for the Pareto plot.

> **Label/length rule:** RAGTruth word-level spans → response-level positive if
> ≥1 annotated span. Trivia+/RAGBench = response-level only. Gray-area cases are
> excluded from the primary analysis and restored in sensitivity analysis.
> Trivia+ inputs over 8,192 tokens are flagged and handled by a documented
> chunk-and-aggregate procedure; the single-window subset is diagnostic only.

---

## 3. Phase map

| Phase | Weeks | Outcome |
|-------|-------|---------|
| **P0 — Setup & in-domain reproduction** | W1–W2 | Environment runs; RAGTruth numbers reproduced within tolerance. |
| **P1 — Zero-shot transfer (RQ1)** | W2–W3 | The generalization gap quantified. |
| **P2 — Baselines + cost harness** | W3–W4 | NLI, LLM-judge, heuristic all wired with cost logging. |
| **P3 — Few-shot adaptation (RQ2)** | W4–W6 | Registered adaptation ladder; domain- and task-shift curves. |
| **P4 — Cost-accuracy analysis (RQ3)** | W6–W7 | Pareto front + final tables/figures. |
| **P5 — Writing & figures** | W5–W9 (heavy W7–W9) | Full draft → polished thesis. |
| **P6 — Finalize & submit** | W9 + buffer | Proofread, format, hand in by 2 Sep. |

Writing runs **in parallel** from W5 — never leave it all to the end.

---

## 4. Week-by-week plan

### Week 1 · 30 Jun – 6 Jul — Setup, data, and freeze the protocol
**Goal:** a reproducible environment and a locked experimental protocol before any model runs.

- [ ] Run `code_and_models/setup_environment.sh`; pull ModernBERT + LettuceDetect checkpoints from HuggingFace (**needs your GPU machine** — torch not installed here).
- [~] Run `datasets/get_datasets.py` to pull full Trivia+ / RAGBench / TofuEval. (Trivia+ is **already complete** — 3,224 rows on disk; RAGBench + TofuEval are optional/fallback, pull when needed.)
- [x] Verify RAGTruth loads: `response.jsonl` + `source_info.jsonl` → 17,790 responses (15,090 train / 2,700 test) load into the common schema.
- [x] Implement and unit-test the **label-mapping** (spans → response-level),
  gray-area handling, and primary/single-window views. → `src/data_prep.py`,
  `tests/test_data_prep.py` (8 tests; runtime re-check required on the local
  Python environment before push).
- [x] Implement exact ModernBERT token counting and flag Trivia+ inputs over
  8,192 tokens. → **314 flagged (9.74%)**; 618 non-gray primary test cases,
  including 574 single-window cases. Chunked evaluation is a Week-2 deliverable.
- [x] Set up experiment tracking (CSV **+** SQLite results log), 3 secondary
  seeds, and 5 primary-comparison seeds. → `src/tracking.py`, `src/protocol.py`.
- [ ] **Resolve the disclosed registration deviation:** pin the LLM-judge model,
  dated USD prices, and numeric reliability operating point. The sample cap is
  already 500/arm; guards block judge execution until the remaining values exist.
- [x] Create a `requirements.txt` / lockfile. → `requirements.txt` (pinned). _Commit the repo skeleton:_ done — `GitHubRepo/` is the reviewer-facing repo (no `.claude`/assistant files there, ever; see its README policy).

**Deliverable:** `data_prep.py` + a one-page "frozen protocol" note (datasets, splits, label rules, metrics, judge model+budget).
**Exit criteria:** every dataset loads into a common schema; protocol note signed off mentally (and ideally by supervisor).

---

### Week 2 · 7 Jul – 13 Jul — Reproduce in-domain, then first transfer run
**Goal:** match the published RAGTruth numbers, then point the model at Trivia+.

- [ ] Train (or load) the QA-only and all-task detectors on RAGTruth.
- [ ] Evaluate on RAGTruth test; **reproduce 79.22 example-F1 / 58.93 span-F1** (large) within ~±1–2 F1.
- [ ] If off-target, debug (tokenization, label alignment, threshold) before moving on — this is the credibility anchor.
- [ ] Build the unified `evaluate.py` (response-level + span-level metrics, bootstrap CIs).
- [ ] First **zero-shot** run on Trivia+ (domain-shift) and RAGTruth-summarization (task-shift).

**Deliverable:** reproduction table + first zero-shot numbers.
**Exit criteria:** in-domain reproduction within tolerance; transfer pipeline produces metrics end-to-end.

---

### Week 3 · 14 Jul – 20 Jul — Lock down RQ1 (the gap)
**Goal:** a clean, defensible measurement of the zero-shot generalization gap.

- [ ] Full zero-shot eval grid: {QA-only, all-task} × {RAGTruth-test, Trivia+, RAGTruth-summ, (optional) FaithBench, TofuEval}.
- [ ] Add bootstrap 95% CIs; 3 seeds for secondary grids and all 5 fixed seeds
  for the primary n=100 comparison.
- [ ] Threshold sensitivity: default vs. target-calibrated threshold (report both).
- [ ] Error analysis sample (≈30–50 cases) — where/why transfer fails (categorize: domain vocab, length, label noise).
- [ ] Run the **sensitivity analysis** re-including gray-area cases.

**Deliverable:** **RQ1 results table + gap figure** (in-domain vs. domain-shift vs. task-shift).
**Exit criteria:** RQ1 is fully answered and frozen — these numbers won't change later.

---

### Week 4 · 21 Jul – 27 Jul — Baselines + cost harness (P2)
**Goal:** every comparator runs and logs cost identically.

- [ ] NLI baseline: DeBERTa-v3 MNLI, windowed + least-entailed aggregation over long contexts.
- [ ] Length / trivial heuristic baseline.
- [ ] LLM-judge via FaithJudge (`code_and_models/faithjudge/eval.py`, `prompt_templates.py`) on the pinned model + capped sample.
- [ ] **Cost instrumentation:** log wall-clock, tokens, GPU-seconds, and USD for
  every method (detector inference, NLI, judge API).
- [ ] Use the registered matched-budget unit: **USD per 1,000 checked
  responses**, with a documented T4-class reference configuration and
  sensitivity band.
- [ ] Begin few-shot adaptation harness (sampling n target examples, train/eval loop).

**Deliverable:** baseline results table + a `cost_log` schema with real numbers for all methods.
**Exit criteria:** all four methods produce both accuracy and cost on the same test sets.

---

### Week 5 · 28 Jul – 3 Aug — Few-shot adaptation, first curves (P3) + start writing
**Goal:** the headline RQ2 experiment running at scale, and the thesis skeleton on paper.

- [ ] Run the registered ladder (threshold recalibration, frozen linear probe,
  LoRA) on Trivia+ and RAGTruth summarization at n ∈ {50, 100, 200, 500}.
- [ ] Run the primary n=100 comparison with all 5 fixed seeds; use 3 seeds for
  the remaining secondary grid. Document any capacity-based pruning of LoRA at
  the smallest n rather than replacing the registered ladder with full fine-tuning.
- [ ] Plot first **learning curves** (target F1 vs. n) with seed variance bands.
- [ ] **Start the thesis document** (LaTeX/Word): set up template, fill Introduction + Related Work from the exposé and `docs/00_REFERENCES_INDEX.md`.

**Deliverable:** first Trivia+ learning curve; thesis skeleton with Intro + Related Work drafted.
**Exit criteria:** adaptation pipeline stable; ≥1 full learning curve produced.

---

### Week 6 · 4 Aug – 10 Aug — Finish adaptation + cost-accuracy front (P3→P4)
**Goal:** all RQ2 runs done; begin the RQ3 Pareto analysis.

- [ ] Complete adaptation runs for both registered targets (Trivia+ domain shift
  and RAGTruth summarization task shift); add FaithBench/TofuEval only if time allows.
- [ ] Compare adaptation vs. all-task model vs. NLI vs. judge at each n.
- [ ] Build the **cost-accuracy Pareto plot**: accuracy (y) vs. USD per 1,000
  responses (x), with wall-clock/GPU sensitivity, all methods + few-shot points.
- [ ] Identify the crossover point: at what budget does the adapted detector match/beat the LLM judge?
- [ ] Draft the **Methodology** chapter while details are fresh.

**Deliverable:** complete RQ2 curves + draft Pareto figure; Methodology chapter drafted.
**Exit criteria:** no experiments left that block the analysis chapter.

---

### Week 7 · 11 Aug – 17 Aug — Lock results, write Results & Analysis (P4→P5)
**Goal:** freeze all numbers and figures; write them up.

- [ ] Finalize every table/figure (consistent style, CIs, captions) — **code freeze on experiments**.
- [ ] Statistical checks: seed variance, significance of few-shot gains, CI overlap on the Pareto front.
- [ ] Write the **Results** and **Discussion/Analysis** chapters (answer RQ1–RQ3 explicitly).
- [ ] Write a **Threats to validity / limitations** section (label noise, single
  judge model, long-context aggregation, dataset licensing).

**Deliverable:** Results + Discussion chapters drafted; final figure set.
**Exit criteria:** all three RQs answered in prose with figures referenced.

---

### Week 8 · 18 Aug – 24 Aug — Complete first full draft
**Goal:** a complete, end-to-end readable thesis.

- [ ] Write Abstract, Introduction (final), Conclusion + Future Work.
- [ ] Write Experimental Setup chapter (datasets, hardware, hyperparameters, reproducibility appendix).
- [ ] Assemble references via `docs/references.bib`; check every citation resolves.
- [ ] Internal consistency pass: numbers in text == numbers in tables.
- [ ] **Send full draft to supervisor** for feedback (leave them a week).

**Deliverable:** **complete v1 draft** sent to Prakash Shreyaasri.
**Exit criteria:** no "TODO" placeholders; every chapter has content.

---

### Week 9 · 25 Aug – 31 Aug — Revise, polish, reproducibility package
**Goal:** incorporate feedback and tighten everything.

- [ ] Incorporate supervisor feedback.
- [ ] Polish figures/tables; proofread for grammar and flow; check formatting against SRH guidelines.
- [ ] Verify page/word count and required front matter (declaration of authorship, abstract, ToC).
- [ ] Clean up the **code repo**: README, `setup_environment.sh`, exact commands to reproduce each table/figure.
- [ ] Final plagiarism / AI-use declaration as required by SRH.

**Deliverable:** submission-ready thesis + clean reproducible repo.
**Exit criteria:** document is final; only formatting/export remains.

---

### Buffer · 1–2 Sep — Submit
- [ ] Final read-through (print or PDF).
- [ ] Export PDF, check embedded fonts/figures render.
- [ ] **Submit by Wed 2 Sep 2026** (and archive a tagged repo snapshot).

---

## 5. Thesis document structure (target ~40–60 pages)

| Chapter | Content | Draft in |
|---|---|---|
| Abstract | 250 words: problem, method, headline result | W8 |
| 1. Introduction | Motivation, RQ1–RQ3, contributions | W5 |
| 2. Background & Related Work | RAG, hallucination detection, LettuceDetect/ModernBERT, NLI, LLM-judge, cost work | W5 |
| 3. Methodology | Detector, datasets, label mapping, adaptation, cost model, metrics | W6 |
| 4. Experimental Setup | Splits, hardware, hyperparameters, seeds, reproducibility | W8 |
| 5. Results | RQ1 gap, RQ2 curves, RQ3 Pareto | W7 |
| 6. Discussion | Interpretation, crossover, when-to-use guidance | W7 |
| 7. Threats to Validity / Limitations | Label noise, single judge, token filtering | W7 |
| 8. Conclusion & Future Work | Summary + extensions | W8 |
| References + Appendices | `references.bib`; full hyperparameters, extra tables | W8–W9 |

---

## 6. Reproducibility & tracking conventions (set once, follow always)

- Secondary grids use fixed seeds **{13, 42, 123}**; the primary n=100
  adapted-detector-vs-judge comparison uses
  **{13, 42, 123, 2026, 3119828}**. Report mean ± std and bootstrap CIs.
- One results log (CSV/SQLite or W&B) — every row: model, train-set, eval-set, n, seed, metrics, cost.
- Each figure/table has a script that regenerates it from the log (`make figureN`).
- Pin all package versions; record GPU type and driver in the setup chapter.
- Commit after every experiment; tag the snapshot used for the final numbers.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Can't reproduce RAGTruth numbers | Med | High | W2 is dedicated to this; debug tokenization/label alignment before proceeding. Reference checkpoints exist on HF. |
| Trivia+ infeasible (format/labels) | Med | High | **RAGBench QA is the documented fallback** domain-shift target. |
| LLM-judge cost/time overrun | Med | Med | Sample cap **pinned at registration**; cache all judge outputs; never re-run. |
| Long contexts exceed 8k window | High | Med | Filter + report exclusions; use windowed aggregation for NLI. |
| GPU availability / training time | Med | High | Front-load heavy runs (W2, W5–W6); LoRA to cut cost; keep an all-task model as fallback. |
| Writing compressed at the end | Med | High | Draft chapters in parallel from W5; full draft to supervisor by W8. |
| Supervisor feedback arrives late | Med | Med | Send full draft end of W8 → leaves a full week of buffer. |
| Scope creep (too many secondary datasets) | High | Med | FaithBench/TofuEval are **optional**; only after RQ1–RQ3 are locked. |

---

## 8. Standing weekly routine

- **Monday:** review this plan, mark last week's boxes, set the 3 must-do items.
- **Mid-week:** one supervisor-update line (what ran, what's next, any blocker).
- **Friday:** commit + back up; write down any result that surprised you (raw material for Discussion).

---

## 9. Definition of done (final submission checklist)

- [ ] RQ1, RQ2, RQ3 each answered with a figure and a number.
- [ ] In-domain reproduction documented and within tolerance.
- [ ] All four methods on the cost-accuracy Pareto plot.
- [ ] Every table/figure regenerable from the committed code.
- [ ] Thesis proofread, formatted to SRH spec, declaration signed.
- [ ] PDF submitted by **Wed 2 Sep 2026**; repo tagged and archived.
