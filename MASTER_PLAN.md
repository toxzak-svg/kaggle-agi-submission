# TemporalBench — Kaggle AGI Submission Plan

**Competition:** [Measuring Progress Toward AGI - Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
**Track:** Reasoning (temporal reasoning sub-category)  
**Deadline:** April 16, 2026  
**Our Data:** `projects/time/` — 4 benchmark versions, 3 seeds each, 10M+ prediction rows, ablation studies, statistical significance testing

---

## The Story Your Data Tells

Your temporal intelligence research produced something the AGI evaluation literature doesn't have yet: **empirical proof that validity windows beat decay functions** for temporal reasoning in AI systems. Specifically:

- **System A paradox**: near_accuracy = 0%, far_accuracy = 73% — a dramatic demonstration that standard evaluation misses systematic temporal blindness
- **D_revised > D**: First proof that validity windows outperform decay on hard queries (p < 0.001 on v1)
- **Ablation**: Removing temporal intervals (D_no_intervals) collapses TRS from 0.68 → 0.31 — intervals are the mechanism
- **Multi-seed rigor**: 3 seeds × 4 versions × 5 systems = industry-standard evaluation depth

This is a legitimate paper-track-quality submission. Here's how to package it.

---

## Phase 1 — Kaggle Benchmark Skeleton (Days 1-2)
**Goal: Get the Kaggle benchmark code running with your existing data**

### 1.1 Create Directory Structure

```
projects/kaggle-agi-submission/
  benchmark/
    tasks/
      as_of.py              # "As of day X, who was CEO of Y?" → TemporalAccuracy
      change_detection.py   # "Did anything change in domain Z?" → ChangeDetectionF1
      causal_trace.py       # "If X had been true at day Y, would Z?" → CausalTraceAccuracy
      staleness.py          # "Is this fact stale?" → StalenessErrorRate
      reversion.py          # Adversarial reversion detection (from adversarial_temporal_*.jsonl)
    benchmark.py            # Bundles all tasks into TemporalBench-v1
    run_benchmark.py        # Executes on Kaggle models
    tasks.json              # Task manifest (IDs, versions, seeds)
    requirements.txt
  writeup/
    TemporalBench_Writeup.md
    TemporalBench_Writeup.pdf
  data/
    # Symlink or copy from projects/time/benchmarks/
    v1_seed0/
    v2_seed0/
    ...
  SUBMISSION.md             # This file
```

### 1.2 Kaggle Task Format

Kaggle benchmarks use the `@kbench.task` decorator. Your questions.jsonl maps directly:

```python
import kaggle_benchmarks as kbench

@kbench.task(name="temporal_as_of")
def temporal_as_of(llm, question_id: str, prompt: str, answer: str):
    """As-of queries: what was true at a given point in time?"""
    response = llm.prompt(prompt)
    kbench.assertions.assert_equal(
        answer, response,
        expectation="Temporal accuracy: correct entity at specified time"
    )
```

### 1.3 Key Mappings

| Your Metric | Kaggle Task | Data Source |
|-------------|-------------|-------------|
| TemporalAccuracy | `temporal_as_of` | `questions.jsonl` (as_of tasks) |
| StalenessErrorRate | `temporal_staleness` | derived from as_of questions |
| ChangeDetectionF1 | `temporal_change_detection` | `questions.jsonl` (change tasks) |
| CausalTraceAccuracy | `temporal_causal_trace` | `questions.jsonl` (counterfactual tasks) |
| TRS (composite) | computed from above | `per_seed_results.csv` |

### 1.4 Validate on Kaggle

Use the pre-installed `kaggle-benchmarks` library in a Kaggle notebook:
```
https://www.kaggle.com/benchmarks/tasks/new
```

Test with `kbench.llms["google/gemini-2.5-flash"]` as a baseline evaluator.

---

## Phase 2 — Writeup Draft (Days 3-5)
**Goal: Tell the story your data supports**

### 2.1 Cognitive Framing

Frame as: **"Testing whether AI systems can reason about time — validity windows vs decay functions"**

The DeepMind cognitive framework identifies 10 core cognitive abilities. Map yours to:
- **Reasoning** (primary): temporal reasoning about facts that change over time
- **Learning** (secondary): systems must learn temporal patterns from event sequences

### 2.2 Writeup Sections (6 scored criteria, 0-5 each)

#### Section 1: Motivation & Relevance (5 points)
- Why temporal reasoning matters for AGI
- Standard benchmarks (MMLU, Big-Bench) don't test time-varying facts
- Real-world stakes: AI systems making decisions based on stale information

#### Section 2: Task Quality & Design (5 points)
- 4 task families: As-Of, Change Detection, Causal Trace, Staleness Detection
- Adversarial reversion tasks that fool decay-based systems
- Progressive difficulty: v1 (pure staleness) → v4 (adversarial reversions)

#### Section 3: Dataset Quality (5 points)
- Synthetic but grounded: entities + events + temporal relations
- 4 versions × 3 seeds = 12 independent evaluation runs
- Total: ~100K+ questions, ~1M+ events/facts

#### Section 4: Evaluation Methodology (5 points)
- 5-metric evaluation framework (TemporalAccuracy, StalenessErrorRate, ChangeDetectionF1, CausalTraceAccuracy, TRS)
- Multi-seed evaluation with standard deviations
- Statistical significance testing (t-tests, confidence intervals)
- Ablation studies (D vs D_no_intervals vs D_no_decay vs D_no_rerank)

#### Section 5: Results & Analysis (5 points)
- **Key finding 1**: System A has 0% near_accuracy but 73% far_accuracy — standard evaluation misses temporal blindness
- **Key finding 2**: D_revised (validity windows) consistently beats D (decay) on hard queries (v1: p < 0.001, v3: p < 0.05)
- **Key finding 3**: Ablation proves temporal intervals are the active mechanism (0.31 → 0.68 TRS)
- **Key finding 4**: -0.71 temporal_distance correlation in System A — older facts handled better than recent ones

#### Section 6: Clarity & Presentation (5 points)
- Clear figures from per_question_correlation.csv and per_seed_results.csv
- Table format for ablation results
- Statistical significance reported with confidence intervals

### 2.3 Narrative Framework

**Opening hook**: "We asked AI systems questions about facts that change over time. What we found surprised us."

**The System A paradox**: "System A always fails on recent facts (near_accuracy = 0%) but handles old ones well (far_accuracy = 73%). This isn't a capability gap — it's a temporal blindness. The system has no mechanism for noticing that new information should override old information."

**The validity window breakthrough**: "The standard solution to staleness is decay functions — gradually forgetting information over time. We tested this directly. Decay functions fail on hard queries because they can't distinguish 'old but still valid' from 'old and superseded.' Validity windows — storing when information was true — consistently beat decay (p < 0.001)."

**The split cognition implication**: "Our finding suggests temporal reasoning should be split from generation. The model should generate; a separate temporal layer should handle validity. This is the split cognition pattern applied to time."

---

## Phase 3 — Adversarial Tasks Integration (Days 4-5)
**Goal: Show your benchmark generalizes to hard cases**

### 3.1 Adversarial Dataset

Your existing data:
- `projects/time/benchmarks/adversarial_temporal_questions.jsonl` (34K, Reversion tasks)
- `projects/time/benchmarks/adversarial_temporal_facts.jsonl` (31K)

### 3.2 Reversion Task

```python
@kbench.task(name="temporal_reversion")
def temporal_reversion(llm, question_id: str, prompt: str, answer: str):
    """Facts that flip back — tests whether systems can handle reversion patterns."""
    response = llm.prompt(prompt)
    kbench.assertions.assert_equal(
        answer, response,
        expectation="Correct entity after reversion event"
    )
```

### 3.3 Show D_revised Handles Reversions

From your data: v4 has no change events but tests reversion patterns. System C/D_revised handle these better than A/B.

---

## Phase 4 — Submission Packaging (Days 6-7)
**Goal: Everything the judges need is clean and reproducible**

### 4.1 Benchmark Submission

- [x] 1. Create `benchmark.json` listing all tasks  → `benchmark/tasks.json` (already existed)
- [x] 2. Zip `benchmark/` directory  → `benchmark.zip` created (35.8 KB)
- [ ] 3. Upload to Kaggle: `kaggle.com/benchmarks/temporalbench-v1`
- [ ] 4. Set all tasks **private** (auto-publish after April 16)

### 4.2 Writeup Finalization

- Convert `TemporalBench_Writeup.md` → PDF
- Keep under 8 pages (excluding appendix)
- Appendix: full ablation table, significance CSV excerpt

### 4.3 Data Packaging

Symlink or copy from `projects/time/benchmarks/`:
```
data/
  temporalbench_v1/    # v1_seed0, v1_seed1, v1_seed2
  temporalbench_v2/
  temporalbench_v3/
  temporalbench_v4/
  adversarial/
```

---

## Phase 5 — Submit & Verify (Days 8-9)
**Goal: Submitted, verified, backed up**

### 5.1 Primary Submission

1. Go to `kaggle.com/competitions/kaggle-measuring-agi`
2. Submit benchmark + writeup
3. Verify all tasks are private

### 5.2 Backup Plan

If Kaggle submission has issues:
- Push `kaggle-agi-submission/` to GitHub as a public repo
- Submit link to repo + writeup PDF hosted on GitHub Pages / HuggingFace

### 5.3 Paper Track (Optional)

The competition also has a **Paper Track** (separate from benchmark track). If you want:
- More formal academic paper format
- Link paper to your benchmark code submission
- Same deadline, evaluated on 6 criteria

---

## ⏱️ Timeline Summary

| Days | Phase | Deliverables |
|------|-------|-------------|
| 1-2 | Benchmark Skeleton | `benchmark/tasks/*.py`, `benchmark.py`, test on Kaggle |
| 3-5 | Writeup Draft | Full writeup with all 6 sections |
| 4-5 | Adversarial Integration | Reversion tasks added to benchmark |
| 6-7 | Packaging | `benchmark.json`, writeup PDF, data packaged |
| 8-9 | Submit + Verify | Kaggle submitted, GitHub backup ready |

---

## 🚨 Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Kaggle benchmark API changes | Use quick-start notebook (pre-installed `kaggle-benchmarks`) |
| Deadline slip | Submit by Day 7 to have buffer |
| Writeup too long | Cap at 8 pages (excluding appendix) |
| Competition format unclear | Reference existing submissions (Immeinen/kaggle-agi-benchmark on GitHub) |
| Not enough competition context | Read the DeepMind cognitive framework paper before writing |

---

## 📊 Evidence Map

| Writeup Section | Your Data |
|-----------------|-----------|
| Task Definitions | `questions.jsonl` + `facts.jsonl` (v1-v4, all seeds) |
| Ablation | `ablation_table.csv` (D vs D_no_intervals = 0.31 vs 0.68) |
| Statistical Significance | `significance.csv` (p < 0.001 on v1, p < 0.05 on v3) |
| Multi-seed Rigor | `per_seed_results.csv` (3 seeds, std reported) |
| System A Paradox | `per_question_correlation.csv` (-0.71 correlation, near_accuracy=0%) |
| Validity > Decay | `main_table_multi_seed.csv`: D_revised TRS=1.0 vs D TRS=0.68 on v1 |
| Adversarial Tasks | `adversarial_temporal_*.jsonl` (Reversion task family) |
| Key Narrative | Phase 2 results (D_revised > D on hard queries, p significant) |

---

## 📁 Key Files You'll Create

```
projects/kaggle-agi-submission/
├── benchmark/
│   ├── tasks/
│   │   ├── as_of.py
│   │   ├── change_detection.py
│   │   ├── causal_trace.py
│   │   ├── staleness.py
│   │   └── reversion.py
│   ├── benchmark.py
│   ├── run_benchmark.py
│   ├── tasks.json
│   └── requirements.txt
├── writeup/
│   ├── TemporalBench_Writeup.md
│   └── TemporalBench_Writeup.pdf
├── data/                          # Symlinks to projects/time/benchmarks/
├── assets/
│   ├── ablation_table.png
│   ├── significance_results.png
│   └── system_comparison.png
└── SUBMISSION.md                  # This file
```

---

## 🎯 Success Criteria

- [ ] Kaggle benchmark created with all 5 task types
- [ ] Writeup covers all 6 scored criteria with strong evidence
- [ ] Adversarial reversion tasks integrated
- [ ] Multi-seed statistical significance reported
- [ ] Submitted before April 14 (2-day buffer)
- [ ] GitHub backup repo created
