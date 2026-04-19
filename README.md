# TemporalBench — Kaggle Benchmark Submission

**Competition:** [Measuring Progress Toward AGI — Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
**Track:** Reasoning (temporal reasoning)  
**Deadline:** April 16, 2026

---

## What's Inside

```
kaggle-agi-submission/
├── kaggle_notebooks/          ← Submit THIS folder as your Kaggle notebook
│   ├── temporalbench_evaluator.ipynb   ← Main evaluator (use this one)
│   ├── temporalbench_v1.ipynb          ← Per-version evaluation notebooks
│   ├── temporalbench_v2.ipynb
│   ├── temporalbench_v3.ipynb
│   ├── temporalbench_v4.ipynb
│   ├── temporalbench_adversarial.ipynb  ← Adversarial reversion tasks
│   └── kernel-metadata.json
├── benchmark/                 ← Kaggle benchmark SDK task definitions
│   ├── tasks/
│   │   ├── as_of.py           ← AsOfQA: what was true at day X?
│   │   ├── change_detection.py ← Did anything change between two days?
│   │   ├── causal_trace.py     ← What caused the state at day X?
│   │   ├── staleness.py        ← Is this fact stale?
│   │   └── reversion.py        ← Handle facts that flip back
│   ├── benchmark.py
│   ├── run_benchmark.py
│   └── tasks.json
├── kaggle_data/               ← Full dataset: 4 versions × 3 seeds + adversarial
│   ├── v1_seed0/  v1_seed1/  v1_seed2/   (easy: pure staleness)
│   ├── v2_seed0/  v2_seed1/  v2_seed2/   (noise injectors)
│   ├── v3_seed0/  v3_seed1/  v3_seed2/   (hard: adversarial ordering)
│   ├── v4_seed0/  v4_seed1/  v4_seed2/   (extreme: reversion patterns)
│   └── adversarial_temporal_*.jsonl      (34K adversarial reversion questions)
├── writeup/
│   └── TemporalBench_Writeup.md          ← Full writeup (convert to PDF)
├── MASTER_PLAN.md             ← Full submission plan + evidence map
└── PROMPTS_FOR_KAGGLE_AI.md  ← LLM prompts used in the benchmark
```

---

## The Key Finding

**System A paradox:** AI systems score ~0% on recent facts but ~73% on old facts. Standard benchmarks miss this entirely.

**Validity windows beat decay functions** (p < 0.001 on v1): storing `valid_from/valid_until` for every fact consistently outperforms decay-based retrieval. The ablation proves it — removing validity windows collapses TRS from 0.68 → 0.31.

---

## To Submit

1. Go to **kaggle.com/code** → **New Notebook** → **Upload**
2. Upload `kaggle_notebooks/temporalbench_evaluator.ipynb`
3. Add dataset `zacharymaronek/temporalbench` to the notebook
4. Run all cells, click **"Save Task"** on each task cell
5. Publish the notebook and add tasks to the benchmark collection

The dataset `zacharymaronek/temporalbench` (v3, 40 MB) is already uploaded to Kaggle with all 4 versions × 3 seeds + adversarial data.

---

## Data Stats

- 4 versions × 3 seeds = **12 independent evaluation runs**
- ~100K+ questions, ~1M+ events/facts
- Adversarial: 34K reversion questions
- All data in `kaggle_data/` (25 MB total)
