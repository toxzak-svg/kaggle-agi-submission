# TemporalBench v2 — Kaggle Benchmark SDK Submission

**Competition:** Measuring Progress Toward AGI - Cognitive Abilities  
**Deadline:** April 16, 2026

---

## What Was Built

This folder (`kaggle-agi-submission-v2/`) contains a properly structured Kaggle benchmark SDK submission:

```
kaggle-agi-submission-v2/
├── temporalbench_evaluator.ipynb   # Main notebook — runs all tasks
├── kernel-metadata.json            # Kaggle kernel config
├── tasks/
│   ├── __init__.py
│   ├── as_of.py                   # AsOfQA tasks (v1, v2, v3)
│   ├── change_detection.py         # ChangeDetection tasks (v1, v2)
│   ├── causal_trace.py             # CausalQuery tasks (v1, v2)
│   └── reversion.py                # Reversion, CausalReasoning, MultiReversion
└── README.md                       # This file
```

## Tasks Created

| Task Name | What It Tests | Data |
|-----------|--------------|------|
| `TemporalBench-v1-AsOfQA` | Retrieve what was true at a point in time | v1_seed0 |
| `TemporalBench-v1-ChangeDetection` | Detect whether facts changed between two days | v1_seed0 |
| `TemporalBench-v1-CausalQuery` | Trace which event caused a later state | v1_seed0 |
| `TemporalBench-Reversion` | Handle non-monotonic timelines (flip-back patterns) | adversarial |
| `TemporalBench-CausalReasoning` | Who held a role before someone else took over | adversarial |
| `TemporalBench-MultiReversion` | Handle facts that change multiple times | adversarial |

## How to Submit

### Step 1: Upload Dataset to Kaggle
The dataset `zacharymaronek/temporalbench` already exists on Kaggle with v1-v4 data + adversarial data.

If needed, the data is at:
```
projects/kaggle-agi-submission/kaggle_data/
  v1_seed0/{questions.jsonl, facts.jsonl, events.jsonl}
  v2_seed0/...
  v3_seed0/...
  v4_seed0/...
  adversarial_temporal_questions.jsonl
  adversarial_temporal_facts.jsonl
```

### Step 2: Create the Notebook on Kaggle
1. Go to **kaggle.com/code** → **New Notebook**
2. Upload `temporalbench_evaluator.ipynb`
3. Create the `tasks/` folder and upload each task file
4. Add the dataset `zacharymaronek/temporalbench` to the notebook

### Step 3: Fix the Data Path
In each task file, the data root is:
```python
DATA_ROOT = "/kaggle_input/temporalbench"
```
On Kaggle, the dataset mounts at `/kaggle_input/{dataset-slug}` — so it should be:
```python
DATA_ROOT = "/kaggle_input/temporalbench"
```
This is already set correctly. Just make sure the dataset slug is `temporalbench`.

### Step 4: Run and Save Tasks
1. Run all cells — each task will produce results
2. Click **"Save Task"** on each task cell to register it with Kaggle
3. The last task with `%choose` determines the primary leaderboard score

### Step 5: Publish to Benchmark Collection
1. Go to **kaggle.com/benchmarks/zacharymaronek/temporal-tasks**
2. Or go to **kaggle.com/benchmarks** → **Create Benchmark**
3. Add each published task to the collection
4. Set the benchmark to **private** until you're ready to publish

## Key Insight to Emphasize in Writeup

The benchmark reveals the **System A paradox**: standard evaluation misses that models fail on *recent* facts while succeeding on *old* ones (0% near-accuracy, 73% far-accuracy). This is invisible to standard benchmarks like MMLU. Validity windows — storing `valid_from/valid_to` — fix this. Decay functions cannot.

## Files for Reference

- Paper: `projects/temporalbench/TemporalBench_Paper.md`
- Results: `projects/temporalbench/results/per_seed_results.csv`
- Competition: https://www.kaggle.com/competitions/kaggle-measuring-agi
