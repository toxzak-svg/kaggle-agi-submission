# TemporalBench — Kaggle AGI Submission

**Competition:** Measuring Progress Toward AGI — Cognitive Abilities
**Track:** Reasoning (temporal reasoning sub-category)
**Deadline:** April 16, 2026

---

## What This Benchmark Tests

We asked AI systems questions about facts that change over time. What we found: **standard evaluation misses a systematic failure mode** — models consistently fail on recent facts while handling old ones correctly.

The benchmark tests whether systems can:
1. Retrieve what was true at a specific point in time (AsOfQA / PastQueryTrap)
2. Detect whether facts changed over an interval (ChangeDetection)
3. Trace causal chains through time (CausalQuery)
4. Detect adversarial reversions — facts that flip back (Reversion)

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **System A paradox** | near_accuracy=0%, far_accuracy=73% — models fail on recent facts, succeed on old ones |
| **Validity windows beat decay** | D_revised (validity-only) consistently beats D (decay) on hard queries (p<0.001 on v1) |
| **Temporal intervals are the mechanism** | Ablation: removing intervals collapses TRS from 0.68→0.31 |
| **-0.71 temporal_distance correlation** | System A handles old facts better than recent ones |

---

## Benchmark Structure

### Versions

| Version | Difficulty | Task Families |
|---------|-----------|---------------|
| v1 | Easy | AsOfQA, ChangeDetection, CausalQuery |
| v2 | Adversarial | + PastQueryTrap, CurrentQuery |
| v3 | Hard | + DecayTrap, OverlapTrap (designed to fool decay systems) |
| v4 | Extreme | + OverlapTrap, PastQueryTrap (reversion-heavy) |

### Task Families

**AsOfQA / PastQueryTrap** — "As of day X, what was true about Y in domain Z?"
- TemporalAccuracy = fraction correctly retrieved
- 1200–2000 questions per version

**ChangeDetection** — "Did anything change for X between day Y and Z?"
- ChangeDetectionF1 = binary change detection accuracy
- 47–400 questions per version

**CausalQuery** — "Which change on day X caused the state on day Y for Z?"
- CausalTraceAccuracy = fraction where model identifies correct event
- 4–400 questions per version

**Reversion** — "As of day X, who was CEO of reversion_Y?"
- Adversarial reversions: facts that flip back to a prior value
- 160 questions (shared across versions)

---

## Metrics

### Composite: TRS (Temporal Retrieval Score)

**v1/v2:**
```
TRS = (TemporalAccuracy × ChangeDetectionF1 × CausalTraceAccuracy × (1 - StalenessErrorRate))^(1/4)
```

**v3/v4:**
```
TRS = (TemporalAccuracy × ChangeDetectionF1 × CausalTraceAccuracy × ReversionAccuracy)^(1/4)
```

### Individual Metrics

- **TemporalAccuracy**: fraction of AsOfQA questions answered correctly
- **StalenessErrorRate**: fraction of current facts incorrectly marked as stale (or vice versa)
- **ChangeDetectionF1**: binary change detection accuracy
- **CausalTraceAccuracy**: fraction of causal trace questions correct
- **ReversionAccuracy**: fraction of reversion questions correct

---

## How to Run

### Local (any OpenAI-compatible API)

```bash
cd benchmark
pip install -r requirements.txt

# Single version/seed
python run_benchmark.py --version v1 --seed 0

# All versions × seeds
python run_benchmark.py --all

# Custom model / local endpoint
python run_benchmark.py --version v3 --model mixtral --base-url http://localhost:11434
```

### Output

Results saved to `benchmark_results_{version}_s{seed}.json` or `benchmark_results_all.json`.

---

## Data Sources

- `projects/time/benchmarks/temporalbench_{v1,v2,v3,v4}_questions.jsonl` — question sets
- `projects/time/benchmarks/temporalbench_{v1,v2,v3,v4}_facts.jsonl` — fact timelines
- `projects/time/benchmarks/v{1,2,3,4}_seed{0,1,2}/` — per-seed data directories
- `projects/time/benchmarks/adversarial_temporal_questions.jsonl` — reversion tasks

---

## Files

```
kaggle-agi-submission/
├── benchmark/
│   ├── benchmark.py          # Main benchmark class + TRS scoring
│   ├── run_benchmark.py       # CLI runner
│   ├── tasks/
│   │   ├── as_of.py          # AsOfQA / PastQueryTrap scoring
│   │   ├── change_detection.py
│   │   ├── causal_trace.py
│   │   ├── staleness.py
│   │   ├── reversion.py
│   │   ├── data_utils.py     # Multi-version content parsing
│   │   └── task_routing.py   # Version → task family mapping
│   ├── tasks.json            # Task manifest
│   └── requirements.txt
└── README.md                  # This file
```