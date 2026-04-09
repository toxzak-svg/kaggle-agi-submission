# TemporalBench Dataset

**TemporalBench: Validity Windows Beat Decay Functions in Temporal Reasoning**

A synthetic benchmark and evaluation framework for temporal reasoning in AI memory systems.

## Files

```
temporalbench/
├── benchmark/          # Python benchmark code
│   ├── benchmark.py    # Main TemporalBench class
│   ├── run_benchmark.py
│   └── tasks/         # AsOfQA, ChangeDetection, CausalQuery, StalenessDetection, Reversion
├── data/              # Benchmark data
│   └── benchmarks/    # v1, v2, v3, v4 question/fact/event files + adversarial
└── README.md          # This file
```

## Quick Start

```bash
cd temporalbench/benchmark
pip install openai
python run_benchmark.py --version v1 --seed 0
```

## Benchmark Structure

- **4 difficulty tiers**: v1 (baseline), v2 (overlap), v3 (noise), v4 (adversarial)
- **5 task families**: AsOfQA, ChangeDetection, CausalQuery, StalenessDetection, Reversion
- **3 seeds per configuration** (60+ runs per system)
- **35+ tasks** across 4 suites

## Citation

If you use this benchmark, cite:
```
Maronek, Z. "TemporalBench: Validity Windows Beat Decay Functions in Temporal Reasoning."
github.com/toxzak-svg/temporal-attention
```

## Key Findings

- System A paradox: near-query accuracy=0%, far-query accuracy=73% under decay-based systems
- D_revised (validity windows) achieves 100% on all tiers; decay-based systems fail on adversarial
- Ablation: removing validity windows collapses TRS from 0.68 to 0.31 (-54%)
- r=-0.71 temporal distance correlation under decay; vanishes (r=-0.08) under D_revised