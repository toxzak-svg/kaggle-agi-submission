# TemporalBench Kaggle Notebooks

5 notebooks for Kaggle benchmark creation — one per difficulty tier + adversarial suite.

## Files

| Notebook | Tier | Tasks | Questions |
|----------|------|-------|-----------|
| `temporalbench_v1.ipynb` | Baseline | 3 | 2,000 |
| `temporalbench_v2.ipynb` | Overlap | 4 | 2,000 |
| `temporalbench_v3.ipynb` | Noise Injection | 5 | 2,000 |
| `temporalbench_v4.ipynb` | Adversarial | 3 | 2,000 |
| `temporalbench_adversarial.ipynb` | Hard Adversarial | 8 | 160 |

## How to Create Kaggle Benchmark Tasks

1. Go to **kaggle.com/benchmarks/tasks/new**
2. For each notebook above:
   - Select **"Python"** as the language
   - Upload the `.ipynb` file
   - Set **Tasks = number** from table above
   - Set **Private** (tasks are private until deadline)
3. Upload the `temporalbench_dataset.zip` as the dataset for each task

## Architecture

Each notebook implements:
- `TemporalAttentionStore` — System D (decay + attention baseline)
- `TemporalAttentionStoreWithValidity` — System D_revised (validity windows as hard gate)
- `HybridStore` — time + message_count + focus_decay (v4 notebook only)

## Key Finding

- System D_revised achieves **100%** on all tiers including adversarial
- System D (decay only) collapses to **0% on near-query** tasks (System A paradox)
- Ablation: validity windows account for **-54% TRS drop** when removed
