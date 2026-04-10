# PROMPTS_FOR_KAGGLE_AI.md
# Copy-paste these prompts directly into Kaggle's AI code generator.
# Generated for: TemporalBench — Measuring Progress Toward AGI (Cognitive Abilities Track)

---

## PART 0 — OVERALL BENCHMARK STRUCTURE

### Prompt: Build the Benchmark Shell

```
Create a new Kaggle benchmark called "TemporalBench" for the competition
"kaggle-measuring-agi" (Measuring Progress Toward AGI - Cognitive Abilities).

This benchmark tests whether AI systems can reason about time-varying facts.
It has 5 task families: as_of, change_detection, causal_trace, staleness, reversion.

Create the following files in the benchmark directory:

1. benchmark.py — Main TemporalBench class with:
   - name = "TemporalBench"
   - run(llm, version, seed) -> dict
   - run_all(llm) -> dict  [runs all version × seed combos]
   - _compute_trs(results, version) -> float  [geometric mean composite]

2. run_benchmark.py — CLI runner with:
   - --version {v1,v2,v3,v4} (default: v1)
   - --seed {0,1,2} (default: 0)
   - --all flag to run everything
   - --model, --base-url, --api-key for LLM configuration

3. requirements.txt:
   openai>=1.0.0
   (the kaggle-benchmarks library is pre-installed in the Kaggle environment)

The llm parameter must have a .prompt(text: str) -> str method.
The benchmark reads data from the attached dataset at:
   /kaggle/input/temporalbench/{version}_seed{seed}/questions.jsonl
   /kaggle/input/temporalbench/{version}_seed{seed}/facts.jsonl
   /kaggle/input/temporalbench/{version}_seed{seed}/events.jsonl

Use the @kbench.task decorator for each task. Use kbench.assertions.assert_equal
for exact answer checking.

Save results as JSON with these fields:
{
  "task_family": str,
  "version": str,
  "seed": int,
  "accuracy": float,          # or staleness_error_rate for staleness
  "correct": int,
  "total": int,
  "TRS": float               # composite Temporal Retrieval Score
}
```

---

## PART 1 — TASK: as_of (Temporal As-Of Queries)

### Prompt: Create as_of.py

```
Create /kaggle/working/as_of.py for the TemporalBench benchmark.

This task answers: "As of day X, what was true about Y in domain Z?"
Mapped from task_family: "AsOfQA" (v1) and "PastQueryTrap" (v2/v3).

DATA FORMAT:
questions.jsonl — one JSON object per line:
{
  "question_id": "q1",
  "task_family": "AsOfQA",           # or "PastQueryTrap" in v2/v3
  "prompt": "As of day 43, what was true about model_3 in medium?",
  "as_of_day": 43,
  "domain": "medium",
  "subject": "model_3",
  "answer": "medium:model_3:d43:v17"  # exact expected answer
}

facts.jsonl — one JSON object per line:
{
  "fact_id": "f1",
  "content": "slow:hardware_0:d1:v1",  # format: domain:subject:d{day}:v{version}
  "t_valid_from": 1,
  "t_valid_until": 1,
  "domain": "slow",
  "confidence": 0.879,
  "decay_fn": "domain_half_life"
}

VERSION DIFFERENCES:
- v1: content like "slow:hardware_0:d1:v1", as_of_day in range 1-60
- v2: content like "slow:hardware_0:day1:v1:7311", uses "day" not "d"
- v3/v4: content like "slow:hw_0:d1:v1:5242", mixed formats

Write a `load_data(version, seed)` function that reads from:
  /kaggle/input/temporalbench/{version}_seed{seed}/questions.jsonl
  /kaggle/input/temporalbench/{version}_seed{seed}/facts.jsonl

Write a `build_harness(version, seed)` function that:
  1. Loads questions and facts
  2. Builds a dict index: key=(domain, subject, day) → value=content string

Write a `score_question(question, index)` function that:
  1. Looks up all facts for (domain, subject) where day <= as_of_day
  2. Picks the one with the highest day (most recent)
  3. Returns: {"correct": bool, "expected": str, "answer": str, "question_id": str}

Write a `run(llm, version, seed)` function that:
  1. Calls build_harness(version, seed)
  2. Iterates all questions where task_family is "AsOfQA" or "PastQueryTrap"
     (use: q.get("task_family") in ("AsOfQA", "PastQueryTrap"))
  3. For each question: calls llm.prompt(q["prompt"]) then score_question()
  4. Returns: {"task_family": "AsOfQA", "version": v, "seed": s,
               "accuracy": correct/total, "correct": c, "total": t, "rows": [...]}

SUCCESS CRITERIA:
- Correct if: the answer string matches the expected answer EXACTLY (after strip)
- v1: ~1200 AsOfQA questions
- v2: ~1600 PastQueryTrap questions
- v3/v4: mix of PastQueryTrap + other types, filter by task_family
- Accuracy = fraction of correct answers

EDGE CASES:
- Some questions may have no matching fact before as_of_day → answer "UNKNOWN"
- Answer format varies: "domain:subject:d{day}:v{val}" (strip and compare)
- In v2 answers may include extra suffix like ":7311" — strip the day/version part
- Handle both "AsOfQA" and "PastQueryTrap" task_family values

EXAMPLE GOOD OUTPUT:
{
  "task_family": "AsOfQA",
  "version": "v1",
  "seed": 0,
  "accuracy": 0.375,
  "correct": 450,
  "total": 1200,
  "rows": [
    {"question_id": "q1", "correct": true, "expected": "medium:model_3:d43:v17",
     "answer": "medium:model_3:d43:v17", "prompt": "As of day 43..."},
    {"question_id": "q2", "correct": false, "expected": "medium:api_13:d1:v1",
     "answer": "medium:api_13:d2:v2", "prompt": "As of day 1..."}
  ]
}
```

---

## PART 2 — TASK: change_detection

### Prompt: Create change_detection.py

```
Create /kaggle/working/change_detection.py for the TemporalBench benchmark.

This task answers: "Did anything change for X between day Y and day Z?"
(task_family: "ChangeDetection")

DATA FORMAT:
questions.jsonl:
{
  "question_id": "q1806",
  "task_family": "ChangeDetection",
  "prompt": "What changed for hardware_19 between day 1 and day 13?",
  "start_day": 1,
  "end_day": 13,
  "domain": "slow",
  "subject": "hardware_19",
  "answer": "slow:hardware_19:day1:v1:1916 -> slow:hardware_19:day13:v3:2916"
}

facts.jsonl:
{
  "fact_id": "f1",
  "content": "slow:hardware_0:d1:v1",
  "t_valid_from": 1,
  "t_valid_until": 1,
  "domain": "slow",
  "confidence": 0.879,
  "decay_fn": "domain_half_life"
}

Write `load_data(version, seed)` that reads:
  /kaggle/input/temporalbench/{version}_seed{seed}/questions.jsonl
  /kaggle/input/temporalbench/{version}_seed{seed}/facts.jsonl
  /kaggle/input/temporalbench/{version}_seed{seed}/events.jsonl  (optional)

Write `build_harness(version, seed)` that:
  1. Loads facts
  2. Builds a timeline dict: key=(domain, subject) → sorted list of (day, value)

Write `detect_change(domain, subject, start_day, end_day, timeline)`:
  - Returns True if there are ≥2 different values in (start_day, end_day] range
  - Returns False otherwise (including if key not in timeline)

Write `score(question, harness)`:
  - Call detect_change with the question's domain, subject, start_day, end_day
  - Expected: "yes" if changed else "no"
  - Parse response: look for "yes" or "no" (case-insensitive substring match)
  - correct = ("yes" in response.lower() == expected == "yes")

Write `run(llm, version, seed)`:
  1. Filter questions where task_family == "ChangeDetection"
  2. For each question: llm.prompt(prompt), score(), accumulate results
  3. Return {"task_family": "ChangeDetection", "accuracy": correct/total,
             "correct": c, "total": t, "rows": [...]}

SUCCESS CRITERIA:
- F1 metric = correct/total (binary: yes/no detection)
- Accuracy near 1.0 expected for well-designed systems (ChangeDetection is easy)
- Handle v1 format (day numbers in "d1" format) and v2 format ("day1" format)
- v2 questions use start_day/end_day; some may use as_of_day for range

EDGE CASES:
- If no fact exists for (domain, subject) → treat as no change
- "Did anything change" vs "What changed" — respond to both prompt styles
- Some versions use "as_of_day" instead of start_day/end_day
  → treat as range (1, as_of_day]

EXAMPLE GOOD OUTPUT:
{
  "task_family": "ChangeDetection",
  "version": "v1",
  "seed": 0,
  "accuracy": 1.0,
  "correct": 400,
  "total": 400,
  "rows": [
    {"question_id": "q1806", "correct": true, "expected": "yes", "response": "yes",
     "prompt": "What changed for hardware_19 between day 1 and day 13?"}
  ]
}
```

---

## PART 3 — TASK: causal_trace

### Prompt: Create causal_trace.py

```
Create /kaggle/working/causal_trace.py for the TemporalBench benchmark.

This task answers counterfactual/causal questions:
"If X had been true at day Y, would Z have been true at day Z?"
(task_family: "CausalQuery")

DATA FORMAT:
questions.jsonl — CausalQuery questions:
{
  "question_id": "cq1",
  "task_family": "CausalQuery",
  "prompt": "If the hardware_0 value had been v5 on day 3, what would the status on day 7 have been?",
  "answer": "status_0:d7:v12"   # or similar causal answer
}

This task is the simplest: exact string match on the answer.

Write `run(llm, version, seed)`:
  1. Load questions from: /kaggle/input/temporalbench/{version}_seed{seed}/questions.jsonl
  2. Filter: q.get("task_family") == "CausalQuery"
  3. For each: call llm.prompt(q["prompt"]), compare response to q["answer"]
  4. Correct if: answer in response OR response in answer (partial match)
     OR exact match after strip().lower()

Return:
  {"task_family": "CausalQuery", "accuracy": correct/total,
   "correct": c, "total": t, "rows": [...]}

SUCCESS CRITERIA:
- Accuracy measured as fraction where expected answer appears in model's response
- Use partial match (substring) since causal answers can vary in format
- v1 has ~400 CausalQuery questions; v2/v3 have ~4; v4 has 0

EDGE CASES:
- CausalQuery questions are rare in v2/v3/v4 — check task_family field carefully
- Some questions may lack task_family → skip them
- Empty response → incorrect

EXAMPLE GOOD OUTPUT:
{
  "task_family": "CausalQuery",
  "version": "v1",
  "seed": 0,
  "accuracy": 1.0,
  "correct": 400,
  "total": 400,
  "rows": [
    {"question_id": "cq1", "correct": true, "expected": "status_0:d7:v12",
     "response": "Based on the causal chain, the answer is status_0:d7:v12"}
  ]
}
```

---

## PART 4 — TASK: staleness

### Prompt: Create staleness.py

```
Create /kaggle/working/staleness.py for the TemporalBench benchmark.

This task answers: "Is this fact stale or current? Has it been superseded?"
Derived from AsOfQA — a fact is stale if there is a newer fact after as_of_day.

DATA FORMAT:
Uses the SAME questions.jsonl as as_of.py (AsOfQA / PastQueryTrap questions).
Does NOT need its own data file.

The staleness detection works by:
1. Take an AsOfQA question (asks what was true at as_of_day)
2. Check if there's a newer fact (day > as_of_day) for same domain+subject
3. If yes → the "current" fact is stale (query was about old info)
4. The model's response to "Is this stale?" reveals whether it detects staleness

Write `is_stale(domain, subject, as_of_day, timeline)`:
  - timeline: dict of (domain, subject) → sorted list of (day, value)
  - Returns True if there exists any fact with day > as_of_day for same key
  - Returns False if as_of_day >= latest_day for that key

Write `build_harness(version, seed)`:
  - Loads facts.jsonl from: /kaggle/input/temporalbench/{version}_seed{seed}/facts.jsonl
  - Builds timeline dict
  - Loads questions from same dir (questions.jsonl)
  - Returns {"timeline": timeline, "questions": [AsOfQA questions]}

Write `run(llm, version, seed)`:
  1. Build harness
  2. Iterate AsOfQA / PastQueryTrap questions
  3. For each question: determine if stale via is_stale()
  4. Prompt the LLM: "Is this fact stale or current? {original_prompt}"
  5. Check: correct if ("stale" in response.lower() == stale)
  6. Return staleness_error_rate = 1.0 - (correct/total)

IMPORTANT: This returns staleness_error_rate (lower is better), NOT accuracy.

Return:
  {"task_family": "StalenessDetection",
   "staleness_error_rate": 1.0 - (correct/total),  # Key metric!
   "correct": c, "total": t, "rows": [...]}

SUCCESS CRITERIA:
- StalenessErrorRate = fraction of wrong staleness judgments
- Well-designed systems (D_revised) get 0.0 staleness_error_rate
- Poor systems (System A) get ~0.95+ staleness_error_rate
- Lower is better: 0.0 = perfect, 1.0 = always wrong

EDGE CASES:
- If domain+subject not in timeline → treat as not stale
- If multiple facts exist with same day → take the last one loaded
- The "Is this stale?" prefix must be consistent across all questions

EXAMPLE GOOD OUTPUT:
{
  "task_family": "StalenessDetection",
  "version": "v1",
  "seed": 0,
  "staleness_error_rate": 0.0,    # 0% error rate = perfect
  "correct": 1200,
  "total": 1200,
  "rows": [
    {"question_id": "q1", "correct": true, "stale": true,
     "prompt": "Is this fact stale or current? As of day 43...",
     "response": "This fact is stale"}
  ]
}
```

---

## PART 5 — TASK: reversion (Adversarial)

### Prompt: Create reversion.py

```
Create /kaggle/working/reversion.py for the TemporalBench benchmark.

This is the ADVERSARIAL task: facts that flip back to a prior value.
"As of day X, who was CEO of reversion_Y?"
These questions fool decay-based systems (which forget old info)
and test whether validity-window systems can handle reversions.

DATA FORMAT:
adversarial_temporal_questions.jsonl — one JSON object per line:
{
  "question_id": "adv_q0",
  "task_family": "Reversion",
  "prompt": "As of day 50, who was CEO of reversion_0?",
  "as_of_day": 50,
  "domain": "ceo",
  "subject": "reversion_0",
  "answer": "ceo:reversion_0:d41:Alice"   # note: reverts to Alice, not the current CEO
}

There are ~160 reversion questions in the adversarial file.
These are the SAME across all versions/seeds (shared adversarial set).

Write `load_data()`:
  - Reads from: /kaggle/input/temporalbench/adversarial_temporal_questions.jsonl
  - (No seed dimension — shared across all versions)
  - Returns list of question dicts

Write `build_harness()`:
  - Returns {"questions": loaded_questions}

Write `score(question, harness=None)`:
  - expected = question["answer"].strip()
  - response = question.get("response", "").strip()
  - correct = expected in response OR response in expected (partial match)
  - Also accept: extract name from answer (e.g., "Alice") appearing in response

Write `run(llm, version="v1", seed=0)`:
  - Ignores version/seed for reversion (uses shared adversarial file)
  - Loads adversarial questions
  - For each: llm.prompt(q["prompt"]), store response, score
  - Return: {"task_family": "Reversion", "accuracy": correct/total,
              "correct": c, "total": t, "rows": [...]}

SUCCESS CRITERIA:
- ReversionAccuracy = fraction correct
- Decay-based systems (D) score near 0% on reversion (they've forgotten the old value)
- Validity-window systems (D_revised) score 100% (they know when values were true)
- This is the hardest/most adversarial task type

EDGE CASES:
- Answer format: "domain:subject:d{day}:{name}" — extract the name part for flexible matching
- E.g., "Alice" should match "ceo:reversion_0:d41:Alice"
- Some responses may include explanation text — accept partial match on name
- reversion_0 through reversion_N subjects — handle any number

EXAMPLE GOOD OUTPUT:
{
  "task_family": "Reversion",
  "version": "v1",
  "seed": 0,
  "accuracy": 1.0,
  "correct": 160,
  "total": 160,
  "rows": [
    {"question_id": "adv_q0", "correct": true,
     "expected": "ceo:reversion_0:d41:Alice", "response": "Alice",
     "prompt": "As of day 50, who was CEO of reversion_0?"}
  ]
}
```

---

## PART 6 — WIRING TASKS TOGETHER

### Prompt: Create benchmark.py

```
Create /kaggle/working/benchmark.py — the main benchmark class.

This wires all 5 task families together and computes the composite TRS score.

TASK_RUNNERS = {
    "AsOfQA":            run_as_of,         # from as_of import run
    "ChangeDetection":   run_change,         # from change_detection import run
    "CausalQuery":       run_causal,        # from causal_trace import run
    "StalenessDetection": run_staleness,    # from staleness import run
    "Reversion":         run_reversion,     # from reversion import run
}

VERSION_TASK_TYPES = {
    "v1": ["AsOfQA", "ChangeDetection", "CausalQuery"],
    "v2": ["AsOfQA", "ChangeDetection", "CausalQuery"],  # PastQueryTrap mapped to AsOfQA
    "v3": ["AsOfQA", "ChangeDetection", "CausalQuery"],
    "v4": ["AsOfQA", "ChangeDetection", "CausalQuery"],
}

class TemporalBench:
    name = "TemporalBench"

    def run(self, llm, version="v1", seed=0):
        """
        Run all applicable task families for a given version/seed.
        llm: any object with llm.prompt(text: str) -> str method
        Returns dict with task_results and TRS composite score.
        """
        # For each task_type in VERSION_TASK_TYPES[version]:
        #   runner = TASK_RUNNERS[task_type]
        #   results[task_type] = runner(llm, version=version, seed=seed)
        # Compute TRS via _compute_trs(results, version)
        # Return {"version": v, "seed": s, "task_results": results, "TRS": trs}

    def run_all(self, llm):
        """Run all combinations: v1/v2/v3/v4 × seed 0/1/2."""
        # Returns dict keyed by (version, seed)

    def _compute_trs(self, task_results, version):
        """
        Compute TRS (Temporal Retrieval Score) = geometric mean of core metrics.

        v1/v2:
          ta = task_results["AsOfQA"]["accuracy"]
          cdf = task_results["ChangeDetection"]["accuracy"]
          cta = task_results["CausalQuery"]["accuracy"]
          ser = task_results["StalenessDetection"]["staleness_error_rate"]
          trs = (ta * cdf * cta * (1 - ser)) ** 0.25

        v3/v4:
          ta = task_results["AsOfQA"]["accuracy"]
          cdf = task_results["ChangeDetection"]["accuracy"]
          cta = task_results["CausalQuery"]["accuracy"]
          rev = task_results["Reversion"]["accuracy"]
          trs = (ta * cdf * cta * rev) ** 0.25

        Return round(trs, 4).
        If any metric is missing or 0, return 0.0.
        """

    def summary_table(self, all_results):
        """Return markdown table: version | seed | TA | ChangeF1 | CausalAcc | RevAcc | TRS"""

    def save_results(self, all_results, path):
        """Save results + key findings to JSON."""
```

---

## PART 7 — VALIDATION

### Prompt: Create validation script

```
Create /kaggle/working/validate.py to validate the benchmark runs correctly.

Run this AFTER creating all task files to verify everything works.

Validation checks:
1. All 5 task files exist (as_of.py, change_detection.py, causal_trace.py,
   staleness.py, reversion.py)
2. All can be imported without errors
3. Each task's run() function accepts (llm, version, seed) and returns a dict
4. The returned dict has required fields: task_family, accuracy (or staleness_error_rate),
   correct, total, rows
5. benchmark.py's TemporalBench class exists and run() works
6. TRS computation produces a float between 0 and 1

Use a dummy LLM (class that returns "dummy_response" from prompt()) for testing.
Run with: python validate.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (print which ones).

Test on version="v1", seed=0 (smallest dataset).
Expect at least 100 questions per task type for v1.
```

---

## PART 8 — SUBMISSION PACKAGING

### Prompt: Package instructions for Kaggle submission

```
FINAL SUBMISSION STEPS for Kaggle:

1. ZIP THE BENCHMARK:
   cd /kaggle/working
   zip -r temporalbench_benchmark.zip as_of.py change_detection.py \
       causal_trace.py staleness.py reversion.py benchmark.py \
       run_benchmark.py validate.py requirements.txt

2. CREATE THE DATASET:
   The dataset should contain:
   - temporalbench_v1_seed0/, temporalbench_v1_seed1/, temporalbench_v1_seed2/
   - temporalbench_v2_seed0/, temporalbench_v2_seed1/, temporalbench_v2_seed2/
   - temporalbench_v3_seed0/, temporalbench_v3_seed1/, temporalbench_v3_seed2/
   - temporalbench_v4_seed0/, temporalbench_v4_seed1/, temporalbench_v4_seed2/
   - adversarial_temporal_questions.jsonl  (shared, same for all)

   Each seed directory contains: questions.jsonl, facts.jsonl, events.jsonl

3. UPLOAD TO KAGGLE:
   - Go to kaggle.com/benchmarks/tasks/new
   - Create 5 tasks:
     a) "temporal_as_of" — uses as_of.py, dataset = temporalbench
     b) "temporal_change_detection" — uses change_detection.py, dataset = temporalbench
     c) "temporal_causal_trace" — uses causal_trace.py, dataset = temporalbench
     d) "temporal_staleness" — uses staleness.py, dataset = temporalbench
     e) "temporal_reversion" — uses reversion.py, dataset = temporalbench
   - Set each task PRIVATE (auto-publish after April 16)
   - Use Python language

4. VERIFY:
   - In Kaggle notebook environment, run each task and verify it completes
   - Check that TRS computed by benchmark.py matches expected ranges:
     * D_revised systems: TRS ~1.0 on all versions
     * System A: TRS ~0.37 on v1, ~0.59 on v2
     * D systems: TRS ~0.68 on v1 (validity windows help even without perfect recall)

5. SUBMISSION:
   - Submit to competition: kaggle-measuring-agi
   - Ensure all tasks are set to private
   - Deadline: April 16, 2026 (submit by April 14 for buffer)
```

---

## QUICK REFERENCE: Data File Locations

```
projects/time/benchmarks/
  temporalbench_v1_questions.jsonl     # v1 questions (top-level, same for all seeds)
  temporalbench_v1_facts.jsonl          # v1 facts
  temporalbench_v1_events.jsonl         # v1 events
  temporalbench_v2_questions.jsonl
  temporalbench_v2_facts.jsonl
  temporalbench_v2_events.jsonl
  temporalbench_v3_questions.jsonl
  temporalbench_v3_facts.jsonl
  temporalbench_v3_events.jsonl
  temporalbench_v4_questions.jsonl
  temporalbench_v4_facts.jsonl
  temporalbench_v4_events.jsonl
  v1_seed0/questions.jsonl, facts.jsonl, events.jsonl   # per-seed dirs (v1 has these)
  v1_seed1/...
  v1_seed2/...
  v2_seed0/...  (same structure for v2, v3, v4 × seeds 0,1,2)
  v3_seed0/...
  v4_seed0/...
  adversarial_temporal_questions.jsonl  # 160 reversion questions (shared)
  adversarial_temporal_facts.jsonl      # corresponding facts
```

---

## QUICK REFERENCE: Answer Format by Version

```
v1 facts:     "slow:hardware_0:d1:v1"           # d{day} format
v2 facts:     "slow:hardware_0:day1:v1:7311"    # day{day} format, extra suffix
v3 facts:     "slow:hw_0:d1:v1:5242"            # abbreviated subject, d{day}
v4 facts:     "fast:sts0:d1:v1"                  # abbreviated everything

v1 answers:   "domain:subject:d{day}:v{val}"    # e.g. "medium:model_3:d43:v17"
v2 answers:   "domain:subject:day{day}:v{val}:{extra}"  # e.g. "medium:api_13:day7:v3:7358"
v3/v4 answers: similar to v1 format with extra suffix

For comparison: strip all versions to core format "domain:subject:d{day}:v{val}"
then compare substrings. For reversion: extract the name (last part after :).
```

---

## QUICK REFERENCE: Expected Results

```
System A  (poor temporal reasoning):   TRS ~0.37 on v1, near_accuracy=0%, far_accuracy=73%
System B:                               TRS ~1.0 on all versions
System C:                               TRS ~1.0 on all versions
System D  (decay only):                 TRS ~0.68 on v1, collapses on hard queries
System D_revised (validity windows):    TRS ~1.0 on all versions, including adversarial

Ablation (D vs variants):
  D:              TRS = 0.68
  D_no_decay:    TRS = 0.66
  D_no_intervals: TRS = 0.31  ← biggest drop when temporal intervals removed
  D_no_rerank:   TRS = 0.33
```

---

## EDGE CASES CHECKLIST

- [ ] v2 uses "day1" not "d1" in content strings — handle with regex r"day(\d+)"
- [ ] v2/v3 answers include extra numeric suffix — strip before comparison
- [ ] Staleness task returns staleness_error_rate (lower=better), NOT accuracy
- [ ] CausalQuery is rare in v2/v3/v4 — don't require it to exist
- [ ] Reversion uses shared adversarial file (ignores version/seed)
- [ ] Some facts may have t_valid_until=None → treat as valid forever
- [ ] Questions with no matching fact → expected answer "UNKNOWN"
- [ ] All string comparisons should be case-insensitive and strip whitespace
