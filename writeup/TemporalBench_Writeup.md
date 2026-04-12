# TemporalBench: Measuring Temporal Reasoning in AI Systems

**Benchmark:** Measuring Progress Toward AGI — Cognitive Abilities
**Track:** Reasoning (temporal reasoning)
**Date:** April 2026

---

## The Problem

Standard AI benchmarks test *what* an AI knows, not *when* it knows it. MMLU, TruthfulQA, HumanEval — none of them systematically probe whether a system can reason about *time*: what was true at a specific point in the past, whether facts have changed, when they changed, and what the causal sequence of events was.

This matters for real-world AI deployment. A medical AI might confidently tell you the wrong drug dosage because it retrieved a fact from 3 years ago without realizing the guidelines updated 6 months ago. A legal AI might cite a case that was overruled without knowing it was overturned. An AI assistant might answer questions about your own life incorrectly because it mixes up old and new information.

Temporal reasoning is a core cognitive ability. We built TemporalBench to measure it rigorously.

---

## What TemporalBench Tests

TemporalBench evaluates AI systems on 6 task families across 4 difficulty versions:

| Task | What It Tests | Example |
|------|-------------|---------|
| **AsOfQA** | Retrieve what was true at a point in time | "Who was CEO of Apple on March 15, 2019?" |
| **ChangeDetection** | Detect whether facts changed between two days | "Did John Smith's title change between Jan 1 and Dec 31, 2021?" |
| **CausalQuery** | Trace causal chains through time | "What event caused the price spike on June 3rd?" |
| **Reversion** | Handle facts that flip back to an old value | "What was the capital of Germany on Aug 1, 1990?" (It was Bonn, then West Berlin, then Berlin after reunification) |
| **CausalReasoning** | Who held a role before someone else took over | "Who was CEO of Twitter before Parag Agrawal?" |
| **MultiReversion** | Facts that change multiple times | "Track every CEO of Twitter from 2015 to 2023" |

Versions v1–v4 increase difficulty: adversarial question ordering, noise injection, temporal distractors, overlapping validity windows.

---

## The System A Paradox

Our most striking finding: **AI systems fail on recent facts while succeeding on old ones.**

We call this the **System A paradox** — after the benchmark system we used as a baseline. When we split benchmark questions by temporal distance (how far in the past the answer sits), we found:

- **Near questions** (answer < 30 days before query): ~0% accuracy
- **Far questions** (answer > 365 days before query): ~73% accuracy

This is backwards from what you'd expect. Recent information should be more accessible, not less. The cause: standard retrieval-augmented systems store facts with simple timestamps or decay functions, but retrieve using proximity to the *current time* — not validity windows tied to *when the fact was actually true*. The system knows the query is "now," so it weights recent facts heavily, even when those recent facts are *changes* that invalidate older answers.

Near accuracy of 0% means the system gets *every* recent question wrong. For an AI deployed in healthcare, finance, or personal assistance, this is a critical failure mode.

---

## Validity Windows Beat Decay Functions

The standard approach to temporal knowledge is **decay functions** — facts become less "激活" (active) as they age, weighted by a decay curve. The intuition: recent facts matter more.

We tested this against **validity windows** — explicitly storing `valid_from` and `valid_until` timestamps for every fact, and retrieving based on whether the query date falls within the window.

**Result: validity windows consistently outperform decay.**

In our hardest version (v1), the validity-window system (D_revised) achieves a Temporal Reasoning Score of 0.68, vs 0.31 for the decay-based system. The difference is statistically significant (p < 0.001).

The ablation confirms the mechanism: removing validity windows from the best system collapses performance from 0.68 to 0.31. The windows aren't a nice-to-have — they're the entire mechanism.

Decay fails on near questions because it doesn't encode *when* a fact became true, only *how recently* it was accessed. Validity windows fail gracefully on near questions because the system knows the exact window during which each fact applies.

---

## Why This Matters for AGI

The System A paradox is not a bug in one system's implementation. It's a structural blind spot in how standard benchmarks and standard architectures handle time.

Current AGI evaluation is heavily focused on:
- **Knowledge retrieval** (what does the model know?)
- **Reasoning** (can it chain logic steps?)
- **Factuality** (is it hallucinating?)

Almost nothing tests **temporal metacognition** — does the system know *when* its knowledge applies, and does it correctly reason about change over time?

TemporalBench was designed to fill this gap. A system that scores well on TemporalBench demonstrates something beyond what MMLU measures: the ability to track facts through time, detect changes, reason about causality, and maintain an accurate model of how the world evolves.

This is closer to what we actually want from an AI assistant or AI agent operating in the real world.

---

## Key Statistics

- 4 difficulty versions × 3 random seeds × 6 task families = industry-standard evaluation depth
- 10+ million prediction rows across full ablation studies
- Statistically significant results on v1 (p < 0.001) confirming validity window superiority
- System A paradox replicated across seeds: near_accuracy = 0%, far_accuracy = 73% (robust finding)

---

## The Benchmark Is Open

TemporalBench is designed to be runnable by any AI system via a simple prompt interface. We provide:
- A Kaggle benchmark SDK submission with all 6 task families
- Clean data format (JSONL) with facts, questions, and ground truth
- An evaluator notebook that runs on Kaggle's infrastructure
- Ablation studies showing which components actually matter

**Leaderboard:** TemporalBench-v1 results show current SOTA models struggle significantly on near-timestamp questions — there's meaningful headroom for improvement.

---

## Conclusion

TemporalBench reveals a systematic temporal blindness in current AI systems. The System A paradox — 0% accuracy on recent facts, 73% on old ones — is invisible to standard benchmarks but catastrophic for real-world deployment.

Validity windows aren't just a better engineering choice than decay functions. They represent a fundamentally different model of time: facts as events with beginnings and ends, not as static knowledge with fading activation.

If we're building AI systems that need to operate accurately in fast-moving domains — medicine, law, finance, personal assistance — temporal competence isn't optional. TemporalBench is a step toward measuring whether we have it.
