"""TemporalBench AsOfQA task — Model D with validity windows."""
import json
import os
import kaggle_benchmarks as kbench


DATA_ROOT = "/kaggle_input/temporalbench"


def load_data(version="v1", seed=0):
    """Load questions and facts for a given version/seed."""
    base = os.path.join(DATA_ROOT, f"{version}_seed{seed}")
    q_path = os.path.join(base, "questions.jsonl")
    f_path = os.path.join(base, "facts.jsonl")

    questions = []
    facts = []

    if os.path.exists(q_path):
        with open(q_path) as f:
            questions = [json.loads(l) for l in f]

    if os.path.exists(f_path):
        with open(f_path) as f:
            facts = [json.loads(l) for l in f]

    return questions, facts


def parse_fact(content: str):
    """Parse 'domain:subject:d{day}:value' string."""
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    day = int(parts[2].replace("d", ""))
    value = parts[3] if len(parts) > 3 else parts[-1]
    return domain, subject, day, value


def best_answer(domain, subject, as_of_day, facts):
    """Find the most recent valid fact for domain:subject at as_of_day."""
    best_day = -1
    best = None
    for f in facts:
        d, s, day, val = parse_fact(f["content"])
        if d == domain and s == subject and day <= as_of_day:
            if day > best_day:
                best_day = day
                best = val
    return best


def build_prompt(question: dict, fact_answer: str | None) -> str:
    """Build enriched prompt with retrieved fact context."""
    prompt = question.get("prompt", "").strip()
    if fact_answer is None:
        return prompt + "\n\n[No relevant fact found. Answer based on your knowledge.]"
    return (
        f"{prompt}\n\n"
        f"[Fact from knowledge base]: {fact_answer}\n\n"
        f"Answer the question above using the fact provided."
    )


def score_answer(expected, response):
    """Score whether response matches expected answer."""
    if not expected or not response:
        return 0.0
    exp = expected.strip().lower()
    resp = response.strip().lower()
    if exp == resp:
        return 1.0
    if exp in resp or resp in exp:
        return 1.0
    exp_tokens = set(exp.split())
    resp_tokens = set(resp.split())
    overlap = len(exp_tokens & resp_tokens) / max(len(exp_tokens), 1)
    return 1.0 if overlap > 0.7 else 0.0


@kbench.task(name="TemporalBench-v1-AsOfQA", description="As of day X, what was true about subject Y? (Model D — validity windows)")
def temporalbench_v1_asofqa(llm, seed: int = 0) -> float:
    """AsOfQA: retrieve correct entity using validity windows."""
    questions, facts = load_data("v1", seed)
    asof_questions = [q for q in questions if q.get("task_family") == "AsOfQA"]

    correct = 0
    total = len(asof_questions)

    for q in asof_questions:
        # Parse domain/subject/day from question
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        # Model D retrieval: validity-window lookup
        fact_answer = best_answer(domain, subject, as_of_day, facts)

        # Build enriched prompt with retrieved fact
        prompt = build_prompt(q, fact_answer)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        s = score_answer(expected, response)
        kbench.assertions.assert_true(
            s > 0,
            f"Expected '{expected}', got '{response}'"
        )
        correct += s

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@kbench.task(name="TemporalBench-v2-AsOfQA", description="As of day X, what was true about subject Y? (v2 adversarial, Model D)")
def temporalbench_v2_asofqa(llm, seed: int = 0) -> float:
    """AsOfQA v2: overlapping windows, past query traps — Model D handles these with validity windows."""
    questions, facts = load_data("v2", seed)
    asof_questions = [q for q in questions if q.get("task_family") in ("AsOfQA", "PastQueryTrap")]

    correct = 0
    total = len(asof_questions)

    for q in asof_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        fact_answer = best_answer(domain, subject, as_of_day, facts)
        prompt = build_prompt(q, fact_answer)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        s = score_answer(expected, response)
        correct += s

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@kbench.task(name="TemporalBench-v3-AsOfQA", description="As of day X, what was true about subject Y? (v3 hard, Model D)")
def temporalbench_v3_asofqa(llm, seed: int = 0) -> float:
    """AsOfQA v3: noise injection, decay traps, overlap traps."""
    questions, facts = load_data("v3", seed)
    asof_questions = [q for q in questions if q.get("task_family") in ("AsOfQA", "DecayTrap", "OverlapTrap")]

    correct = 0
    total = len(asof_questions)

    for q in asof_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        fact_answer = best_answer(domain, subject, as_of_day, facts)
        prompt = build_prompt(q, fact_answer)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        s = score_answer(expected, response)
        correct += s

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
