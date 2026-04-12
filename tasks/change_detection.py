"""TemporalBench ChangeDetection task — Model D with validity windows."""
import json
import os
import kaggle_benchmarks as kbench


DATA_ROOT = "/kaggle_input/temporalbench"


def load_data(version="v1", seed=0):
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
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    day = int(parts[2].replace("d", ""))
    value = parts[3] if len(parts) > 3 else parts[-1]
    return domain, subject, day, value


def get_value_at_day(domain, subject, day, facts):
    """Get the most recent valid fact for domain:subject at a specific day."""
    best_day = -1
    best = None
    for f in facts:
        d, s, d_day, val = parse_fact(f["content"])
        if d == domain and s == subject and d_day <= day:
            if d_day > best_day:
                best_day = d_day
                best = val
    return best


def build_prompt(question: dict, value_start: str | None, value_end: str | None) -> str:
    """Build prompt with retrieved values at both time points for comparison."""
    prompt = question.get("prompt", "").strip()
    ctx_parts = []
    if value_start is not None:
        ctx_parts.append(f"At the earlier time: {value_start}")
    if value_end is not None:
        ctx_parts.append(f"At the later time: {value_end}")
    if ctx_parts:
        return prompt + "\n\n[Context]\n" + "\n".join(ctx_parts) + "\n\nDid the value change? Answer yes or no, and if yes, specify what changed."
    return prompt + "\n\n[No relevant facts found. Answer based on your knowledge.]"


def score_answer(expected, response):
    if not expected or not response:
        return 0.0
    exp = expected.strip().lower()
    resp = response.strip().lower()

    if exp == resp:
        return 1.0

    if "->" in exp:
        parts = exp.split("->")
        start_val = parts[0].strip()
        end_val = parts[1].strip()
        if start_val in resp and end_val in resp:
            return 1.0
        if "changed" in resp and start_val in resp:
            return 1.0

    if "nothing" in exp or "no change" in exp:
        if "nothing" in resp or "no change" in resp or "same" in resp:
            return 1.0

    return 0.0


@kbench.task(name="TemporalBench-v1-ChangeDetection", description="Did anything change for subject between day X and Y? (Model D)")
def temporalbench_v1_changedetection(llm, seed: int = 0) -> float:
    """ChangeDetection v1: compare values at two time points using validity windows."""
    questions, facts = load_data("v1", seed)
    cd_questions = [q for q in questions if q.get("task_family") == "ChangeDetection"]

    correct = 0
    total = len(cd_questions)

    for q in cd_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        day_start = int(q.get("day_start", 0))
        day_end = int(q.get("day_end", 0))

        # Model D: retrieve values at both time points
        value_start = get_value_at_day(domain, subject, day_start, facts)
        value_end = get_value_at_day(domain, subject, day_end, facts)

        prompt = build_prompt(q, value_start, value_end)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        correct += score_answer(expected, response)

    f1 = correct / total if total > 0 else 0.0
    return f1


@kbench.task(name="TemporalBench-v2-ChangeDetection", description="Did anything change for subject between day X and Y? (v2, Model D)")
def temporalbench_v2_changedetection(llm, seed: int = 0) -> float:
    """ChangeDetection v2: adversarial ordering, overlapping windows."""
    questions, facts = load_data("v2", seed)
    cd_questions = [q for q in questions if q.get("task_family") == "ChangeDetection"]

    correct = 0
    total = len(cd_questions)

    for q in cd_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        day_start = int(q.get("day_start", 0))
        day_end = int(q.get("day_end", 0))

        value_start = get_value_at_day(domain, subject, day_start, facts)
        value_end = get_value_at_day(domain, subject, day_end, facts)

        prompt = build_prompt(q, value_start, value_end)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        correct += score_answer(expected, response)

    f1 = correct / total if total > 0 else 0.0
    return f1
