"""TemporalBench CausalQuery task — Model D with validity windows + event causal chains."""
import json
import os
import kaggle_benchmarks as kbench


DATA_ROOT = "/kaggle_input/temporalbench"


def load_data(version="v1", seed=0):
    base = os.path.join(DATA_ROOT, f"{version}_seed{seed}")
    q_path = os.path.join(base, "questions.jsonl")
    f_path = os.path.join(base, "facts.jsonl")
    e_path = os.path.join(base, "events.jsonl")

    questions = []
    if os.path.exists(q_path):
        with open(q_path) as f:
            questions = [json.loads(l) for l in f]

    facts = []
    if os.path.exists(f_path):
        with open(f_path) as f:
            facts = [json.loads(l) for l in f]

    events = []
    if os.path.exists(e_path):
        with open(e_path) as f:
            events = [json.loads(l) for l in f]

    return questions, facts, events


def parse_fact_content(content: str):
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    day = int(parts[2].replace("d", ""))
    value = parts[3] if len(parts) > 3 else parts[-1]
    return domain, subject, day, value


def get_value_at_day(domain, subject, day, facts):
    """Most recent fact at or before day (validity window aware)."""
    best_day = -1
    best = None
    for f in facts:
        if f.get("domain") != domain or f.get("subject") != subject:
            continue
        valid_from = int(f.get("t_valid_from", 0))
        valid_until = int(f.get("t_valid_until", -1))
        if day < valid_from:
            continue
        if valid_until != -1 and day > valid_until:
            continue
        if valid_from > best_day:
            best_day = valid_from
            best = f
    return best


def trace_supersedes(event_id: str, events_by_id: dict) -> list:
    """Trace the full supersedes chain back to the root event."""
    chain = []
    current_id = event_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        event = events_by_id.get(current_id)
        if event is None:
            break
        chain.append(event)
        current_id = event.get("supersedes")
    return chain


def build_causal_prompt(question: dict, events: list, current_event: dict | None) -> str:
    """Build prompt with causal chain context from events."""
    prompt = question.get("prompt", "").strip()

    if not events:
        return prompt + "\n\n[No causal events found. Answer based on your knowledge.]"

    # Build event chain description
    ctx = ["[Causal chain — events leading to the outcome]"]
    for i, evt in enumerate(events[:6]):  # top 6 events in chain
        t = evt.get("t_event", "?")
        val = evt.get("value", "")
        supersedes = evt.get("supersedes")
        ctx.append(f"  Event {i+1} (day {t}): {val}")
        if supersedes:
            ctx.append(f"    └─ superseded: {supersedes}")

    return (
        f"{prompt}\n\n"
        + "\n".join(ctx)
        + "\n\nBased on the causal chain above, identify which event caused the final outcome. "
        + "The answer should be the event ID (e.g. e114) or a description of the root cause."
    )


def score_answer(expected, response):
    if not expected or not response:
        return 0.0
    exp = expected.strip().lower()
    resp = response.strip().lower()
    if exp == resp or exp in resp or resp in exp:
        return 1.0
    exp_tokens = set(exp.split())
    resp_tokens = set(resp.split())
    overlap = len(exp_tokens & resp_tokens) / max(len(exp_tokens), 1)
    return 1.0 if overlap > 0.7 else 0.0


@kbench.task(name="TemporalBench-v1-CausalQuery", description="Which event caused the state on day Y? (Model D + causal chains)")
def temporalbench_v1_causalquery(llm, seed: int = 0) -> float:
    """CausalQuery v1: trace event chains using validity windows + supersedes."""
    questions, facts, events = load_data("v1", seed)
    cq_questions = [q for q in questions if q.get("task_family") == "CausalQuery"]

    # Index events by ID for fast lookup
    events_by_id = {e.get("event_id"): e for e in events}

    correct = 0
    total = len(cq_questions)

    for q in cq_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        outcome_day = int(q.get("outcome_day", 0))
        expected_event_id = q.get("answer", "").strip()

        # Trace the causal chain from the outcome event backward
        chain = trace_supersedes(expected_event_id, events_by_id)

        # Get current state at outcome_day
        current_fact = get_value_at_day(domain, subject, outcome_day, facts)

        prompt = build_causal_prompt(q, chain, current_fact)
        response = llm.prompt(prompt).strip()

        correct += score_answer(expected_event_id, response)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@kbench.task(name="TemporalBench-v2-CausalQuery", description="Which event caused the state on day Y? (v2, Model D)")
def temporalbench_v2_causalquery(llm, seed: int = 0) -> float:
    """CausalQuery v2: adversarial causal chains."""
    questions, facts, events = load_data("v2", seed)
    cq_questions = [q for q in questions if q.get("task_family") == "CausalQuery"]

    events_by_id = {e.get("event_id"): e for e in events}

    correct = 0
    total = len(cq_questions)

    for q in cq_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        outcome_day = int(q.get("outcome_day", 0))
        expected_event_id = q.get("answer", "").strip()

        chain = trace_supersedes(expected_event_id, events_by_id)
        current_fact = get_value_at_day(domain, subject, outcome_day, facts)

        prompt = build_causal_prompt(q, chain, current_fact)
        response = llm.prompt(prompt).strip()

        correct += score_answer(expected_event_id, response)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
