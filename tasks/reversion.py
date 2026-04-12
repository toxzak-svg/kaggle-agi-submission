"""TemporalBench Reversion task — Model D with validity windows."""
import json
import os
import kaggle_benchmarks as kbench


DATA_ROOT = "/kaggle_input/temporalbench"


def load_adversarial():
    """Load adversarial questions and facts."""
    base = DATA_ROOT
    adv_q_path = os.path.join(base, "adversarial_temporal_questions.jsonl")
    adv_f_path = os.path.join(base, "adversarial_temporal_facts.jsonl")

    questions = []
    if os.path.exists(adv_q_path):
        with open(adv_q_path) as f:
            questions = [json.loads(l) for l in f]

    facts = []
    if os.path.exists(adv_f_path):
        with open(adv_f_path) as f:
            facts = [json.loads(l) for l in f]

    return questions, facts


def parse_fact_content(content: str):
    """Parse 'domain:subject:d{day}:value' string."""
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    day = int(parts[2].replace("d", ""))
    value = parts[3] if len(parts) > 3 else parts[-1]
    return domain, subject, day, value


def retrieve_validity_window(domain, subject, as_of_day, facts):
    """
    Model D retrieval: find the most recent fact where as_of_day falls
    within [t_valid_from, t_valid_until].
    """
    candidates = []
    for f in facts:
        if f.get("domain") != domain or f.get("subject") != subject:
            continue
        valid_from = int(f.get("t_valid_from", 0))
        valid_until = int(f.get("t_valid_until", -1))
        if as_of_day < valid_from:
            continue
        if valid_until != -1 and as_of_day > valid_until:
            continue
        candidates.append((f, valid_from))

    if not candidates:
        return None

    # Most recent = highest t_valid_from
    best = max(candidates, key=lambda x: x[1])
    return best[0]


def build_prompt(question: dict, fact_answer: str | None) -> str:
    prompt = question.get("prompt", "").strip()
    if fact_answer is None:
        return prompt + "\n\n[No relevant fact found. Answer based on your knowledge.]"
    return (
        f"{prompt}\n\n"
        f"[Fact from knowledge base]: {fact_answer}\n\n"
        f"Answer the question above using the fact provided."
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


@kbench.task(name="TemporalBench-Reversion", description="Adversarial reversion: facts that flip back (Model D validity windows)")
def temporalbench_reversion(llm) -> float:
    """Reversion: handle non-monotonic timelines using validity windows."""
    questions, facts = load_adversarial()
    rev_questions = [q for q in questions if q.get("task_family") == "Reversion"]

    correct = 0
    total = len(rev_questions)

    for q in rev_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        # Model D: validity-window retrieval
        fact = retrieve_validity_window(domain, subject, as_of_day, facts)

        if fact is not None:
            _, _, _, value = parse_fact_content(fact["content"])
            answer_from_fact = value
        else:
            answer_from_fact = None

        prompt = build_prompt(q, answer_from_fact)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        correct += score_answer(expected, response)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@kbench.task(name="TemporalBench-CausalReasoning", description="Who held a role before someone else took over? (Model D)")
def temporalbench_causalreasoning(llm) -> float:
    """CausalReasoning: who held a role before someone else took over."""
    questions, facts = load_adversarial()
    cr_questions = [q for q in questions if q.get("task_family") == "CausalReasoning"]

    correct = 0
    total = len(cr_questions)

    for q in cr_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        # Model D: validity-window retrieval
        fact = retrieve_validity_window(domain, subject, as_of_day, facts)

        if fact is not None:
            _, _, _, value = parse_fact_content(fact["content"])
            answer_from_fact = value
        else:
            answer_from_fact = None

        prompt = build_prompt(q, answer_from_fact)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        correct += score_answer(expected, response)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@kbench.task(name="TemporalBench-MultiReversion", description="Facts that flip multiple times (Model D)")
def temporalbench_multireversion(llm) -> float:
    """MultiReversion: facts that change multiple times, Model D tracks validity windows."""
    questions, facts = load_adversarial()
    mr_questions = [q for q in questions if q.get("task_family") == "MultiReversion"]

    correct = 0
    total = len(mr_questions)

    for q in mr_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = int(q.get("as_of_day", 0))

        # Model D: validity-window retrieval
        fact = retrieve_validity_window(domain, subject, as_of_day, facts)

        if fact is not None:
            _, _, _, value = parse_fact_content(fact["content"])
            answer_from_fact = value
        else:
            answer_from_fact = None

        prompt = build_prompt(q, answer_from_fact)
        response = llm.prompt(prompt).strip()
        expected = q.get("answer", "").strip()

        correct += score_answer(expected, response)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
