"""
TemporalBench — StalenessDetection Task
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: Is this fact stale or current? Has it been superseded?
Derived from AsOfQA — if the latest fact for domain:subject is after as_of_day, it's stale.
"""
!pip install kaggle-benchmarks -q

import kbench
import kbench.assertions

from .store import build_store, load_questions


@kbench.task(name="StalenessDetection", description="Is this fact stale or current?")
def staleness_task(llm, version: str = "v1", seed: int = 0) -> None:
    """StalenessDetection: Binary classification — is the queried fact stale?"""
    store = build_store(version=version, seed=seed)
    questions = load_questions(version=version, seed=seed)

    # Filter to AsOfQA/PastQueryTrap (staleness derived from these)
    asof_questions = [
        q
        for q in questions
        if q.get("task_family") in ("AsOfQA", "PastQueryTrap")
    ]

    for q in asof_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = q.get("as_of_day", 50)

        if not domain or not subject:
            continue

        # Stale = latest fact for this domain:subject is AFTER as_of_day
        key = (domain, subject)
        latest_day = None
        if key in store.timeline:
            latest_day = max(day for day, _ in store.timeline[key])

        stale = latest_day is not None and latest_day > as_of_day

        # LLM evaluation: ask directly
        prompt = f"Is this fact stale or current? {q.get('prompt', '')}"
        response = llm.prompt(prompt).strip()

        resp_lower = response.lower()
        # LLM says "stale" if it thinks it's old/superseded
        llm_says_stale = "stale" in resp_lower or "outdated" in resp_lower or "old" in resp_lower
        correct = llm_says_stale == stale

        kbench.assertions.assert_true(
            correct,
            f"[Staleness] domain={domain} subject={subject} "
            f"as_of_day={as_of_day} latest_day={latest_day} stale={stale} | "
            f"LLM: {response}",
        )
