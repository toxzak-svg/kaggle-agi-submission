"""
TemporalBench — AsOfQA Task
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: As of day X, what was true about subject Y?
Filters for AsOfQA (v1) and PastQueryTrap (v2/v3/v4) question types.
"""
!pip install kaggle-benchmarks -q

import kbench
import kbench.assertions

from .store import build_store, load_questions


@kbench.task(name="AsOfQA", description="As of day X, what was true about subject Y?")
def asofqa_task(llm, version: str = "v1", seed: int = 0) -> None:
    """AsOfQA: Query facts at a specific point in time using validity windows."""
    # Build temporal store from facts/events
    store = build_store(version=version, seed=seed)
    questions = load_questions(version=version, seed=seed)

    # Filter to As-of questions
    asof_questions = [
        q
        for q in questions
        if q.get("task_family") in ("AsOfQA", "PastQueryTrap")
    ]

    for q in asof_questions:
        as_of_day = q.get("as_of_day", 50)
        domain = q.get("domain", "")
        subject = q.get("subject", "")

        if not domain or not subject:
            continue

        # Query the store: what was true at as_of_day?
        predicted = store.get(domain, subject, as_of_day)
        expected = q.get("answer", "").strip()

        # LLM evaluates the question
        prompt = q.get("prompt", "").strip()
        response = llm.prompt(prompt).strip()

        # Assert: LLM response should be consistent with the correct answer
        # Use partial match since LLM may paraphrase
        kbench.assertions.assert_true(
            _answer_matches(expected, predicted, response),
            f"[AsOfQA] Day {as_of_day} | domain={domain} subject={subject} | "
            f"Expected: {expected} | Store: {predicted} | LLM: {response}",
        )


def _answer_matches(expected: str, predicted: str, response: str) -> bool:
    """
    Check if the answer is correct.
    Match modes (any order):
      1. expected content appears in LLM response
      2. LLM response is a substring of expected (shorter LLM answers)
      3. LLM response matches the stored value (predicted)
    """
    exp = expected.lower().strip()
    pred = (predicted or "").lower().strip()
    resp = response.lower().strip()

    if not exp:
        return False

    # Content match: answer in response (most flexible)
    if exp in resp or resp in exp:
        return True

    # Store-based match: predicted value in response
    if pred and (pred in resp or resp in pred):
        return True

    return False
