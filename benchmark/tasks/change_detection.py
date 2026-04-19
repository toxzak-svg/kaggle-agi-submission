"""
TemporalBench — ChangeDetection Task
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: Did anything change for domain:subject between day X and day Y?
Filters for ChangeDetection task_family.
"""
!pip install kaggle-benchmarks -q

import kbench
import kbench.assertions

from .store import build_store, load_questions


@kbench.task(name="ChangeDetection", description="Did anything change between day X and day Y?")
def change_detection_task(llm, version: str = "v1", seed: int = 0) -> None:
    """ChangeDetection: Binary detection of whether facts changed in a time range."""
    store = build_store(version=version, seed=seed)
    questions = load_questions(version=version, seed=seed)

    # Filter to ChangeDetection questions
    change_questions = [q for q in questions if q.get("task_family") == "ChangeDetection"]

    for q in change_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")

        # v2/v3 use start_day/end_day; v1 uses as_of_day for range
        start_day = q.get("start_day", 1)
        end_day = q.get("end_day", q.get("as_of_day", 50))

        if not domain or not subject:
            continue

        # Did anything change?
        changed = store.detect_change(domain, subject, start_day, end_day)

        # LLM evaluation
        prompt = q.get("prompt", "").strip()
        response = llm.prompt(prompt).strip()

        # Binary: yes/no response should match our ground truth
        resp_lower = response.lower()
        predicted_yes = changed
        response_yes = "yes" in resp_lower or "changed" in resp_lower or "true" in resp_lower
        correct = response_yes == predicted_yes

        kbench.assertions.assert_true(
            correct,
            f"[ChangeDetection] domain={domain} subject={subject} "
            f"[{start_day}, {end_day}] | Changed={changed} | LLM: {response}",
        )
