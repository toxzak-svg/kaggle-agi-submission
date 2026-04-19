"""
TemporalBench — CausalQuery / CausalTrace Task
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: Which change on day X caused the state on day Y for subject Z?
Answers are event IDs (e.g. "e114").
"""
!pip install kaggle-benchmarks -q

import kbench
import kbench.assertions

from .store import build_store, load_questions


@kbench.task(name="CausalQuery", description="Which change caused the state on day Y?")
def causal_trace_task(llm, version: str = "v1", seed: int = 0) -> None:
    """CausalQuery: Identify which event caused a downstream state."""
    store = build_store(version=version, seed=seed)
    questions = load_questions(version=version, seed=seed)

    # Filter to CausalQuery questions
    causal_questions = [q for q in questions if q.get("task_family") == "CausalQuery"]

    for q in causal_questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        action_day = q.get("action_day", 1)
        outcome_day = q.get("outcome_day", 3)

        if not domain or not subject:
            continue

        # Ground truth: most recent event at action_day for this subject
        event = store.latest_change_event(domain, subject, before_day=outcome_day)
        expected = q.get("answer", "").strip()  # e.g. "e114"
        event_id = event.get("event_id", "") if event else ""

        # LLM evaluation
        prompt = q.get("prompt", "").strip()
        response = llm.prompt(prompt).strip()

        # Match: expected event ID or store event ID appears in response
        resp_lower = response.lower()
        matches = (
            expected.lower() in resp_lower
            or event_id.lower() in resp_lower
            or expected.lower() in event_id.lower()
        )

        kbench.assertions.assert_true(
            matches,
            f"[CausalQuery] domain={domain} subject={subject} "
            f"action_day={action_day} outcome_day={outcome_day} | "
            f"Expected={expected} StoreEvent={event_id} | LLM: {response}",
        )
