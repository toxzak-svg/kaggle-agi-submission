"""
TemporalBench — ReversionDetection (Adversarial) Task
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: As of day X, who was subject Y?
Adversarial facts that flip back — tests whether systems handle reversions.
Data: adversarial_temporal_questions.jsonl (top-level, not versioned).
"""
!pip install kaggle-benchmarks -q

import json
import os

import kbench
import kbench.assertions

from .store import get_data_dir


def _load_adversarial_questions():
    """Load the adversarial question set (one flat file, all seeds pooled)."""
    data_dir = get_data_dir()
    adv_path = os.path.join(data_dir, "adversarial_temporal_questions.jsonl")
    if not os.path.exists(adv_path):
        return []
    return [json.loads(l) for l in open(adv_path, encoding="utf-8")]


def _load_adversarial_facts():
    """Load the adversarial facts set."""
    data_dir = get_data_dir()
    adv_path = os.path.join(data_dir, "adversarial_temporal_facts.jsonl")
    if not os.path.exists(adv_path):
        return {}
    facts = {}
    for line in open(adv_path, encoding="utf-8"):
        f = json.loads(line)
        # Facts keyed by domain:subject for quick lookup
        domain = f.get("domain", "")
        content = f.get("content", "")
        parts = content.split(":")
        subject = parts[1] if len(parts) > 1 else ""
        key = f"{domain}:{subject}"
        facts[key] = content
    return facts


@kbench.task(
    name="ReversionDetection",
    description="As of day X, who was subject Y? (Adversarial reversion — facts flip back)",
)
def reversion_detection_task(llm) -> None:
    """ReversionDetection: Handles adversarial flipping facts that revert to prior values."""
    questions = _load_adversarial_questions()

    for q in questions:
        domain = q.get("domain", "")
        subject = q.get("subject", "")
        as_of_day = q.get("as_of_day", 50)

        if not domain or not subject:
            continue

        # Parse content to extract day from "domain:subject:d{day}:..."
        content = q.get("content", "")
        parts = content.split(":")
        # Expected answer is the full content string (what was true at as_of_day)
        expected = q.get("answer", "").strip()

        # LLM evaluation
        prompt = q.get("prompt", "").strip()
        response = llm.prompt(prompt).strip()

        # Partial match on the subject + value (ignoring the encoded day)
        resp_lower = response.lower()
        # Match if LLM returns the right subject and a reasonable value
        match = (
            expected.lower() in resp_lower
            or _content_match(expected, content, response)
        )

        kbench.assertions.assert_true(
            match,
            f"[Reversion] domain={domain} subject={subject} "
            f"as_of_day={as_of_day} | Expected={expected} | LLM: {response}",
        )


def _content_match(expected: str, content: str, response: str) -> bool:
    """Match expected answer in LLM response (handles rephrasing)."""
    resp_lower = response.lower()
    exp_lower = expected.lower()
    # Extract subject and value (ignoring encoded day)
    exp_parts = exp_lower.split(":")
    if len(exp_parts) >= 4:
        # domain:subject:dX:value → match subject and value
        subj = exp_parts[1]
        val = exp_parts[-1]
        return subj in resp_lower and val in resp_lower
    return exp_lower in resp_lower or exp_lower in resp_lower
