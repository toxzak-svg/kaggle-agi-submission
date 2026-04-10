"""
TemporalBench — ChangeDetection (legacy run() for local benchmark.py)
"""

import json
import os
from . import data_utils

TASK_FAMILY = "ChangeDetection"
DATA_VERSION = "v1"


def load_data(version: str = DATA_VERSION, seed: int = 0):
    kaggle_input = "/kaggle/input/temporalbench"
    bench_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_data_dir = os.path.join(bench_root, "..", "time", "benchmarks")
    data_dir = kaggle_input if os.path.isdir(kaggle_input) else local_data_dir
    q_path, f_path, e_path = data_utils.get_data_paths(version, seed, data_dir)

    questions = [json.loads(l) for l in open(q_path, encoding="utf-8")]
    facts = [json.loads(l) for l in open(f_path, encoding="utf-8")]

    events = []
    if os.path.exists(e_path):
        events = [json.loads(l) for l in open(e_path, encoding="utf-8")]

    return questions, facts, events


def build_harness(version: str = DATA_VERSION, seed: int = 0):
    questions, facts, events = load_data(version, seed)

    timeline: dict[tuple, list[tuple]] = {}
    for f in facts:
        domain, subject, day, value = data_utils.parse_content_flexible(f["content"])
        key = (domain, subject)
        if key not in timeline:
            timeline[key] = []
        timeline[key].append((day, value))
        timeline[key].sort(key=lambda x: x[0])

    return {"questions": questions, "facts": facts, "events": events, "timeline": timeline}


def detect_change(domain: str, subject: str, start_day: int, end_day: int, timeline: dict) -> bool:
    key = (domain, subject)
    if key not in timeline:
        return False
    values_in_range = [(d, v) for d, v in timeline[key] if start_day < d <= end_day]
    return len(values_in_range) > 1


def score(question: dict, harness: dict) -> dict:
    domain = question["domain"]
    subject = question["subject"]
    start_day = question.get("start_day", 1)
    end_day = question.get("end_day", question.get("as_of_day", 50))

    changed = detect_change(domain, subject, start_day, end_day, harness["timeline"])

    expected = "yes" if changed else "no"
    response = question.get("response", "").lower().strip()
    correct = ("yes" in response) == changed or ("no" in response) == (not changed)

    return {
        "correct": correct,
        "expected": expected,
        "question_id": question["question_id"],
        "task_family": TASK_FAMILY,
    }


def run(llm, version: str = DATA_VERSION, seed: int = 0):
    harness = build_harness(version, seed)
    questions = harness["questions"]

    correct = 0
    total = 0
    rows = []

    for q in questions:
        if q.get("task_family") != TASK_FAMILY:
            continue

        prompt = q["prompt"].strip()
        response = llm.prompt(prompt).strip()
        q["response"] = response
        result = score(q, harness)
        result["prompt"] = prompt
        result["response"] = response
        rows.append(result)

        if result["correct"]:
            correct += 1
        total += 1

    f1 = correct / total if total > 0 else 0.0
    return {
        "task_family": TASK_FAMILY,
        "version": version,
        "seed": seed,
        "accuracy": f1,
        "correct": correct,
        "total": total,
        "rows": rows,
    }
