"""
TemporalBench — Kaggle Benchmark Tasks
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: Staleness Detection
" Is this fact stale or current? Has it been superseded?"
Derived from AsOfQA — answer differs from latest fact = staleness error.
"""

import json
import os
from typing import Any
from . import data_utils

TASK_FAMILY = "StalenessDetection"
DATA_VERSION = "v1"


def load_data(version: str = DATA_VERSION, seed: int = 0):
    bench_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(bench_root, "..", "time", "benchmarks")
    q_path, f_path, _ = data_utils.get_data_paths(version, seed, data_dir)
    questions = [json.loads(l) for l in open(q_path, encoding="utf-8")]
    facts = [json.loads(l) for l in open(f_path, encoding="utf-8")]
    return questions, facts


def build_harness(version: str = DATA_VERSION, seed: int = 0):
    questions, facts = load_data(version, seed)
    timeline: dict[tuple, list[tuple]] = {}
    for f in facts:
        domain, subject, day, value = data_utils.parse_content_flexible(f["content"])
        key = (domain, subject)
        if key not in timeline:
            timeline[key] = []
        timeline[key].append((day, value))
    for key in timeline:
        timeline[key].sort(key=lambda x: x[0])
    return {"questions": questions, "facts": facts, "timeline": timeline}


def is_stale(domain: str, subject: str, as_of_day: int, timeline: dict) -> bool:
    key = (domain, subject)
    if key not in timeline:
        return False
    latest_day = max(day for day, _ in timeline[key])
    return latest_day > as_of_day


def run(llm, version: str = DATA_VERSION, seed: int = 0):
    harness = build_harness(version, seed)
    questions = harness["questions"]

    correct = 0
    total = 0
    rows = []

    for q in questions:
        if q.get("task_family") != "AsOfQA":
            continue

        prompt = f"Is this fact stale or current? {q['prompt']}"
        response = llm.prompt(prompt).strip()
        q["response"] = response

        domain = q["domain"]
        subject = q["subject"]
        as_of_day = q.get("as_of_day", 50)
        stale = is_stale(domain, subject, as_of_day, harness["timeline"])

        response_lc = response.lower()
        correct_guess = ("stale" in response_lc and stale) or ("current" in response_lc and not stale)

        rows.append({
            "correct": correct_guess,
            "stale": stale,
            "question_id": q["question_id"],
            "task_family": TASK_FAMILY,
            "prompt": prompt,
            "response": response,
        })

        if correct_guess:
            correct += 1
        total += 1

    staleness_error_rate = 1.0 - (correct / total) if total > 0 else 1.0
    return {
        "task_family": TASK_FAMILY,
        "version": version,
        "seed": seed,
        "staleness_error_rate": staleness_error_rate,
        "correct": correct,
        "total": total,
        "rows": rows,
    }