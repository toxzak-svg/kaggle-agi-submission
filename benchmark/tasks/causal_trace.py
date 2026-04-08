"""
TemporalBench — Kaggle Benchmark Tasks
https://www.kaggle.com/competitions/kaggle-measuring-agi

Task: Causal Query / Counterfactual
" If X had been true at day Y, would Z have been true at day Z?"
"""

import json
import os
from typing import Any
from . import data_utils

TASK_FAMILY = "CausalQuery"
DATA_VERSION = "v1"


def load_data(version: str = DATA_VERSION, seed: int = 0):
    bench_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(bench_root, "..", "time", "benchmarks")
    q_path, _, _ = data_utils.get_data_paths(version, seed, data_dir)
    questions = [json.loads(l) for l in open(q_path, encoding="utf-8")]
    return questions, []


def score(question: dict, harness: dict = None) -> dict[str, Any]:
    expected = question.get("answer", "").strip()
    response = question.get("response", "").strip()
    correct = expected in response or response in expected
    return {
        "correct": correct,
        "expected": expected,
        "response": response,
        "question_id": question["question_id"],
        "task_family": TASK_FAMILY,
    }


def run(llm, version: str = DATA_VERSION, seed: int = 0):
    questions, _ = load_data(version, seed)

    correct = 0
    total = 0
    rows = []

    for q in questions:
        if q.get("task_family") != TASK_FAMILY:
            continue

        prompt = q["prompt"].strip()
        response = llm.prompt(prompt).strip()
        q["response"] = response
        result = score(q, {})
        result["prompt"] = prompt
        result["response"] = response
        rows.append(result)

        if result["correct"]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "task_family": TASK_FAMILY,
        "version": version,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "rows": rows,
    }