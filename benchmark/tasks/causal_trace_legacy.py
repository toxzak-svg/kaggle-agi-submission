"""
TemporalBench — CausalQuery (legacy run() for local benchmark.py)
"""

import json
import os
from . import data_utils

TASK_FAMILY = "CausalQuery"
DATA_VERSION = "v1"


def load_data(version: str = DATA_VERSION, seed: int = 0):
    kaggle_input = "/kaggle/input/temporalbench"
    bench_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_data_dir = os.path.join(bench_root, "..", "time", "benchmarks")
    data_dir = kaggle_input if os.path.isdir(kaggle_input) else local_data_dir
    q_path, _, _ = data_utils.get_data_paths(version, seed, data_dir)
    questions = [json.loads(l) for l in open(q_path, encoding="utf-8")]
    return questions, []


def score(question: dict, harness: dict = None) -> dict:
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
