"""
TemporalBench — AsOfQA (legacy run() for local benchmark.py)
"""

import json
import os
from . import data_utils
from . import task_routing

TASK_FAMILY = "AsOfQA"
DATA_VERSION = "v1"


def load_data(version: str = DATA_VERSION, seed: int = 0):
    kaggle_input = "/kaggle/input/temporalbench"
    bench_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_data_dir = os.path.join(bench_root, "..", "time", "benchmarks")
    data_dir = kaggle_input if os.path.isdir(kaggle_input) else local_data_dir
    q_path, f_path, _ = data_utils.get_data_paths(version, seed, data_dir)
    questions = [json.loads(l) for l in open(q_path, encoding="utf-8")]
    facts = [json.loads(l) for l in open(f_path, encoding="utf-8")]
    return questions, facts


def build_harness(version: str = DATA_VERSION, seed: int = 0):
    questions, facts = load_data(version, seed)
    index: dict[tuple, str] = {}
    for f in facts:
        domain, subject, day, _ = data_utils.parse_content_flexible(f["content"])
        key = (domain, subject, day)
        index[key] = f["content"]
    return {"questions": questions, "facts": facts, "index": index}


def score_question(question: dict, index: dict) -> dict:
    domain = question["domain"]
    subject = question["subject"]
    as_of_day = question.get("as_of_day", 50)

    best_key = None
    best_day = -1
    for (d, s, day), ans in index.items():
        if d == domain and s == subject and day <= as_of_day:
            if day > best_day:
                best_day = day
                best_key = (d, s, day)

    expected = index.get(best_key, "UNKNOWN") if best_key else "UNKNOWN"
    correct = question.get("answer", "").strip() == expected

    return {
        "correct": correct,
        "answer": expected,
        "expected": question.get("answer", "").strip(),
        "question_id": question["question_id"],
        "task_family": question.get("task_family", "AsOfQA"),
        "as_of_day": as_of_day,
    }


def run(llm, version: str = DATA_VERSION, seed: int = 0):
    harness = build_harness(version, seed)
    questions = harness["questions"]

    correct = 0
    total = 0
    rows = []

    for q in questions:
        if not task_routing.is_as_of_question(q):
            continue

        prompt = q["prompt"].strip()
        response = llm.prompt(prompt).strip()
        result = score_question(q, harness["index"])
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
