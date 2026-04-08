"""
TemporalBench — Shared data format utilities
Handles format differences across v1/v2/v3/v4 seed directories and top-level files.
"""

import os
import re


def parse_content_v1(content: str):
    """v1: 'domain:subject:d{day}:{value}' e.g. 'slow:hardware_0:d1:v1'"""
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    day = int(parts[2].replace("d", ""))
    value = parts[3] if len(parts) > 3 else parts[-1]
    return domain, subject, day, value


def parse_content_v2(content: str):
    """v2: 'domain:subject:day{day}:v{version}:{extra}' e.g. 'slow:hardware_0:day1:v1:7311'"""
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    m = re.search(r"day(\d+)", content)
    day = int(m.group(1)) if m else 0
    value = parts[-1]
    return domain, subject, day, value


def parse_content_v3(content: str):
    """v3: 'domain:subject:d{day}:v{version}:{extra}' e.g. 'slow:hw_0:d1:v1:5242'"""
    parts = content.split(":")
    domain = parts[0]
    subject = parts[1]
    m = re.search(r"d(\d+)", content)
    day = int(m.group(1)) if m else 0
    value = parts[-1]
    return domain, subject, day, value


def parse_content_v4(content: str):
    """v4: 'domain:subject:d{day}:v{version}' e.g. 'fast:sts0:d1:v1'"""
    return parse_content_v3(content)  # same pattern


def detect_version(content: str):
    """Detect format from content string."""
    if "day" in content and "d" not in content.split(":")[2]:
        return "v2"
    elif "d" in content:
        return "v3v4"
    return "v1"


def parse_content_flexible(content: str):
    """Auto-detect format and parse accordingly."""
    if "day" in content and "d" not in [p for p in content.split(":") if p]:
        return parse_content_v2(content)
    elif ":d" in content or content.count(":") >= 3:
        return parse_content_v3(content)
    else:
        return parse_content_v1(content)


def get_data_paths(version: str, seed: int, data_dir: str):
    """
    Return (q_path, f_path, e_path) for a given version/seed.
    Handles seed-specific dirs vs top-level prefixed files.
    """
    seed_dir = os.path.join(data_dir, f"{version}_seed{seed}")

    if os.path.isdir(seed_dir):
        return (
            os.path.join(seed_dir, "questions.jsonl"),
            os.path.join(seed_dir, "facts.jsonl"),
            os.path.join(seed_dir, "events.jsonl"),
        )
    else:
        return (
            os.path.join(data_dir, f"temporalbench_{version}_questions.jsonl"),
            os.path.join(data_dir, f"temporalbench_{version}_facts.jsonl"),
            os.path.join(data_dir, f"temporalbench_{version}_events.jsonl"),
        )


def get_question_version(version: str) -> str:
    """Map benchmark version to question task family set."""
    return version  # used for routing