"""
TemporalBench — Temporal store with validity windows.
Builds a queryable store from facts + events data.
"""

import json
import os
from typing import Optional


def get_data_dir():
    """Find the temporalbench data directory (Kaggle input or local)."""
    kaggle_input = "/kaggle/input/temporalbench"
    if os.path.isdir(kaggle_input):
        return kaggle_input
    # Local development path
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "time", "benchmarks")


def get_data_paths(version: str, seed: int, data_dir: str):
    """Return (questions_path, facts_path, events_path) for version/seed."""
    seed_dir = os.path.join(data_dir, f"{version}_seed{seed}")
    if os.path.isdir(seed_dir):
        return (
            os.path.join(seed_dir, "questions.jsonl"),
            os.path.join(seed_dir, "facts.jsonl"),
            os.path.join(seed_dir, "events.jsonl"),
        )
    return (
        os.path.join(data_dir, f"temporalbench_{version}_questions.jsonl"),
        os.path.join(data_dir, f"temporalbench_{version}_facts.jsonl"),
        os.path.join(data_dir, f"temporalbench_{version}_events.jsonl"),
    )


class TemporalStore:
    """
    Queryable temporal store with validity windows.
    Supports:
      - get(domain, subject, as_of_day): what was true at day X
      - detect_change(domain, subject, start_day, end_day): did it change
      - latest_change_event(domain, subject, before_day): most recent event before day X
    """

    def __init__(self):
        # (domain, subject) -> list of (day, content) sorted ascending
        self.timeline: dict[tuple, list[tuple[int, str]]] = {}
        # (domain, subject) -> list of event dicts sorted by t_event
        self.events: dict[tuple, list[dict]] = {}

    def add_fact(self, domain: str, subject: str, day: int, content: str):
        key = (domain, subject)
        if key not in self.timeline:
            self.timeline[key] = []
        self.timeline[key].append((day, content))

    def add_event(self, domain: str, subject: str, event: dict):
        key = (domain, subject)
        if key not in self.events:
            self.events[key] = []
        self.events[key].append(event)

    def sort_timelines(self):
        for key in self.timeline:
            self.timeline[key].sort(key=lambda x: x[0])

    def get(self, domain: str, subject: str, as_of_day: int) -> Optional[str]:
        """Return the most recent fact content for (domain, subject) as of day X."""
        key = (domain, subject)
        if key not in self.timeline:
            return None
        candidates = [(d, c) for d, c in self.timeline[key] if d <= as_of_day]
        if not candidates:
            return None
        # Return the most recent
        return max(candidates, key=lambda x: x[0])[1]

    def latest_change_event(
        self, domain: str, subject: str, before_day: int
    ) -> Optional[dict]:
        """Return the most recent FACT_OBSERVED event for (domain, subject) before day X."""
        key = (domain, subject)
        if key not in self.events:
            return None
        candidates = [e for e in self.events[key] if e.get("t_event", 0) < before_day]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.get("t_event", 0))

    def detect_change(
        self, domain: str, subject: str, start_day: int, end_day: int
    ) -> bool:
        """Return True if domain:subject changed between start and end day (exclusive/inclusive)."""
        key = (domain, subject)
        if key not in self.timeline:
            return False
        values = [
            (d, c)
            for d, c in self.timeline[key]
            if start_day < d <= end_day
        ]
        return len(values) > 1


def build_store(version: str, seed: int, data_dir: str = None) -> TemporalStore:
    """Build a TemporalStore from facts and events data files."""
    if data_dir is None:
        data_dir = get_data_dir()
    q_path, f_path, e_path = get_data_paths(version, seed, data_dir)

    store = TemporalStore()

    # Load facts
    for line in open(f_path, encoding="utf-8"):
        f = json.loads(line)
        domain = f.get("domain", "")
        content = f.get("content", "")

        # Parse domain:subject:d{day}:{value}
        parts = content.split(":")
        if len(parts) >= 3:
            subj = parts[1]
            m_day = None
            for p in parts[2:]:
                if p.startswith("d") and p[1:].isdigit():
                    m_day = int(p[1:])
                    break
                if "day" in p and any(c.isdigit() for c in p):
                    import re
                    m = re.search(r"day(\d+)", p)
                    if m:
                        m_day = int(m.group(1))
                        break
            if m_day is not None:
                store.add_fact(domain, subj, m_day, content)

    # Load events (for causal trace)
    if os.path.exists(e_path):
        for line in open(e_path, encoding="utf-8"):
            e = json.loads(line)
            domain = e.get("domain", "")
            subject = e.get("subject", "")
            store.add_event(domain, subject, e)

    store.sort_timelines()
    return store


def load_questions(version: str, seed: int, data_dir: str = None) -> list:
    """Load question list for a given version/seed."""
    if data_dir is None:
        data_dir = get_data_dir()
    q_path, _, _ = get_data_paths(version, seed, data_dir)
    return [json.loads(l) for l in open(q_path, encoding="utf-8")]
