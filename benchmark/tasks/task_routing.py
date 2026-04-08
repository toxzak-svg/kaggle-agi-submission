"""
TemporalBench — Version-aware task family mapping.
Maps raw task_family strings (v1,v2,v3,v4) to benchmark task types.
"""

# Map raw task_family -> benchmark task type
TASK_TYPE_MAP = {
    # v1 names
    "AsOfQA": "AsOfQA",
    "ChangeDetection": "ChangeDetection",
    "CausalQuery": "CausalQuery",
    # v2/v3/v4 names
    "PastQueryTrap": "AsOfQA",        # "As of day X, what was true about Y?"
    "CurrentQuery": "CurrentState",   # "What is true about Y now?"
    "DecayTrap": "DecayTrap",         # queries designed to fool decay-based systems
    "OverlapTrap": "OverlapTrap",     # overlapping time intervals
}

# Which task types to run for each version
VERSION_TASK_TYPES = {
    "v1": ["AsOfQA", "ChangeDetection", "CausalQuery"],
    "v2": ["AsOfQA", "ChangeDetection", "CausalQuery", "CurrentState"],
    "v3": ["AsOfQA", "ChangeDetection", "CausalQuery", "CurrentState", "DecayTrap", "OverlapTrap"],
    "v4": ["CurrentState", "OverlapTrap", "AsOfQA"],
}


def get_task_type(task_family: str) -> str:
    """Map raw task_family to benchmark type."""
    return TASK_TYPE_MAP.get(task_family, task_family)


def get_version_tasks(version: str) -> list:
    """Return list of task types to run for a given version."""
    return VERSION_TASK_TYPES.get(version, VERSION_TASK_TYPES["v1"])


def is_as_of_question(question: dict) -> bool:
    """Check if question is an AsOf query (any version)."""
    tf = question.get("task_family", "")
    return tf in ("AsOfQA", "PastQueryTrap") or "as of day" in question.get("prompt", "").lower()


def is_current_query(question: dict) -> bool:
    """Check if question asks about current state."""
    tf = question.get("task_family", "")
    return tf == "CurrentQuery" or question.get("as_of_day", 0) >= 999


def is_change_detection(question: dict) -> bool:
    """Check if question asks about change detection."""
    return question.get("task_family") == "ChangeDetection"


def is_causal_query(question: dict) -> bool:
    """Check if question is a causal/counterfactual query."""
    return question.get("task_family") == "CausalQuery"