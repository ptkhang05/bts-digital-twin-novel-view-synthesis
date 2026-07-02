from __future__ import annotations

from dataclasses import dataclass

from bts_nvs.exceptions import DataValidationError


CONTEST_SLUG = "var-2026"
CONTEST_NAME = "Bai 1 - BTS Digital Twin (Novel View Synthesis)"
CONTEST_URL = "https://competition.viettel.vn/contests/var-2026"
SUBMISSION_TYPE = "FILE_ZIP"
WORKER_TYPE = "GPU"

DEFAULT_CONTEST_PHASE = "phase1"


@dataclass(frozen=True)
class ContestPhase:
    order: int
    name: str
    starts_at_vn: str
    ends_at_vn: str
    submission_type: str = SUBMISSION_TYPE
    worker_type: str = WORKER_TYPE


@dataclass(frozen=True)
class ContestRules:
    key: str
    label: str
    train_image_min: int
    train_image_max: int
    target_view_min: int
    target_view_max: int


PHASES = (
    ContestPhase(1, "Vong 1 - So loai", "2026-07-02 00:00:00", "2026-07-30 23:59:59"),
    ContestPhase(2, "Vong 2 - So khao", "2026-08-17 00:00:00", "2026-08-19 23:59:59"),
    ContestPhase(3, "Vong 3 - Chung ket", "2026-09-09 00:00:00", "2026-09-10 23:59:59"),
)

RULES = {
    "overview": ContestRules("overview", "general problem statement", 100, 300, 20, 50),
    "phase1": ContestRules("phase1", "round 1", 150, 300, 40, 70),
}
RULE_ALIASES = {
    "1": "phase1",
    "round1": "phase1",
    "vong1": "phase1",
    "general": "overview",
    "problem": "overview",
}


def get_contest_rules(phase: str | int | None = DEFAULT_CONTEST_PHASE) -> ContestRules:
    key = str(phase if phase is not None else DEFAULT_CONTEST_PHASE).lower().replace("-", "").replace("_", "")
    normalized = RULE_ALIASES.get(key, key)
    if normalized in RULES:
        return RULES[normalized]
    expected = ", ".join(sorted(RULES))
    raise DataValidationError(f"Unknown contest phase '{phase}'. Expected one of: {expected}")


def validate_training_image_count(count: int, phase: str | int | None = DEFAULT_CONTEST_PHASE) -> None:
    rules = get_contest_rules(phase)
    _validate_range("training images per scene", count, rules.train_image_min, rules.train_image_max, rules)


def validate_target_view_count(count: int, phase: str | int | None = DEFAULT_CONTEST_PHASE) -> None:
    rules = get_contest_rules(phase)
    _validate_range("target views per scene", count, rules.target_view_min, rules.target_view_max, rules)


def _validate_range(label: str, count: int, minimum: int, maximum: int, rules: ContestRules) -> None:
    if count < minimum or count > maximum:
        raise DataValidationError(f"Contest {rules.key} expects {minimum}-{maximum} {label}; got {count}")
