from __future__ import annotations

from dataclasses import dataclass

from bts_nvs.exceptions import DataValidationError


CONTEST_SLUG = "var-2026"
CONTEST_NAME = "Bai 1 - BTS Digital Twin (Novel View Synthesis)"
CONTEST_URL = "https://competition.viettel.vn/contests/var-2026"
SUBMISSION_TYPE = "FILE_ZIP"
WORKER_TYPE = "GPU"

TRAIN_IMAGE_MIN = 100
TRAIN_IMAGE_MAX = 300
TARGET_VIEW_MIN = 20
TARGET_VIEW_MAX = 60


@dataclass(frozen=True)
class ContestPhase:
    order: int
    name: str
    starts_at_vn: str
    ends_at_vn: str
    submission_type: str = SUBMISSION_TYPE
    worker_type: str = WORKER_TYPE


PHASES = (
    ContestPhase(1, "Vong 1 - So loai", "2026-07-02 00:00:00", "2026-07-30 23:59:59"),
    ContestPhase(2, "Vong 2 - So khao", "2026-08-17 00:00:00", "2026-08-19 23:59:59"),
    ContestPhase(3, "Vong 3 - Chung ket", "2026-09-09 00:00:00", "2026-09-10 23:59:59"),
)


def validate_training_image_count(count: int) -> None:
    _validate_range("training images per scene", count, TRAIN_IMAGE_MIN, TRAIN_IMAGE_MAX)


def validate_target_view_count(count: int) -> None:
    _validate_range("target views per scene", count, TARGET_VIEW_MIN, TARGET_VIEW_MAX)


def _validate_range(label: str, count: int, minimum: int, maximum: int) -> None:
    if count < minimum or count > maximum:
        raise DataValidationError(f"Contest expects {minimum}-{maximum} {label}; got {count}")
