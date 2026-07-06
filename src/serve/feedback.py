"""User feedback capture for detection results.

Stores each "was this correct?" response as a JSON line plus a copy of the
input image, so the pair (image, corrected label) can later be folded into
a retraining or calibration set. This module never touches a model's raw
confidence score - a single user's agreement isn't validation, it's a
labeled sample to accumulate for offline review.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = _PROJECT_ROOT / "data" / "feedback"
IMAGES_DIR = FEEDBACK_DIR / "images"
LOG_PATH = FEEDBACK_DIR / "feedback_log.jsonl"


def log_feedback(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    is_correct: bool,
    threshold: float,
    corrected_class: str | None = None,
    topk: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist one feedback record and its source image. Returns the record."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_id = uuid.uuid4().hex
    image_path = IMAGES_DIR / f"{image_id}.jpg"
    image.convert("RGB").save(image_path, format="JPEG", quality=90)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path.relative_to(_PROJECT_ROOT).as_posix(),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "threshold": threshold,
        "is_correct": is_correct,
        "corrected_class": corrected_class,
        "topk": topk or [],
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def read_feedback() -> list[dict[str, Any]]:
    """All logged feedback records, oldest first. Empty list if none yet."""
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def feedback_summary() -> dict[str, Any]:
    """Aggregate counts used by the analytics page - not a confidence signal,
    just a view into how much labeled feedback has accumulated so far."""
    records = read_feedback()
    correct = sum(1 for r in records if r["is_correct"])
    return {
        "total": len(records),
        "correct": correct,
        "incorrect": len(records) - correct,
    }


def out_of_taxonomy_requests(class_names: list[str]) -> list[tuple[str, int]]:
    """Corrected labels users typed via "Other" that aren't one of the known
    classes - i.e. cars the model can never get right until the dataset grows.
    Returns (label, count) pairs sorted most-requested first."""
    known = {c.lower() for c in class_names}
    counts: dict[str, int] = {}
    for r in read_feedback():
        corrected = r.get("corrected_class")
        if corrected and corrected.lower() not in known:
            counts[corrected] = counts.get(corrected, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
