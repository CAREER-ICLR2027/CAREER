"""Shared inference records. Boxes use original-image (x, y, width, height)."""

from dataclasses import dataclass
from typing import Any

from PIL import Image

Box = tuple[int, int, int, int]


class OutputError(ValueError):
    """A model returned an unusable decision or structured output."""


class CapacityError(ValueError):
    """The complete evidence cannot fit the receiving model."""


@dataclass(frozen=True)
class Completion:
    text: str
    first_token_logits: tuple[float, ...] | None
    token_count: int
    input_tokens: int
    visual_tokens: int


@dataclass(frozen=True)
class Localization:
    boxes: tuple[Box, ...]
    scores: tuple[float, ...] = ()


@dataclass
class Candidate:
    id: str
    box: Box
    sources: tuple[str, ...]
    children: tuple[str, ...] = ()
    score: float = 0.0
    visited: bool = False
    sam_prompt: str | None = None
    localization: Localization | None = None


@dataclass(frozen=True)
class View:
    id: str
    image: Image.Image
    box: Box
    source: str
    candidate_id: str | None = None
    action: str = "GLOBAL"
    sam_prompt: str | None = None
    sam_input_box: Box | None = None
    localization: Localization | None = None

    def metadata(self) -> dict[str, Any]:
        result = {
            "id": self.id, "box": list(self.box), "source": self.source,
            "candidate_id": self.candidate_id, "action": self.action,
            "scale": [self.image.width / self.box[2], self.image.height / self.box[3]],
            "sam_prompt": self.sam_prompt,
            "sam_input_box": self.sam_input_box,
        }
        result["localization"] = None if self.localization is None else {
            "boxes": self.localization.boxes, "scores": self.localization.scores,
        }
        return result


def enclose(boxes: tuple[Box, ...] | list[Box]) -> Box:
    if not boxes:
        raise ValueError("cannot enclose an empty set")
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    return (x, y, max(b[0] + b[2] for b in boxes) - x,
            max(b[1] + b[3] for b in boxes) - y)


def clip_box(box: Box, size: tuple[int, int]) -> Box | None:
    x, y, w, h = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(size[0], x + w), min(size[1], y + h)
    if right <= left or bottom <= top:
        return None
    return (left, top, right - left, bottom - top)
