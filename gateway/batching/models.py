"""Public dynamic-batching result models."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """The ordinary Chat Completion response for one logical batch member."""

    response: dict[str, Any]
