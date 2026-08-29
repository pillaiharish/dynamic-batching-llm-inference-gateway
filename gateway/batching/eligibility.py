"""Explicit bounded dynamic-batching eligibility decisions."""

from dataclasses import dataclass
from typing import Literal

from gateway.schemas.chat import ChatCompletionRequest

BatchDecision = Literal["eligible", "bypass"]
BatchEligibilityReason = Literal["eligible", "disabled", "streaming", "n_gt_1"]


@dataclass(frozen=True, slots=True)
class BatchEligibility:
    """A metrics-safe eligibility decision."""

    decision: BatchDecision
    reason: BatchEligibilityReason

    @property
    def eligible(self) -> bool:
        return self.decision == "eligible"


def batching_eligibility(
    request: ChatCompletionRequest,
    *,
    enabled: bool,
) -> BatchEligibility:
    """Decide whether a validated request may use vLLM's batch endpoint."""
    if not enabled:
        return BatchEligibility("bypass", "disabled")
    if request.stream:
        return BatchEligibility("bypass", "streaming")
    if request.n > 1:
        return BatchEligibility("bypass", "n_gt_1")
    return BatchEligibility("eligible", "eligible")
