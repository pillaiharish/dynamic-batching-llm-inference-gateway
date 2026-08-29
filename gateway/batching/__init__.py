"""Compatibility-aware process-local dynamic batching."""

from gateway.batching.dynamic import DynamicBatcher
from gateway.batching.eligibility import BatchEligibility, batching_eligibility
from gateway.batching.models import BatchItemResult

__all__ = [
    "BatchEligibility",
    "BatchItemResult",
    "DynamicBatcher",
    "batching_eligibility",
]
