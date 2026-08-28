"""Credential-free backend routing state models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendSlotSnapshot:
    """Safe state for one configured backend."""

    healthy: bool
    inflight: int


@dataclass(frozen=True, slots=True)
class BackendPoolSnapshot:
    """Safe point-in-time state for the process-local routing pool."""

    closed: bool
    backends: dict[str, BackendSlotSnapshot]
