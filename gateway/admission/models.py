"""Safe admission-control diagnostics without credential material."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantAdmissionSnapshot:
    inflight: int
    queued: int


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    global_inflight: int
    global_queued: int
    tenants: dict[str, TenantAdmissionSnapshot]
    closed: bool
