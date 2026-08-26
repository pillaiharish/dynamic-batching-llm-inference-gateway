"""Inference backend contracts and implementations."""

from gateway.backends.base import InferenceBackend
from gateway.backends.fake import FakeBackend

__all__ = ["FakeBackend", "InferenceBackend"]
