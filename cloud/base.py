"""
Edge-Guardian Cloud Telemetry Sink Interface (P5).

Defines the optional cloud-integration boundary. The core Edge-Guardian
pipeline MUST run without any of these adapters — they are strictly
optional egress points for telemetry.

Architecture rule (plan §32):
    The local Edge AI pipeline must never depend on AWS or any cloud sink.
    `LocalSink` is the default and requires no credentials or network.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Iterable, List

from data.models import TelemetrySample

logger = logging.getLogger(__name__)


class TelemetrySink(ABC):
    """
    Abstract egress point for telemetry samples.

    Implementations may forward samples to local memory, AWS IoT Core,
    AWS SiteWise, or any future destination. The pipeline interacts only
    with this interface, never with a concrete cloud SDK.
    """

    @property
    def name(self) -> str:
        """Human-readable sink name."""
        return self.__class__.__name__

    @abstractmethod
    def send(self, sample: TelemetrySample) -> None:
        """Forward a single telemetry sample to the destination."""
        raise NotImplementedError

    def send_batch(self, samples: Iterable[TelemetrySample]) -> int:
        """
        Forward a batch of telemetry samples.

        Returns the number of samples sent. The default implementation
        simply calls `send` per sample; adapters may override for
        batched transport.
        """
        count = 0
        for sample in samples:
            self.send(sample)
            count += 1
        return count

    def flush(self) -> None:
        """Flush any buffered samples. No-op by default."""
        return None

    def close(self) -> None:
        """Flush and release any resources. Safe to call multiple times."""
        self.flush()


class LocalSink(TelemetrySink):
    """
    Default, dependency-free sink.

    Keeps a bounded in-memory buffer of the most recent samples and logs
    at DEBUG level. This is the sink used by `python main.py --demo`,
    guaranteeing the demo runs with no AWS credentials or network access.
    """

    def __init__(self, max_buffer: int = 1000) -> None:
        self._buffer: Deque[TelemetrySample] = deque(maxlen=max_buffer)
        self._total_sent = 0

    def send(self, sample: TelemetrySample) -> None:
        self._buffer.append(sample)
        self._total_sent += 1
        logger.debug(
            "LocalSink received sample for %s (load=%.1f%%, T=%.1f°C)",
            sample.asset_id,
            sample.load_percent,
            sample.transformer_temperature_c,
        )

    @property
    def total_sent(self) -> int:
        """Total number of samples ever sent to this sink."""
        return self._total_sent

    @property
    def buffer(self) -> List[TelemetrySample]:
        """A snapshot copy of the current bounded buffer."""
        return list(self._buffer)

    def latest(self) -> TelemetrySample | None:
        """Return the most recently sent sample, or None if empty."""
        return self._buffer[-1] if self._buffer else None


def get_default_sink() -> TelemetrySink:
    """
    Return the default telemetry sink.

    Always returns a `LocalSink`; cloud sinks must be constructed
    explicitly and never become a default dependency.
    """
    return LocalSink()
