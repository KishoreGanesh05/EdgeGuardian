"""
Edge-Guardian optional cloud integration layer (P5).

The core local pipeline never depends on anything in this package.
`LocalSink` (the default) requires no credentials or network access.
"""

from cloud.base import TelemetrySink, LocalSink, get_default_sink

__all__ = ["TelemetrySink", "LocalSink", "get_default_sink"]
