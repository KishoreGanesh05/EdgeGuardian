"""
Edge-Guardian AWS IoT Core Sink (P5) — OPTIONAL.

This is an optional adapter. Importing this module never requires boto3;
`boto3` is imported lazily only when a connection is actually established.
The core demo (`python main.py --demo`) never constructs this sink.

Stage 2 scope:
    This publishes telemetry JSON to an MQTT topic on AWS IoT Core. It is a
    thin, conceptual adapter demonstrating the cloud boundary — not a
    production-hardened IoT client.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from data.models import TelemetrySample
from cloud.base import TelemetrySink

logger = logging.getLogger(__name__)


def boto3_available() -> bool:
    """Return True if boto3 can be imported in this environment."""
    try:
        import boto3  # noqa: F401
        return True
    except ImportError:
        return False


class AwsIoTSink(TelemetrySink):
    """
    Publishes telemetry to AWS IoT Core over the data-plane HTTPS API.

    Credentials are resolved by boto3's standard provider chain (env vars,
    shared config, instance profile) — Edge-Guardian never handles raw
    secrets. The client is created lazily on first `send`.
    """

    def __init__(
        self,
        topic: str = "edge-guardian/telemetry",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        client: Optional[object] = None,
    ) -> None:
        self._topic = topic
        self._region = region
        self._endpoint_url = endpoint_url
        self._client = client  # allows dependency injection for testing
        self._total_sent = 0

    def _ensure_client(self):
        """Lazily create the boto3 iot-data client."""
        if self._client is not None:
            return self._client
        if not boto3_available():
            raise RuntimeError(
                "AwsIoTSink requires the optional 'boto3' dependency. "
                "Install it with 'pip install boto3' and configure AWS "
                "credentials. The core Edge-Guardian pipeline runs without it."
            )
        import boto3

        kwargs = {}
        if self._region:
            kwargs["region_name"] = self._region
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        self._client = boto3.client("iot-data", **kwargs)
        logger.info("AwsIoTSink connected (topic=%s, region=%s)", self._topic, self._region)
        return self._client

    def send(self, sample: TelemetrySample) -> None:
        client = self._ensure_client()
        payload = sample.model_dump(mode="json")
        client.publish(
            topic=self._topic,
            qos=1,
            payload=json.dumps(payload),
        )
        self._total_sent += 1
        logger.debug("AwsIoTSink published sample for %s", sample.asset_id)

    @property
    def total_sent(self) -> int:
        return self._total_sent
