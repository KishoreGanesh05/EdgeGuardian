"""
Edge-Guardian AWS IoT SiteWise Sink (P5) — OPTIONAL.

Optional adapter that forwards physics-derived asset properties to AWS IoT
SiteWise. Like `AwsIoTSink`, boto3 is imported lazily and the core demo
never constructs this sink.

Stage 2 scope:
    Conceptual mapping of telemetry to SiteWise asset property values via
    BatchPutAssetPropertyValue. A real deployment would map Edge-Guardian
    assets to SiteWise asset/property IDs through a model registry.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from data.models import TelemetrySample
from cloud.base import TelemetrySink
from cloud.aws_iot import boto3_available

logger = logging.getLogger(__name__)


class SiteWiseSink(TelemetrySink):
    """
    Forwards telemetry to AWS IoT SiteWise.

    `property_map` maps Edge-Guardian asset IDs to a SiteWise
    (asset_id, property_id) pair. Assets not present in the map are skipped
    with a warning rather than raising, so partial fleets degrade gracefully.
    """

    def __init__(
        self,
        property_map: Optional[Dict[str, Dict[str, str]]] = None,
        region: Optional[str] = None,
        client: Optional[object] = None,
    ) -> None:
        self._property_map = property_map or {}
        self._region = region
        self._client = client
        self._total_sent = 0

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not boto3_available():
            raise RuntimeError(
                "SiteWiseSink requires the optional 'boto3' dependency. "
                "Install it with 'pip install boto3' and configure AWS "
                "credentials. The core Edge-Guardian pipeline runs without it."
            )
        import boto3

        kwargs = {}
        if self._region:
            kwargs["region_name"] = self._region
        self._client = boto3.client("iotsitewise", **kwargs)
        logger.info("SiteWiseSink connected (region=%s)", self._region)
        return self._client

    def send(self, sample: TelemetrySample) -> None:
        mapping = self._property_map.get(sample.asset_id)
        if not mapping:
            logger.warning(
                "SiteWiseSink has no property mapping for asset %s — skipping",
                sample.asset_id,
            )
            return

        client = self._ensure_client()
        timestamp_s = int(sample.timestamp.timestamp())
        client.batch_put_asset_property_value(
            entries=[
                {
                    "entryId": f"{sample.asset_id}-{timestamp_s}",
                    "assetId": mapping["asset_id"],
                    "propertyId": mapping["property_id"],
                    "propertyValues": [
                        {
                            "value": {"doubleValue": float(sample.load_percent)},
                            "timestamp": {"timeInSeconds": timestamp_s},
                            "quality": "GOOD",
                        }
                    ],
                }
            ]
        )
        self._total_sent += 1
        logger.debug("SiteWiseSink forwarded sample for %s", sample.asset_id)

    @property
    def total_sent(self) -> int:
        return self._total_sent
