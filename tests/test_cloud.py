"""
Tests for the optional cloud integration layer (P5).

These tests require no TensorFlow and no AWS — they verify that the core
never depends on cloud SDKs and that cloud sinks fail loudly (not silently)
when their optional dependency is missing.
"""

import pytest
from datetime import datetime, timezone

from data.models import TelemetrySample
from cloud.base import TelemetrySink, LocalSink, get_default_sink
from cloud.aws_iot import AwsIoTSink, boto3_available
from cloud.sitewise import SiteWiseSink


def _sample(asset_id="T001"):
    return TelemetrySample(
        timestamp=datetime.now(timezone.utc),
        asset_id=asset_id,
        voltage_feeder_v=11000.0,
        current_feeder_a=20.0,
        voltage_secondary_v=415.0,
        current_secondary_a=500.0,
        frequency_hz=50.0,
        ambient_temperature_c=35.0,
        transformer_temperature_c=60.0,
        load_percent=60.0,
    )


class TestLocalSink:
    def test_default_sink_is_local(self):
        assert isinstance(get_default_sink(), LocalSink)

    def test_send_and_count(self):
        sink = LocalSink()
        for _ in range(5):
            sink.send(_sample())
        assert sink.total_sent == 5
        assert len(sink.buffer) == 5
        assert sink.latest().asset_id == "T001"

    def test_send_batch(self):
        sink = LocalSink()
        n = sink.send_batch([_sample(), _sample(), _sample()])
        assert n == 3
        assert sink.total_sent == 3

    def test_bounded_buffer(self):
        sink = LocalSink(max_buffer=10)
        for _ in range(25):
            sink.send(_sample())
        assert sink.total_sent == 25
        assert len(sink.buffer) == 10  # bounded

    def test_empty_latest(self):
        assert LocalSink().latest() is None

    def test_is_telemetry_sink(self):
        assert isinstance(LocalSink(), TelemetrySink)


class TestAwsSinksOptional:
    def test_import_does_not_require_boto3(self):
        # Constructing the sink must never import boto3.
        sink = AwsIoTSink(topic="x")
        assert sink.total_sent == 0

    @pytest.mark.skipif(boto3_available(), reason="boto3 installed; skip missing-dep test")
    def test_aws_iot_raises_without_boto3(self):
        sink = AwsIoTSink(topic="edge-guardian/telemetry")
        with pytest.raises(RuntimeError, match="boto3"):
            sink.send(_sample())

    @pytest.mark.skipif(boto3_available(), reason="boto3 installed; skip missing-dep test")
    def test_sitewise_raises_without_boto3_when_mapped(self):
        sink = SiteWiseSink(property_map={"T001": {"asset_id": "a", "property_id": "p"}})
        with pytest.raises(RuntimeError, match="boto3"):
            sink.send(_sample("T001"))

    def test_sitewise_skips_unmapped_asset(self):
        # Unmapped asset is skipped gracefully (no boto3 needed, no raise).
        sink = SiteWiseSink(property_map={})
        sink.send(_sample("T999"))
        assert sink.total_sent == 0

    def test_injected_client_is_used(self):
        # A dependency-injected fake client avoids boto3 entirely.
        class FakeClient:
            def __init__(self):
                self.published = []

            def publish(self, topic, qos, payload):
                self.published.append((topic, qos, payload))

        fake = FakeClient()
        sink = AwsIoTSink(topic="t", client=fake)
        sink.send(_sample("T001"))
        assert sink.total_sent == 1
        assert fake.published[0][0] == "t"
        assert "T001" in fake.published[0][2]
