"""
Edge-Guardian Digital Shadow (P3).

Represents the complete, structured state of a single transformer asset,
including operational metrics, health, AI diagnostics, and history.
"""

from data.models import (
    DigitalShadowState,
    TransformerConfig,
    TelemetrySample,
    PhysicsResult,
    AIResult,
    HealthResult,
    RULResult
)
from config.settings import HISTORY_MAX_LENGTH

class DigitalShadow:
    """
    State manager for a single virtual transformer asset.
    """

    def __init__(self, config: TransformerConfig):
        self._state = DigitalShadowState(
            asset_id=config.asset_id,
            config=config
        )
        self.rul_history = []

    @property
    def state(self) -> DigitalShadowState:
        """Get the current authoritative state of the asset."""
        return self._state

    def update(
        self,
        telemetry: TelemetrySample,
        physics: PhysicsResult,
        ai: AIResult,
        health: HealthResult,
        rul: RULResult
    ) -> None:
        """
        Update the Digital Shadow with the latest simulation step data.
        """
        s = self._state

        # Update timestamps and operational state
        s.timestamp = telemetry.timestamp
        s.load_percent = telemetry.load_percent
        s.temperature = telemetry.transformer_temperature_c
        
        # Physics state
        s.rms_v = physics.rms_v
        s.rms_i = physics.rms_i
        s.power_w = physics.power_w
        s.efficiency = physics.efficiency
        s.power_factor = physics.power_factor
        s.loss_w = physics.loss_w
        s.eff_delta = physics.eff_delta
        s.thermal_stress = physics.thermal_stress

        # AI and diagnostic state
        s.anomaly_score = ai.anomaly_score
        s.is_anomaly = ai.is_anomaly
        s.fault_class = ai.fault_class
        s.fault_confidence = ai.fault_confidence

        # Health and RUL state
        s.health_index = health.health_index
        s.health_state = health.health_state
        s.degradation_score = health.degradation_score
        s.rul_hours = rul.rul_hours

        # Update bounded history
        self._append_history(s.health_history, health.health_index)
        self._append_history(s.anomaly_history, ai.anomaly_score)
        self._append_history(s.efficiency_history, physics.efficiency)
        self._append_history(s.temperature_history, telemetry.transformer_temperature_c)
        self._append_history(self.rul_history, rul.rul_hours)

    def _append_history(self, history_list: list, value: float) -> None:
        """Append a value to a history list, maintaining the max length bound."""
        history_list.append(value)
        if len(history_list) > HISTORY_MAX_LENGTH:
            history_list.pop(0)

