"""
Edge-Guardian Health Calculator (P3).

Calculates Health Index (0-100) and Remaining Useful Life (RUL)
based on deterministic rules applied to upstream telemetry and AI results.
"""

from data.models import HealthResult, HealthState, RULResult, PhysicsResult, AIResult, TelemetrySample
from config.settings import (
    HEALTH_WARNING_THRESHOLD,
    HEALTH_CRITICAL_THRESHOLD,
    HEALTH_WEIGHTS,
    RUL_CRITICAL_THRESHOLD,
    RUL_BASE_LIFETIME_HOURS
)

class HealthCalculator:
    """
    Calculates deterministic health and RUL metrics.

    Note: The Health Index and RUL are synthetic, model-based metrics
    intended for Stage 2 demonstration, not validated real-world metrics.
    """

    def calculate_health(
        self,
        telemetry: TelemetrySample,
        physics: PhysicsResult,
        ai: AIResult,
        previous_degradation: float = 0.0
    ) -> HealthResult:
        """
        Calculate health index from 0 to 100 based on upstream data.
        """
        # 1. Anomaly severity (0 to 1)
        # Cap AI anomaly score for scaling. Assume max severity if flagged as anomaly.
        anomaly_penalty = 1.0 if ai.is_anomaly else min(ai.anomaly_score / 10.0, 1.0)

        # 2. Thermal stress (0 to 1)
        # Stress > 1.0 means operating above rated temperature.
        thermal_penalty = min(max(physics.thermal_stress - 1.0, 0.0), 1.0)

        # 3. Overload exposure (0 to 1)
        # Load > 100% contributes to overload penalty, capping at 150%.
        overload_penalty = min(max((telemetry.load_percent - 100.0) / 50.0, 0.0), 1.0)

        # 4. Efficiency degradation (0 to 1)
        # Negative eff_delta means degradation. Max penalty at -5% efficiency.
        eff_penalty = min(max(-physics.eff_delta / 0.05, 0.0), 1.0)

        # 5. Historical degradation (0 to 1)
        history_penalty = previous_degradation

        # Calculate weighted sum of penalties
        total_penalty = (
            anomaly_penalty * HEALTH_WEIGHTS.get("anomaly_severity", 0.3) +
            thermal_penalty * HEALTH_WEIGHTS.get("thermal_stress", 0.2) +
            overload_penalty * HEALTH_WEIGHTS.get("overload_exposure", 0.15) +
            eff_penalty * HEALTH_WEIGHTS.get("efficiency_degradation", 0.2) +
            history_penalty * HEALTH_WEIGHTS.get("degradation_history", 0.15)
        )

        total_penalty = min(max(total_penalty, 0.0), 1.0)

        # Health Index is inversely proportional to total penalty
        health_index = 100.0 * (1.0 - total_penalty)

        # Determine categorical health state
        if health_index < HEALTH_CRITICAL_THRESHOLD:
            state = HealthState.CRITICAL
        elif health_index < HEALTH_WARNING_THRESHOLD:
            state = HealthState.WARNING
        else:
            state = HealthState.NORMAL

        return HealthResult(
            health_index=health_index,
            health_state=state,
            degradation_score=total_penalty
        )

    def estimate_rul(self, health: HealthResult) -> RULResult:
        """
        Estimate model-based Remaining Useful Life (RUL) in hours.
        """
        if health.health_index <= RUL_CRITICAL_THRESHOLD:
            rul_hours = 0.0
        else:
            # Scale remaining life between critical threshold and 100.
            health_ratio = (health.health_index - RUL_CRITICAL_THRESHOLD) / (100.0 - RUL_CRITICAL_THRESHOLD)
            rul_hours = RUL_BASE_LIFETIME_HOURS * health_ratio

        return RULResult(
            rul_hours=rul_hours,
            critical_threshold=RUL_CRITICAL_THRESHOLD,
            current_degradation=health.degradation_score,
            model_based=True
        )

