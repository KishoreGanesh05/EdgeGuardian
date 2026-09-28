"""
Edge-Guardian Fault Scenario Module.

Defines parameterized fault scenarios that modify transformer operating
conditions in physically meaningful ways. Each scenario produces a
deterministic chain of consequences:

    Higher resistance → Higher I²R loss → Lower efficiency
    → Temperature increase → Feature deviation → Anomaly

Scenarios are applied to TransformerConfig and telemetry parameters
before physics calculations, ensuring the fault effects propagate
naturally through the physics engine rather than being faked.

Scenarios:
    NORMAL                       — No fault, baseline operation
    FEEDER_RESISTANCE_INCREASE   — Feeder cable degradation
    OVERLOAD                     — Load exceeds normal envelope
    THERMAL_STRESS               — Elevated ambient temperature
    EFFICIENCY_DEGRADATION       — Core/insulation losses increase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from data.models import FaultScenario, TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class ScenarioEffect:
    """
    The physical effects a fault scenario applies to a transformer's
    operating conditions for one telemetry sample.

    These effects are applied BEFORE physics calculations, so the
    physics engine naturally produces the correct downstream consequences.
    """
    # Multiplicative factor on feeder resistance (1.0 = no change)
    resistance_multiplier: float = 1.0
    # Additive load adjustment in percentage points
    load_adjustment_percent: float = 0.0
    # Additive ambient temperature adjustment (°C)
    ambient_temp_adjustment_c: float = 0.0
    # Multiplicative factor on efficiency (1.0 = no change, 0.95 = 5% reduction)
    efficiency_multiplier: float = 1.0
    # Metadata
    scenario_id: str = "NORMAL"
    severity: float = 0.0
    description: str = ""


# ── Scenario Definitions ──────────────────────────────────────────────────

def apply_scenario(
    scenario: FaultScenario,
    config: TransformerConfig,
    severity: Optional[float] = None,
    parameters: Optional[Dict[str, float]] = None,
) -> ScenarioEffect:
    """
    Calculate the physical effects of a fault scenario.

    This function does NOT mutate the TransformerConfig. It returns a
    ScenarioEffect that the telemetry generator uses to modify operating
    conditions before physics calculations.

    Args:
        scenario:   The fault scenario to apply.
        config:     The transformer configuration (read-only).
        severity:   Overall severity factor (0 to 1, where applicable).
        parameters: Scenario-specific parameters (overrides defaults).

    Returns:
        ScenarioEffect with the physical modifications to apply.
    """
    params = parameters or {}

    if scenario == FaultScenario.NORMAL:
        return _normal_scenario()

    elif scenario == FaultScenario.FEEDER_RESISTANCE_INCREASE:
        return _feeder_resistance_increase(config, severity, params)

    elif scenario == FaultScenario.OVERLOAD:
        return _overload(config, severity, params)

    elif scenario == FaultScenario.THERMAL_STRESS:
        return _thermal_stress(config, severity, params)

    elif scenario == FaultScenario.EFFICIENCY_DEGRADATION:
        return _efficiency_degradation(config, severity, params)

    else:
        logger.warning(f"Unknown scenario '{scenario}', treating as NORMAL")
        return _normal_scenario()


def _normal_scenario() -> ScenarioEffect:
    """No fault — all multipliers at baseline."""
    return ScenarioEffect(
        scenario_id="NORMAL",
        description="Normal operating conditions",
    )


def _feeder_resistance_increase(
    config: TransformerConfig,
    severity: Optional[float],
    params: Dict[str, float],
) -> ScenarioEffect:
    """
    Feeder cable resistance increase.

    Physical cause: aging, corrosion, loose connections, partial strand break.

    R_fault = R_base × (1 + increase_percent / 100)

    Chain of consequences:
        Higher R → Higher I²R loss → Lower efficiency → Temperature rise
        → Feature vector deviation → Anomaly detection
    """
    increase_pct = params.get("resistance_increase_percent", 25.0)

    # Severity scales the increase (0.0 = no effect, 1.0 = full effect)
    if severity is not None:
        increase_pct *= severity

    resistance_multiplier = 1.0 + (increase_pct / 100.0)

    logger.info(
        f"FEEDER_RESISTANCE_INCREASE: R × {resistance_multiplier:.3f} "
        f"({increase_pct:.1f}% increase)"
    )

    return ScenarioEffect(
        resistance_multiplier=resistance_multiplier,
        scenario_id="FEEDER_RESISTANCE_INCREASE",
        severity=increase_pct,
        description=f"Feeder resistance +{increase_pct:.1f}%",
    )


def _overload(
    config: TransformerConfig,
    severity: Optional[float],
    params: Dict[str, float],
) -> ScenarioEffect:
    """
    Overload scenario.

    Physical cause: demand exceeds design envelope, loss of parallel path,
    load redistribution after network reconfiguration.

    Chain of consequences:
        Higher load → Higher current → Higher I²R loss → Temperature rise
        → Insulation stress → Health degradation
    """
    overload_pct = params.get("overload_percent", 130.0)

    if severity is not None:
        # Severity scales from normal (100%) toward overload target
        overload_pct = 100.0 + (overload_pct - 100.0) * severity

    # Load adjustment: how much above 100% the load should be pushed
    # The generator adds this to the natural load profile
    load_adjustment = overload_pct - 100.0

    logger.info(f"OVERLOAD: load target {overload_pct:.1f}% (+{load_adjustment:.1f}pp)")

    return ScenarioEffect(
        load_adjustment_percent=load_adjustment,
        scenario_id="OVERLOAD",
        severity=overload_pct,
        description=f"Overload to {overload_pct:.1f}%",
    )


def _thermal_stress(
    config: TransformerConfig,
    severity: Optional[float],
    params: Dict[str, float],
) -> ScenarioEffect:
    """
    Thermal stress scenario.

    Physical cause: elevated ambient temperature (heatwave, ventilation
    failure, enclosure overheating), sustained high loading.

    Chain of consequences:
        Higher ambient → Higher winding temp → Higher thermal stress
        → Accelerated insulation aging → Health degradation
    """
    ambient_increase = params.get("ambient_increase_c", 15.0)

    if severity is not None:
        ambient_increase *= severity

    logger.info(
        f"THERMAL_STRESS: ambient +{ambient_increase:.1f}°C "
        f"(from {config.ambient_temperature_c:.1f}°C to "
        f"{config.ambient_temperature_c + ambient_increase:.1f}°C)"
    )

    return ScenarioEffect(
        ambient_temp_adjustment_c=ambient_increase,
        scenario_id="THERMAL_STRESS",
        severity=ambient_increase,
        description=f"Ambient temperature +{ambient_increase:.1f}°C",
    )


def _efficiency_degradation(
    config: TransformerConfig,
    severity: Optional[float],
    params: Dict[str, float],
) -> ScenarioEffect:
    """
    Efficiency degradation scenario.

    Physical cause: insulation degradation, increased core losses (hysteresis,
    eddy currents), partial discharge, oil degradation.

    Chain of consequences:
        Higher core/insulation losses → Lower efficiency
        → Negative eff_delta → Feature deviation → Anomaly
    """
    reduction_pct = params.get("efficiency_reduction_percent", 5.0)

    if severity is not None:
        reduction_pct *= severity

    efficiency_multiplier = 1.0 - (reduction_pct / 100.0)
    efficiency_multiplier = max(0.5, efficiency_multiplier)  # Floor at 50%

    logger.info(
        f"EFFICIENCY_DEGRADATION: efficiency × {efficiency_multiplier:.3f} "
        f"({reduction_pct:.1f}% reduction)"
    )

    return ScenarioEffect(
        efficiency_multiplier=efficiency_multiplier,
        scenario_id="EFFICIENCY_DEGRADATION",
        severity=reduction_pct,
        description=f"Efficiency reduction {reduction_pct:.1f}%",
    )


# ── Utility ────────────────────────────────────────────────────────────────

def get_available_scenarios() -> list[str]:
    """Return list of available fault scenario names."""
    return [s.value for s in FaultScenario]


def get_scenario_description(scenario: FaultScenario) -> str:
    """Return a human-readable description of a scenario."""
    descriptions = {
        FaultScenario.NORMAL: "Normal operating conditions — no fault injected.",
        FaultScenario.FEEDER_RESISTANCE_INCREASE: (
            "Feeder cable resistance increase due to aging, corrosion, "
            "or loose connections. Causes higher I²R losses."
        ),
        FaultScenario.OVERLOAD: (
            "Load exceeds normal operating envelope. Causes higher current, "
            "losses, and temperature."
        ),
        FaultScenario.THERMAL_STRESS: (
            "Elevated ambient temperature or sustained high loading. "
            "Accelerates insulation aging."
        ),
        FaultScenario.EFFICIENCY_DEGRADATION: (
            "Gradual efficiency loss due to insulation or core degradation. "
            "Increases losses and heat generation."
        ),
    }
    return descriptions.get(scenario, "Unknown scenario.")
