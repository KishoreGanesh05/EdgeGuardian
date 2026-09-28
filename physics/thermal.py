"""
Edge-Guardian Thermal Physics Module.

Implements a simplified deterministic thermal model for transformer
winding temperature estimation and thermal stress calculation.

The thermal model follows:
    T_winding = T_ambient + ΔT_load + ΔT_loss

where:
    ΔT_load is proportional to load² (I²R heating)
    ΔT_loss accounts for additional loss-induced heating

Thermal stress captures how far the transformer operates above
its normal thermal envelope. This feeds into the health index and
is part of the physics-derived context for Edge AI.

NOTE: This is a project simulation model for Stage 2 validation,
not a certified transformer thermal model (e.g. IEEE C57.91).
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)

# Default constants — should match config/settings.py
_DEFAULT_THERMAL_RISE_COEFF = 40.0   # °C rise at full load above ambient
_DEFAULT_RATED_TEMP_RISE = 65.0      # Rated winding temperature rise (°C)


def calculate_transformer_temperature(
    ambient_temperature_c: float,
    load_fraction: float,
    thermal_rise_coefficient: float = _DEFAULT_THERMAL_RISE_COEFF,
    additional_loss_heating_c: float = 0.0,
) -> float:
    """
    Estimate transformer winding temperature.

    T_winding = T_ambient + (load_fraction² × thermal_rise_coeff) + additional_heating

    Temperature rise follows the square of the load fraction because
    winding losses are I²R and current is proportional to load.

    Args:
        ambient_temperature_c:    Ambient temperature (°C).
        load_fraction:            Load as a fraction of rated capacity (0 to ~2.0).
        thermal_rise_coefficient: Max temperature rise at full load (°C).
        additional_loss_heating_c: Extra heating from fault-induced losses (°C).

    Returns:
        Estimated winding temperature in °C.
    """
    _validate_finite(ambient_temperature_c, "ambient_temperature_c")
    _validate_finite(load_fraction, "load_fraction")

    load_fraction = max(0.0, load_fraction)

    # Temperature rise is proportional to load²
    load_heating = (load_fraction ** 2) * thermal_rise_coefficient
    additional = max(0.0, additional_loss_heating_c)

    temperature = ambient_temperature_c + load_heating + additional
    return temperature


def calculate_thermal_stress(
    transformer_temperature_c: float,
    ambient_temperature_c: float,
    rated_temperature_rise_c: float = _DEFAULT_RATED_TEMP_RISE,
) -> float:
    """
    Calculate thermal stress factor.

    thermal_stress = actual_rise / rated_rise

    A value > 1.0 means the transformer operates above its rated
    thermal envelope, indicating accelerated insulation aging.

    Args:
        transformer_temperature_c: Current winding temperature (°C).
        ambient_temperature_c:     Current ambient temperature (°C).
        rated_temperature_rise_c:  Rated temperature rise (°C).

    Returns:
        Thermal stress factor (≥ 0). Values > 1.0 indicate over-temperature.
    """
    _validate_finite(transformer_temperature_c, "transformer_temperature_c")
    _validate_finite(ambient_temperature_c, "ambient_temperature_c")

    if rated_temperature_rise_c <= 0.0:
        logger.warning("Rated temperature rise is zero or negative; returning 0.0")
        return 0.0

    actual_rise = transformer_temperature_c - ambient_temperature_c
    actual_rise = max(0.0, actual_rise)

    stress = actual_rise / rated_temperature_rise_c
    return max(0.0, stress)


def calculate_thermal_state(
    ambient_temperature_c: float,
    load_fraction: float,
    thermal_rise_coefficient: float = _DEFAULT_THERMAL_RISE_COEFF,
    rated_temperature_rise_c: float = _DEFAULT_RATED_TEMP_RISE,
    additional_loss_heating_c: float = 0.0,
) -> dict:
    """
    Calculate complete thermal state in one call.

    Convenience function that returns both temperature and thermal stress.

    Args:
        ambient_temperature_c:     Ambient temperature (°C).
        load_fraction:             Load as fraction of rated capacity.
        thermal_rise_coefficient:  Max temperature rise at full load (°C).
        rated_temperature_rise_c:  Rated temperature rise (°C).
        additional_loss_heating_c: Extra heating from fault-induced losses (°C).

    Returns:
        Dict with keys:
            - temperature_c: Estimated winding temperature (°C)
            - thermal_stress: Thermal stress factor (≥ 0)
            - temperature_rise_c: Actual temperature rise (°C)
    """
    temperature = calculate_transformer_temperature(
        ambient_temperature_c=ambient_temperature_c,
        load_fraction=load_fraction,
        thermal_rise_coefficient=thermal_rise_coefficient,
        additional_loss_heating_c=additional_loss_heating_c,
    )

    stress = calculate_thermal_stress(
        transformer_temperature_c=temperature,
        ambient_temperature_c=ambient_temperature_c,
        rated_temperature_rise_c=rated_temperature_rise_c,
    )

    return {
        "temperature_c": temperature,
        "thermal_stress": stress,
        "temperature_rise_c": temperature - ambient_temperature_c,
    }


def estimate_loss_induced_heating(
    excess_loss_w: float,
    thermal_resistance: float = 0.01,
) -> float:
    """
    Estimate additional temperature rise from excess losses.

    ΔT_extra = P_excess × R_thermal

    This models the additional heating when feeder resistance increases
    and causes higher I²R losses beyond the normal envelope.

    Args:
        excess_loss_w:      Excess loss compared to normal operation (W).
        thermal_resistance:  Simplified thermal resistance (°C/W).

    Returns:
        Additional temperature rise in °C. Never negative.
    """
    if excess_loss_w <= 0.0:
        return 0.0
    return excess_loss_w * max(0.0, thermal_resistance)


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate_finite(value: float, name: str) -> None:
    """Raise ValueError if value is NaN or infinite."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
