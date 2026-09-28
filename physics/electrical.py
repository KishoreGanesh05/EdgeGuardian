"""
Edge-Guardian Electrical Physics Module.

Pure functions implementing fundamental electrical engineering calculations.
These are deterministic, testable equations — no ML, no randomness.

All calculations use SI units unless documented otherwise.

NOTE: This is a physics-informed simulation model for Stage 2 validation,
not a validated transformer protection relay model.
"""

from __future__ import annotations

import math
import logging
from typing import Union, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def calculate_rms(values: Union[Sequence[float], np.ndarray]) -> float:
    """
    Calculate the Root Mean Square (RMS) of a set of values.

    For sinusoidal signals, RMS = peak / sqrt(2).
    For arbitrary waveforms, RMS = sqrt(mean(x²)).

    Args:
        values: Array of instantaneous sample values.

    Returns:
        RMS value. Returns 0.0 for empty input.

    Raises:
        ValueError: If input contains NaN or infinite values.
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0:
        return 0.0

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Input contains NaN or infinite values — cannot compute RMS.")

    return float(np.sqrt(np.mean(arr ** 2)))


def calculate_real_power(voltage: float, current: float, power_factor: float) -> float:
    """
    Calculate real (active) power.

    P = V × I × cos(φ)

    Args:
        voltage:      RMS voltage (V).
        current:      RMS current (A).
        power_factor: Power factor (0 to 1).

    Returns:
        Real power in watts (W). Never negative.
    """
    _validate_non_negative(voltage, "voltage")
    _validate_non_negative(current, "current")
    power_factor = _clamp(power_factor, 0.0, 1.0)
    return abs(voltage * current * power_factor)


def calculate_apparent_power(voltage: float, current: float) -> float:
    """
    Calculate apparent power.

    S = V × I

    Args:
        voltage: RMS voltage (V).
        current: RMS current (A).

    Returns:
        Apparent power in volt-amperes (VA). Never negative.
    """
    _validate_non_negative(voltage, "voltage")
    _validate_non_negative(current, "current")
    return abs(voltage * current)


def calculate_power_factor(real_power: float, apparent_power: float) -> float:
    """
    Calculate power factor from real and apparent power.

    PF = P / S

    Args:
        real_power:     Real power in watts (W).
        apparent_power: Apparent power in volt-amperes (VA).

    Returns:
        Power factor, clamped to [0, 1]. Returns 1.0 if apparent power is zero
        (no-load condition — undefined PF defaults to unity).
    """
    if apparent_power <= 0.0:
        # No-load: undefined power factor, default to unity
        return 1.0

    pf = real_power / apparent_power
    return _clamp(pf, 0.0, 1.0)


def calculate_resistive_loss(current: float, resistance: float) -> float:
    """
    Calculate resistive (I²R) loss.

    P_loss = I² × R

    This is the fundamental feeder/winding loss equation.
    Higher resistance (e.g. from corrosion, loose connections)
    directly increases losses.

    Args:
        current:    RMS current (A).
        resistance: Effective resistance (Ω). Must be non-negative.

    Returns:
        Resistive loss in watts (W). Never negative.
    """
    _validate_non_negative(current, "current")
    _validate_non_negative(resistance, "resistance")
    return current ** 2 * resistance


def calculate_feeder_voltage_drop(current: float, resistance: float) -> float:
    """
    Calculate voltage drop across feeder resistance.

    V_drop = I × R

    Args:
        current:    RMS current (A).
        resistance: Effective feeder resistance (Ω).

    Returns:
        Voltage drop in volts (V).
    """
    _validate_non_negative(current, "current")
    _validate_non_negative(resistance, "resistance")
    return current * resistance


def calculate_effective_resistance(
    base_resistance: float,
    temperature_rise: float,
    thermal_coefficient: float,
) -> float:
    """
    Calculate temperature-adjusted effective resistance.

    R_eff = R_base × (1 + α × ΔT)

    Resistance increases with temperature due to positive thermal
    coefficient of copper/aluminium conductors.

    Args:
        base_resistance:    Nominal resistance at reference temperature (Ω).
        temperature_rise:   Temperature rise above reference (°C).
        thermal_coefficient: Temperature coefficient of resistance (1/°C).

    Returns:
        Effective resistance in ohms (Ω). Never less than base_resistance.
    """
    _validate_non_negative(base_resistance, "base_resistance")
    _validate_non_negative(thermal_coefficient, "thermal_coefficient")

    r_eff = base_resistance * (1.0 + thermal_coefficient * max(temperature_rise, 0.0))
    return max(r_eff, base_resistance)


def calculate_input_power(
    voltage_feeder: float,
    current_feeder: float,
    power_factor: float,
) -> float:
    """
    Calculate input power at the feeder side.

    P_in = V_feeder × I_feeder × PF

    Args:
        voltage_feeder: Feeder-side RMS voltage (V).
        current_feeder: Feeder-side RMS current (A).
        power_factor:   Power factor at the feeder.

    Returns:
        Input power in watts (W).
    """
    return calculate_real_power(voltage_feeder, current_feeder, power_factor)


def calculate_output_power(
    voltage_secondary: float,
    current_secondary: float,
    power_factor: float,
) -> float:
    """
    Calculate output power at the secondary side.

    P_out = V_sec × I_sec × PF

    Args:
        voltage_secondary: Secondary-side RMS voltage (V).
        current_secondary: Secondary-side RMS current (A).
        power_factor:      Power factor at the secondary.

    Returns:
        Output power in watts (W).
    """
    return calculate_real_power(voltage_secondary, current_secondary, power_factor)


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate_non_negative(value: float, name: str) -> None:
    """Raise ValueError if value is negative, NaN, or infinite."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))
