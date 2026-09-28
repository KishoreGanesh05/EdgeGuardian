"""
Edge-Guardian Efficiency Physics Module.

Calculates transformer efficiency and efficiency delta — the key
metrics that connect feeder losses to the Edge AI feature vector.

Efficiency delta (eff_delta) is the primary indicator that something
has changed in the electrical path. When feeder resistance increases,
losses increase, efficiency drops, and eff_delta becomes more negative.
This physics-derived signal is what the Edge AI learns to detect.

NOTE: This is a physics-informed simulation model for Stage 2 validation,
not a validated transformer efficiency standard.
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


def calculate_efficiency(input_power: float, output_power: float) -> float:
    """
    Calculate transformer efficiency.

    η = P_out / P_in

    Args:
        input_power:  Input power at feeder side (W). Must be positive.
        output_power: Output power at secondary side (W). Must be non-negative.

    Returns:
        Efficiency as a fraction [0, 1].
        Returns 0.0 if input_power is zero or negative (no-load / invalid).
    """
    if math.isnan(input_power) or math.isinf(input_power):
        raise ValueError(f"input_power must be finite, got {input_power}")
    if math.isnan(output_power) or math.isinf(output_power):
        raise ValueError(f"output_power must be finite, got {output_power}")

    if input_power <= 0.0:
        logger.debug("Input power is zero or negative; returning efficiency 0.0")
        return 0.0

    if output_power < 0.0:
        logger.warning("Negative output power detected; clamping to 0.0")
        output_power = 0.0

    efficiency = output_power / input_power

    # Clamp to [0, 1] — efficiency cannot exceed 100% in a passive transformer
    return max(0.0, min(1.0, efficiency))


def calculate_efficiency_delta(
    current_efficiency: float,
    baseline_efficiency: float,
) -> float:
    """
    Calculate efficiency delta from baseline.

    eff_delta = current_efficiency - baseline_efficiency

    A negative eff_delta indicates the transformer is less efficient
    than its baseline — typically caused by increased losses.

    This is one of the seven physics-derived features fed to Edge AI.
    The AI learns that sustained negative eff_delta is abnormal.

    Args:
        current_efficiency:  Currently measured efficiency [0, 1].
        baseline_efficiency: Baseline/design efficiency [0, 1].

    Returns:
        Efficiency delta (can be negative, zero, or slightly positive).
    """
    if math.isnan(current_efficiency) or math.isinf(current_efficiency):
        raise ValueError(f"current_efficiency must be finite, got {current_efficiency}")
    if math.isnan(baseline_efficiency) or math.isinf(baseline_efficiency):
        raise ValueError(f"baseline_efficiency must be finite, got {baseline_efficiency}")

    return current_efficiency - baseline_efficiency


def calculate_loss_from_efficiency(
    input_power: float,
    efficiency: float,
) -> float:
    """
    Calculate total losses from input power and efficiency.

    P_loss = P_in × (1 - η)

    Args:
        input_power: Input power (W).
        efficiency:  Current efficiency [0, 1].

    Returns:
        Total losses in watts (W). Never negative.
    """
    if input_power <= 0.0:
        return 0.0

    loss = input_power * (1.0 - max(0.0, min(1.0, efficiency)))
    return max(0.0, loss)
