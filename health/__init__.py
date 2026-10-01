"""
Edge-Guardian Health Module (P3).

Calculates a bounded Health Index and model-based Remaining Useful Life (RUL)
based on upstream telemetry, physics, and Edge AI anomaly results.
"""

from health.calculator import HealthCalculator

__all__ = ["HealthCalculator"]

