"""
Edge-Guardian Digital Shadow Module (P3).

Manages the structured current and historical state of assets,
aggregating telemetry, physics, AI, and health results into a single
Digital Shadow representation. Also provides fleet-level management.
"""

from digital_shadow.shadow import DigitalShadow
from digital_shadow.fleet import FleetManager

__all__ = ["DigitalShadow", "FleetManager"]

