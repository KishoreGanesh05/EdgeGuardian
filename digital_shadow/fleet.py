"""
Edge-Guardian Fleet Manager (P3).

Aggregates multiple Digital Shadow instances to provide fleet-level metrics
and centralized asset management.
"""

from typing import Dict, Optional
from data.models import HealthState
from digital_shadow.shadow import DigitalShadow

class FleetManager:
    """
    Manages a collection of Digital Shadows and provides fleet-level metrics.
    """

    def __init__(self):
        self._shadows: Dict[str, DigitalShadow] = {}

    def add_shadow(self, shadow: DigitalShadow) -> None:
        """Register a Digital Shadow in the fleet."""
        self._shadows[shadow.state.asset_id] = shadow

    def get_shadow(self, asset_id: str) -> Optional[DigitalShadow]:
        """Retrieve a specific Digital Shadow by asset_id."""
        return self._shadows.get(asset_id)

    def get_fleet_metrics(self) -> dict:
        """
        Calculate deterministic aggregated metrics for the entire fleet.
        """
        if not self._shadows:
            return {
                "asset_count": 0,
                "healthy_count": 0,
                "warning_count": 0,
                "critical_count": 0,
                "average_health": 0.0,
                "minimum_health": 0.0,
                "average_rul_hours": 0.0,
            }

        states = [s.state for s in self._shadows.values()]
        healths = [s.health_index for s in states]
        ruls = [s.rul_hours for s in states]

        return {
            "asset_count": len(states),
            "healthy_count": sum(1 for s in states if s.health_state == HealthState.NORMAL),
            "warning_count": sum(1 for s in states if s.health_state == HealthState.WARNING),
            "critical_count": sum(1 for s in states if s.health_state == HealthState.CRITICAL),
            "average_health": sum(healths) / len(healths),
            "minimum_health": min(healths),
            "average_rul_hours": sum(ruls) / len(ruls),
        }

