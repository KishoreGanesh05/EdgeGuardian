"""
Edge-Guardian Agent Tools (P5).

The structured tool layer the GPT-5 agent uses to inspect system state.
Every tool:
    - reads from the authoritative Digital Shadow / Fleet (never recomputes),
    - returns a plain, JSON-serializable dict of summarized evidence,
    - never fabricates values; missing data yields an explicit {"error": ...}.

These tools are the ONLY way the agent is allowed to see asset state. The
agent must never read raw datasets or recompute physics itself (plan §34).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from data.models import FaultScenario, HealthState
from digital_shadow.fleet import FleetManager
from digital_shadow.shadow import DigitalShadow
from simulator.what_if import WhatIfSimulator

logger = logging.getLogger(__name__)


# What-if scenario aliases (plan §20) mapped onto the FaultScenario enum.
# Each entry: alias -> (FaultScenario, parameter translation function).
def _load_increase_params(params: Dict) -> Dict:
    """Translate a 'load_increase_percent' into OVERLOAD's overload_percent."""
    out = dict(params)
    inc = out.pop("load_increase_percent", None)
    if inc is not None and "overload_percent" not in out:
        out["overload_percent"] = 100.0 + float(inc)
    return out


_SCENARIO_ALIASES = {
    "LOAD_INCREASE": (FaultScenario.OVERLOAD, _load_increase_params),
    "AMBIENT_TEMPERATURE_INCREASE": (FaultScenario.THERMAL_STRESS, lambda p: dict(p)),
}


def _round(value, ndigits: int = 4):
    """Round floats for compact evidence; pass through non-floats/None."""
    if isinstance(value, float):
        return round(value, ndigits)
    return value


class AgentTools:
    """
    Dependency-injected tool surface over a FleetManager + WhatIfSimulator.

    The agent (LLM or deterministic fallback) calls these methods and
    receives structured evidence. No tool mutates live state.
    """

    def __init__(self, fleet: FleetManager, simulator: WhatIfSimulator):
        self._fleet = fleet
        self._simulator = simulator

    # ── internal helpers ────────────────────────────────────────────────

    def _shadows(self) -> List[DigitalShadow]:
        """All shadows in the fleet (read-only view)."""
        return list(getattr(self._fleet, "_shadows", {}).values())

    def _require_shadow(self, asset_id: str) -> Optional[DigitalShadow]:
        return self._fleet.get_shadow(asset_id)

    @staticmethod
    def _error(asset_id: str, message: str) -> Dict:
        logger.warning("AgentTools error for %s: %s", asset_id, message)
        return {"error": message, "asset_id": asset_id}

    # ── Tool: transformer status ─────────────────────────────────────────

    def get_transformer_status(self, asset_id: str) -> Dict:
        """Current authoritative state snapshot for one asset."""
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")
        s = shadow.state
        return {
            "asset_id": s.asset_id,
            "timestamp": s.timestamp.isoformat(),
            "rated_power_kva": _round(s.config.rated_power_kva, 1),
            "load_percent": _round(s.load_percent, 2),
            "temperature_c": _round(s.temperature, 2),
            "efficiency": _round(s.efficiency, 6),
            "power_factor": _round(s.power_factor, 4),
            "health_index": _round(s.health_index, 2),
            "health_state": s.health_state.value,
            "anomaly_score": _round(s.anomaly_score, 6),
            "is_anomaly": bool(s.is_anomaly),
            "fault_class": s.fault_class,
            "fault_confidence": _round(s.fault_confidence, 4),
            "rul_hours": _round(s.rul_hours, 1),
            "degradation_score": _round(s.degradation_score, 4),
            "thermal_stress": _round(s.thermal_stress, 4),
            "eff_delta": _round(s.eff_delta, 6),
            "loss_w": _round(s.loss_w, 2),
        }

    # ── Tool: asset history ──────────────────────────────────────────────

    def get_asset_history(self, asset_id: str, max_points: int = 50) -> Dict:
        """Recent bounded history plus simple trend direction for one asset."""
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")
        s = shadow.state

        def tail(seq: List[float]) -> List[float]:
            return [_round(v, 4) for v in seq[-max_points:]]

        def trend(seq: List[float]) -> str:
            if len(seq) < 2:
                return "insufficient_data"
            delta = seq[-1] - seq[0]
            if abs(delta) < 1e-9:
                return "flat"
            return "rising" if delta > 0 else "falling"

        return {
            "asset_id": asset_id,
            "points": min(len(s.health_history), max_points),
            "health_history": tail(s.health_history),
            "anomaly_history": tail(s.anomaly_history),
            "efficiency_history": tail(s.efficiency_history),
            "temperature_history": tail(s.temperature_history),
            "trends": {
                "health": trend(s.health_history),
                "anomaly": trend(s.anomaly_history),
                "efficiency": trend(s.efficiency_history),
                "temperature": trend(s.temperature_history),
            },
        }

    # ── Tool: fleet health ───────────────────────────────────────────────

    def get_fleet_health(self) -> Dict:
        """Aggregated fleet metrics computed from individual Digital Shadows."""
        metrics = self._fleet.get_fleet_metrics()
        shadows = self._shadows()
        if shadows:
            efficiencies = [s.state.efficiency for s in shadows]
            total_loss = sum(s.state.loss_w for s in shadows)
            anomaly_count = sum(1 for s in shadows if s.state.is_anomaly)
            metrics = dict(metrics)
            metrics["average_efficiency"] = round(sum(efficiencies) / len(efficiencies), 6)
            metrics["total_energy_loss_w"] = round(total_loss, 2)
            metrics["anomaly_count"] = anomaly_count
        else:
            metrics = dict(metrics)
            metrics["average_efficiency"] = 0.0
            metrics["total_energy_loss_w"] = 0.0
            metrics["anomaly_count"] = 0
        # Round core numeric metrics for compactness
        for k in ("average_health", "minimum_health", "average_rul_hours"):
            if k in metrics:
                metrics[k] = round(metrics[k], 2)
        return metrics

    # ── Tool: high-risk assets ───────────────────────────────────────────

    def get_high_risk_assets(self) -> Dict:
        """Assets in CRITICAL or WARNING state, plus any flagged anomalies."""
        critical, warning = [], []
        for shadow in self._shadows():
            s = shadow.state
            entry = {
                "asset_id": s.asset_id,
                "health_index": _round(s.health_index, 2),
                "health_state": s.health_state.value,
                "anomaly_score": _round(s.anomaly_score, 6),
                "fault_class": s.fault_class,
                "rul_hours": _round(s.rul_hours, 1),
            }
            if s.health_state == HealthState.CRITICAL:
                critical.append(entry)
            elif s.health_state == HealthState.WARNING:
                warning.append(entry)
        critical.sort(key=lambda e: e["health_index"])
        warning.sort(key=lambda e: e["health_index"])
        return {
            "critical_count": len(critical),
            "warning_count": len(warning),
            "critical_assets": critical,
            "warning_assets": warning,
        }

    # ── Tool: anomaly details ────────────────────────────────────────────

    def get_anomaly_details(self, asset_id: str) -> Dict:
        """Focused anomaly/fault evidence for one asset."""
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")
        s = shadow.state
        return {
            "asset_id": asset_id,
            "is_anomaly": bool(s.is_anomaly),
            "anomaly_score": _round(s.anomaly_score, 6),
            "fault_class": s.fault_class,
            "fault_confidence": _round(s.fault_confidence, 4),
            "contributing_factors": {
                "efficiency_delta": _round(s.eff_delta, 6),
                "thermal_stress": _round(s.thermal_stress, 4),
                "load_percent": _round(s.load_percent, 2),
                "loss_w": _round(s.loss_w, 2),
            },
            "model_note": (
                "Anomaly score is autoencoder reconstruction error (MSE); "
                "fault_class is a RandomForest prediction. Both are model-based."
            ),
        }

    # ── Tool: RUL ────────────────────────────────────────────────────────

    def calculate_rul(self, asset_id: str) -> Dict:
        """Current model-based RUL estimate for one asset (read from shadow)."""
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")
        s = shadow.state
        return {
            "asset_id": asset_id,
            "rul_hours": _round(s.rul_hours, 1),
            "rul_days": _round(s.rul_hours / 24.0, 1),
            "health_index": _round(s.health_index, 2),
            "degradation_score": _round(s.degradation_score, 4),
            "model_based": True,
            "note": "Model-based synthetic estimate — NOT validated against real failure data.",
        }

    # ── Tool: what-if scenario ───────────────────────────────────────────

    def run_scenario(
        self,
        asset_id: str,
        scenario: str,
        parameters: Optional[Dict] = None,
    ) -> Dict:
        """
        Run a what-if scenario on an asset WITHOUT mutating live state.

        Accepts canonical FaultScenario names and the what-if aliases
        LOAD_INCREASE and AMBIENT_TEMPERATURE_INCREASE (plan §20).
        """
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")

        params = dict(parameters or {})
        severity = float(params.pop("severity", 1.0))

        name = (scenario or "").upper()
        if name in _SCENARIO_ALIASES:
            enum_scenario, translate = _SCENARIO_ALIASES[name]
            params = translate(params)
        else:
            try:
                enum_scenario = FaultScenario(name)
            except ValueError:
                return self._error(
                    asset_id,
                    f"Unknown scenario '{scenario}'. Valid: "
                    f"{[s.value for s in FaultScenario]} "
                    f"or aliases {list(_SCENARIO_ALIASES.keys())}",
                )

        # Snapshot live health to prove non-mutation to callers/tests.
        pre_health = shadow.state.health_index

        result = self._simulator.run_scenario(
            shadow=shadow,
            scenario=enum_scenario,
            severity=severity,
            parameters=params,
        )

        post_health = shadow.state.health_index
        if abs(post_health - pre_health) > 1e-9:
            logger.error(
                "run_scenario mutated live state for %s (%.4f -> %.4f)",
                asset_id, pre_health, post_health,
            )

        def pick(state: Dict) -> Dict:
            return {
                "health_index": _round(state.get("health_index"), 2),
                "health_state": state.get("health_state"),
                "rul_hours": _round(state.get("rul_hours"), 1),
                "efficiency": _round(state.get("efficiency"), 6),
                "temperature": _round(state.get("temperature"), 2),
                "anomaly_score": _round(state.get("anomaly_score"), 6),
                "load_percent": _round(state.get("load_percent"), 2),
            }

        return {
            "asset_id": asset_id,
            "scenario": result.scenario,
            "parameters": result.parameters,
            "current": pick(result.current_state),
            "projected": pick(result.projected_state),
            "deltas": {k: _round(v, 4) for k, v in result.deltas.items()},
            "warnings": list(result.warnings),
            "note": "Projection only — live Digital Shadow is unchanged.",
        }

    # ── Tool: compare two assets ─────────────────────────────────────────

    def compare_assets(self, asset_a: str, asset_b: str) -> Dict:
        """Side-by-side comparison of two assets' key metrics."""
        a = self.get_transformer_status(asset_a)
        b = self.get_transformer_status(asset_b)
        if "error" in a:
            return a
        if "error" in b:
            return b
        metrics = [
            "health_index", "rul_hours", "efficiency",
            "temperature_c", "anomaly_score", "load_percent",
        ]
        comparison = {
            m: {
                asset_a: a.get(m),
                asset_b: b.get(m),
                "difference": _round(
                    (a.get(m) or 0.0) - (b.get(m) or 0.0), 4
                ),
            }
            for m in metrics
        }
        healthier = asset_a if (a["health_index"] >= b["health_index"]) else asset_b
        return {
            "asset_a": asset_a,
            "asset_b": asset_b,
            "healthier_asset": healthier,
            "comparison": comparison,
        }

    # ── Tool: maintenance plan ───────────────────────────────────────────

    def generate_maintenance_plan(self, asset_id: str) -> Dict:
        """
        Produce a DRAFT engineering maintenance plan from current evidence.

        Stage 2: a draft recommendation only — it does not dispatch work.
        """
        shadow = self._require_shadow(asset_id)
        if shadow is None:
            return self._error(asset_id, f"Unknown asset '{asset_id}'")
        s = shadow.state

        evidence: List[str] = []
        checks: List[str] = []

        if s.is_anomaly:
            evidence.append(
                f"Anomaly detected (score={round(s.anomaly_score, 4)}, "
                f"class={s.fault_class or 'UNKNOWN'})."
            )
        if s.eff_delta < -0.005:
            evidence.append(f"Efficiency down {round(-s.eff_delta * 100, 2)}% vs baseline.")
            checks.append("Inspect feeder connections and measure contact resistance.")
        if s.thermal_stress > 1.0:
            evidence.append(f"Thermal stress elevated ({round(s.thermal_stress, 2)}).")
            checks.append("Verify cooling/ventilation and check winding temperature sensors.")
        if s.load_percent > 100.0:
            evidence.append(f"Operating above rated load ({round(s.load_percent, 1)}%).")
            checks.append("Review loading schedule; assess load transfer options.")

        fault = (s.fault_class or "").upper()
        if fault == "FEEDER_RESISTANCE_INCREASE":
            checks.append("Thermographic survey of feeder joints and terminations.")

        # Priority from health state / RUL
        if s.health_state == HealthState.CRITICAL:
            priority = "HIGH"
            next_action = "Schedule inspection within 24-48 hours."
        elif s.health_state == HealthState.WARNING:
            priority = "MEDIUM"
            next_action = "Plan inspection at next maintenance window."
        else:
            priority = "LOW"
            next_action = "Continue routine monitoring."

        if not evidence:
            evidence.append("No abnormal indicators in current state.")
        if not checks:
            checks.append("Standard periodic inspection.")

        reason = (
            f"Health {round(s.health_index, 1)} ({s.health_state.value}); "
            f"model-based RUL {round(s.rul_hours, 1)} h."
        )

        return {
            "asset_id": asset_id,
            "priority": priority,
            "reason": reason,
            "evidence": evidence,
            "recommended_checks": checks,
            "next_action": next_action,
            "disclaimer": (
                "DRAFT engineering recommendation for Stage 2. Does not dispatch "
                "maintenance or issue any control/safety action."
            ),
        }
