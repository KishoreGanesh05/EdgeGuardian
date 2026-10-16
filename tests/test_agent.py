"""
Tests for the P5 agent layer: AgentTools and the EdgeGuardianAgent fallback.

All tests force the deterministic backend (no OpenAI key / network), proving
the agent is fully functional without GPT-5.
"""

from data.models import FaultScenario
from agent.agent import EdgeGuardianAgent
from agent.tools import (
    EFFICIENCY_TREND_TOLERANCE,
    EFFICIENCY_TREND_Z,
    _trend,
)
from tests.conftest import TARGET


# ── AgentTools ───────────────────────────────────────────────────────────

class TestAgentTools:
    def test_status_known_asset(self, normal_backend):
        status = normal_backend.tools.get_transformer_status(TARGET)
        assert "error" not in status
        for key in (
            "asset_id", "health_index", "health_state",
            "anomaly_score", "rul_hours", "efficiency",
        ):
            assert key in status
        assert status["asset_id"] == TARGET

    def test_status_unknown_asset(self, normal_backend):
        status = normal_backend.tools.get_transformer_status("T999")
        assert "error" in status
        assert status["asset_id"] == "T999"

    def test_fleet_health_aggregates(self, normal_backend):
        fh = normal_backend.tools.get_fleet_health()
        for key in (
            "asset_count", "average_health", "average_efficiency",
            "total_energy_loss_w", "anomaly_count",
        ):
            assert key in fh
        assert fh["asset_count"] == len(normal_backend.configs)

    def test_high_risk_structure(self, normal_backend):
        risk = normal_backend.tools.get_high_risk_assets()
        assert set(risk) >= {
            "critical_count", "warning_count",
            "critical_assets", "warning_assets",
        }

    def test_history_and_trends(self, normal_backend):
        hist = normal_backend.tools.get_asset_history(TARGET)
        assert "trends" in hist
        assert hist["points"] > 0

    def test_rul_tool(self, normal_backend):
        rul = normal_backend.tools.calculate_rul(TARGET)
        assert rul["model_based"] is True
        assert rul["rul_hours"] >= 0

    def test_compare_assets(self, normal_backend):
        a = normal_backend.configs[0].asset_id
        b = normal_backend.configs[1].asset_id
        cmp = normal_backend.tools.compare_assets(a, b)
        assert cmp["healthier_asset"] in (a, b)
        assert "health_index" in cmp["comparison"]

    def test_maintenance_plan_structure(self, normal_backend):
        plan = normal_backend.tools.generate_maintenance_plan(TARGET)
        for key in (
            "priority", "reason", "evidence",
            "recommended_checks", "next_action", "disclaimer",
        ):
            assert key in plan
        assert plan["priority"] in ("LOW", "MEDIUM", "HIGH")
        assert (
            "does not" in plan["disclaimer"].lower()
            or "draft" in plan["disclaimer"].lower()
        )

    def test_scenario_does_not_mutate_live_state(self, normal_backend):
        before = normal_backend.fleet.get_shadow(
            TARGET
        ).state.health_index

        result = normal_backend.tools.run_scenario(
            TARGET,
            "LOAD_INCREASE",
            {"load_increase_percent": 30, "severity": 1.0},
        )

        after = normal_backend.fleet.get_shadow(
            TARGET
        ).state.health_index

        assert after == before  # live state unchanged
        assert "projected" in result and "current" in result
        assert result["note"].lower().startswith("projection")

    def test_scenario_alias_mapping(self, normal_backend):
        res = normal_backend.tools.run_scenario(
            TARGET,
            "AMBIENT_TEMPERATURE_INCREASE",
            {"ambient_increase_c": 20},
        )
        assert "error" not in res
        assert res["scenario"] == FaultScenario.THERMAL_STRESS.value

    def test_scenario_unknown(self, normal_backend):
        res = normal_backend.tools.run_scenario(TARGET, "NONSENSE", {})
        assert "error" in res

    def test_scenario_unknown_asset(self, normal_backend):
        res = normal_backend.tools.run_scenario(
            "T999", "LOAD_INCREASE", {}
        )
        assert "error" in res


# ── EdgeGuardianAgent (deterministic fallback) ─────────────────────────

class TestAgentFallback:
    def test_backend_is_fallback_without_key(self, normal_backend):
        agent = EdgeGuardianAgent(
            normal_backend.tools,
            api_key="",
            force_fallback=True,
        )
        assert agent.uses_llm is False
        assert agent.backend == "deterministic-fallback"

    def test_why_unhealthy_cites_evidence(self, normal_backend):
        out = normal_backend.agent.ask(f"Why is {TARGET} unhealthy?")
        assert out["backend"].startswith("deterministic")
        assert TARGET in out["answer"]
        # Must have actually called tools (evidence-based, not invented).
        names = {c["name"] for c in out["tool_calls"]}
        assert "get_transformer_status" in names

    def test_what_if_runs_scenario(self, normal_backend):
        out = normal_backend.agent.ask(
            f"What if {TARGET} load increases another 20%?"
        )
        names = {c["name"] for c in out["tool_calls"]}
        assert "run_scenario" in names
        assert (
            "projection only" in out["answer"].lower()
            or "projection" in out["answer"].lower()
        )

    def test_what_if_ignores_asset_id_when_extracting_percentage(self):
        scenario, params = EdgeGuardianAgent._infer_scenario(
            "What if T023 load increases another 20%?"
        )
        assert scenario == "LOAD_INCREASE"
        assert params["load_increase_percent"] == 20.0

    def test_maintenance_question(self, normal_backend):
        out = normal_backend.agent.ask(
            f"Give me a maintenance plan for {TARGET}."
        )
        names = {c["name"] for c in out["tool_calls"]}
        assert "generate_maintenance_plan" in names

    def test_fleet_question(self, normal_backend):
        out = normal_backend.agent.ask(
            "How many assets are at risk across the fleet?"
        )
        names = {c["name"] for c in out["tool_calls"]}
        assert "get_fleet_health" in names

    def test_missing_asset_question(self, normal_backend):
        out = normal_backend.agent.ask(
            "Why is this transformer failing?"
        )
        # No asset id and not fleet-level → asks for clarification, invents nothing.
        assert (
            "asset id" in out["answer"].lower()
            or "could not identify" in out["answer"].lower()
        )

    def test_agent_does_not_invent_for_unknown_asset(self, normal_backend):
        out = normal_backend.agent.ask("Why is T999 unhealthy?")
        assert (
            "no data" in out["answer"].lower()
            or "unknown" in out["answer"].lower()
        )


# ── Trend classification ─────────────────────────────────────────────────

def _noisy(level: float, n: int, amplitude: float = 0.008):
    """Deterministic ±amplitude alternating noise around a constant level."""
    return [level + (amplitude if i % 2 else -amplitude) for i in range(n)]


def _efficiency_trend(seq):
    return _trend(
        seq, tolerance=EFFICIENCY_TREND_TOLERANCE, noise_z=EFFICIENCY_TREND_Z
    )


class TestTrend:
    def test_noisy_constant_efficiency_is_flat(self):
        assert _efficiency_trend(_noisy(0.965, 40)) == "flat"

    def test_noise_level_shift_is_flat(self):
        # Half-mean shift of +0.0023 (the demo T023 case) exceeds the 0.002
        # floor but is well within 2 standard errors of per-sample noise.
        seq = _noisy(0.965, 20) + _noisy(0.9673, 20)
        assert _efficiency_trend(seq) == "flat"

    def test_clearly_rising_efficiency(self):
        seq = _noisy(0.950, 20) + _noisy(0.970, 20)
        assert _efficiency_trend(seq) == "rising"

    def test_clearly_falling_efficiency(self):
        seq = _noisy(0.965, 20) + _noisy(0.917, 20)
        assert _efficiency_trend(seq) == "falling"

    def test_insufficient_history(self):
        assert _efficiency_trend([]) == "insufficient_data"
        assert _efficiency_trend([0.96]) == "insufficient_data"

    def test_short_history_uses_tolerance_floor(self):
        # Halves of one point: no variance estimate, so only the floor applies.
        assert _efficiency_trend([0.960, 0.970]) == "rising"
        assert _efficiency_trend([0.965, 0.966]) == "flat"
        assert _efficiency_trend([0.970, 0.965, 0.960]) == "falling"

    def test_other_metrics_keep_exact_sensitivity(self):
        assert _trend([50.0, 50.1]) == "rising"
        assert _trend([50.0, 50.0]) == "flat"

    def test_asset_history_reports_flat_for_noisy_efficiency(self, fresh_backend):
        state = fresh_backend.fleet.get_shadow(TARGET).state
        state.efficiency_history[:] = _noisy(0.965, 40)
        hist = fresh_backend.tools.get_asset_history(TARGET)
        assert hist["trends"]["efficiency"] == "flat"

        state.efficiency_history[:] = _noisy(0.965, 20) + _noisy(0.917, 20)
        hist = fresh_backend.tools.get_asset_history(TARGET)
        assert hist["trends"]["efficiency"] == "falling"
