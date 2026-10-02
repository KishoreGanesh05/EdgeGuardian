"""
Edge-Guardian Agent Prompts & Tool Schemas (P5).

Holds the system prompt (plan §35) and the OpenAI-style function-calling
tool schemas that mirror AgentTools. Kept separate from agent.py so the
behavioral contract is easy to review and audit.
"""

from __future__ import annotations

from typing import List, Dict


SYSTEM_PROMPT = """You are the Edge-Guardian engineering assistant.

Analyze transformer condition using structured system tools.

Do not invent measurements.

Use tool results as the source of truth for current asset state.

Distinguish between:
- simulated/ingested telemetry
- physics-derived values
- ML predictions (anomaly score, fault class)
- model-based RUL
- recommendations

Do not claim the Edge-Guardian Health Index or RUL is industry-certified.

Do not issue breaker commands or safety-critical control actions.

When evidence is insufficient or a tool returns an error, state what is
missing rather than guessing.

When explaining why an asset is unhealthy, cite the specific tool values
(health index, anomaly score, fault class, efficiency delta, thermal stress,
load) that support your conclusion.
"""


def get_tool_schemas() -> List[Dict]:
    """
    Return OpenAI function-calling tool schemas for AgentTools.

    The names match AgentTools method names exactly so the agent can
    dispatch tool calls by name.
    """
    def fn(name, description, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    asset = {"type": "string", "description": "Asset identifier, e.g. 'T023'."}

    return [
        fn("get_transformer_status",
           "Current authoritative state snapshot for one transformer.",
           {"asset_id": asset}, ["asset_id"]),
        fn("get_asset_history",
           "Recent bounded history and trend direction for one transformer.",
           {"asset_id": asset,
            "max_points": {"type": "integer", "description": "Max history points (default 50)."}},
           ["asset_id"]),
        fn("get_fleet_health",
           "Aggregated fleet-level health, efficiency, loss and anomaly metrics.",
           {}, []),
        fn("get_high_risk_assets",
           "List assets currently in CRITICAL or WARNING state.",
           {}, []),
        fn("get_anomaly_details",
           "Focused anomaly and fault-classification evidence for one asset.",
           {"asset_id": asset}, ["asset_id"]),
        fn("calculate_rul",
           "Current model-based remaining-useful-life estimate for one asset.",
           {"asset_id": asset}, ["asset_id"]),
        fn("run_scenario",
           "Run a what-if scenario on an asset without mutating live state. "
           "scenario may be FEEDER_RESISTANCE_INCREASE, OVERLOAD, THERMAL_STRESS, "
           "EFFICIENCY_DEGRADATION, or aliases LOAD_INCREASE / "
           "AMBIENT_TEMPERATURE_INCREASE.",
           {"asset_id": asset,
            "scenario": {"type": "string", "description": "Scenario name or alias."},
            "parameters": {"type": "object",
                           "description": "Optional params, e.g. {'severity':1.0,'load_increase_percent':20}."}},
           ["asset_id", "scenario"]),
        fn("compare_assets",
           "Compare key metrics of two assets side by side.",
           {"asset_a": asset, "asset_b": {"type": "string", "description": "Second asset id."}},
           ["asset_a", "asset_b"]),
        fn("generate_maintenance_plan",
           "Produce a DRAFT engineering maintenance plan from current evidence.",
           {"asset_id": asset}, ["asset_id"]),
    ]


# Tool names that take a single asset_id — used by the deterministic fallback.
ASSET_TOOLS = {
    "get_transformer_status",
    "get_asset_history",
    "get_anomaly_details",
    "calculate_rul",
    "generate_maintenance_plan",
}
