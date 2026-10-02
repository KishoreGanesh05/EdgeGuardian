"""
Edge-Guardian Engineering Assistant (P5).

Wraps the AgentTools surface with two interchangeable reasoning backends:

  1. LLM backend (optional): if an OpenAI API key is configured and the
     `openai` package is installed, the agent runs a GPT-5/GPT-4o
     tool-calling loop using the schemas in prompts.get_tool_schemas().

  2. Deterministic fallback (always available): a rule-based investigator
     that calls the SAME tools and composes an evidence-based answer from
     real values only. This guarantees `python main.py --demo` works with
     NO API key and NO network — the core never depends on GPT-5 (plan §34).

The agent never fabricates measurements; it only reports what tools return.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from agent.tools import AgentTools
from agent.prompts import SYSTEM_PROMPT, get_tool_schemas
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

_ASSET_RE = re.compile(r"\b([A-Za-z]\d{2,4})\b")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


def openai_available() -> bool:
    """True if the optional 'openai' package is importable."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


class EdgeGuardianAgent:
    """
    Tool-driven engineering assistant.

    Args:
        tools:   An AgentTools instance (dependency-injected).
        api_key: OpenAI key; defaults to settings.OPENAI_API_KEY.
        model:   Model id; defaults to settings.OPENAI_MODEL.
        force_fallback: If True, always use the deterministic backend.
    """

    def __init__(
        self,
        tools: AgentTools,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        force_fallback: bool = False,
    ):
        self._tools = tools
        self._api_key = api_key if api_key is not None else OPENAI_API_KEY
        self._model = model or OPENAI_MODEL
        self._force_fallback = force_fallback

    # ── capability ───────────────────────────────────────────────────────

    @property
    def uses_llm(self) -> bool:
        """Whether a real LLM backend is active for this agent."""
        return (
            not self._force_fallback
            and bool(self._api_key)
            and openai_available()
        )

    @property
    def backend(self) -> str:
        return "llm" if self.uses_llm else "deterministic-fallback"

    # ── tool dispatch ────────────────────────────────────────────────────

    def _dispatch(self, name: str, args: Dict) -> Dict:
        """Invoke a named AgentTools method with keyword args."""
        method = getattr(self._tools, name, None)
        if method is None or not callable(method):
            return {"error": f"Unknown tool '{name}'"}
        try:
            return method(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for '{name}': {exc}"}

    # ── public API ───────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict:
        """
        Answer an engineering question using the tool layer.

        Returns a dict:
            {"answer": str, "backend": str, "tool_calls": [ {name,args,result} ]}
        """
        if self.uses_llm:
            try:
                return self._ask_llm(question)
            except Exception as exc:  # pragma: no cover - network/SDK failures
                logger.warning("LLM backend failed (%s); using fallback", exc)
                result = self._ask_fallback(question)
                result["backend"] = "deterministic-fallback (llm-error)"
                return result
        return self._ask_fallback(question)

    # ── LLM backend ──────────────────────────────────────────────────────

    def _ask_llm(self, question: str, max_rounds: int = 6) -> Dict:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        schemas = get_tool_schemas()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_log: List[Dict] = []

        for _ in range(max_rounds):
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=schemas,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                return {
                    "answer": msg.content or "",
                    "backend": "llm",
                    "model": self._model,
                    "tool_calls": tool_log,
                }

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch(name, args)
                tool_log.append({"name": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        return {
            "answer": "Investigation did not converge within the tool-call budget.",
            "backend": "llm",
            "model": self._model,
            "tool_calls": tool_log,
        }

    # ── Deterministic fallback backend ───────────────────────────────────

    def _ask_fallback(self, question: str) -> Dict:
        q = question.lower()
        asset_ids = _ASSET_RE.findall(question)
        asset_id = asset_ids[0].upper() if asset_ids else None
        tool_log: List[Dict] = []

        def call(name: str, **args) -> Dict:
            result = self._dispatch(name, args)
            tool_log.append({"name": name, "args": args, "result": result})
            return result

        # ── Fleet-level questions ────────────────────────────────────────
        if asset_id is None and any(w in q for w in ("fleet", "overall", "how many", "risk", "critical")):
            fleet = call("get_fleet_health")
            risk = call("get_high_risk_assets")
            answer = self._compose_fleet(fleet, risk)
            return {"answer": answer, "backend": "deterministic-fallback", "tool_calls": tool_log}

        if asset_id is None:
            return {
                "answer": (
                    "I could not identify an asset in the question. Please include "
                    "an asset id such as 'T023', or ask a fleet-level question."
                ),
                "backend": "deterministic-fallback",
                "tool_calls": tool_log,
            }

        # ── What-if questions ────────────────────────────────────────────
        if any(w in q for w in ("what if", "what-if", "scenario", "would happen", "if ")):
            scenario, params = self._infer_scenario(q)
            result = call("run_scenario", asset_id=asset_id, scenario=scenario, parameters=params)
            answer = self._compose_scenario(asset_id, result)
            return {"answer": answer, "backend": "deterministic-fallback", "tool_calls": tool_log}

        # ── Maintenance questions ────────────────────────────────────────
        if any(w in q for w in ("maintenance", "plan", "fix", "repair", "action")):
            status = call("get_transformer_status", asset_id=asset_id)
            plan = call("generate_maintenance_plan", asset_id=asset_id)
            answer = self._compose_maintenance(asset_id, status, plan)
            return {"answer": answer, "backend": "deterministic-fallback", "tool_calls": tool_log}

        # ── Default: health / "why unhealthy" investigation ──────────────
        status = call("get_transformer_status", asset_id=asset_id)
        anomaly = call("get_anomaly_details", asset_id=asset_id)
        history = call("get_asset_history", asset_id=asset_id)
        plan = call("generate_maintenance_plan", asset_id=asset_id)
        answer = self._compose_health(asset_id, status, anomaly, history, plan)
        return {"answer": answer, "backend": "deterministic-fallback", "tool_calls": tool_log}

    # ── scenario inference for fallback ──────────────────────────────────

    @staticmethod
    def _infer_scenario(q: str):
        nums = _NUM_RE.findall(q)
        amount = float(nums[0]) if nums else 20.0
        if "resistance" in q or "feeder" in q:
            return "FEEDER_RESISTANCE_INCREASE", {"resistance_increase_percent": amount}
        if "ambient" in q or "temperature" in q or "heat" in q:
            return "AMBIENT_TEMPERATURE_INCREASE", {"ambient_increase_c": amount}
        # default: load increase
        return "LOAD_INCREASE", {"load_increase_percent": amount}

    # ── answer composition (facts only) ──────────────────────────────────

    @staticmethod
    def _compose_health(asset_id, status, anomaly, history, plan) -> str:
        if "error" in status:
            return f"No data available for {asset_id}: {status['error']}."
        lines = [
            f"{asset_id} health assessment (evidence from system tools):",
            f"- Health index: {status['health_index']} ({status['health_state']}).",
            f"- Anomaly: {'YES' if status['is_anomaly'] else 'no'} "
            f"(score={status['anomaly_score']}, fault_class={status['fault_class']}, "
            f"confidence={status['fault_confidence']}).",
            f"- Efficiency delta vs baseline: {status['eff_delta']}; "
            f"thermal stress: {status['thermal_stress']}; load: {status['load_percent']}%.",
            f"- Model-based RUL: {status['rul_hours']} h.",
        ]
        if isinstance(history, dict) and "trends" in history:
            t = history["trends"]
            lines.append(
                f"- Trends — health: {t['health']}, anomaly: {t['anomaly']}, "
                f"efficiency: {t['efficiency']}, temperature: {t['temperature']}."
            )
        # Causal narrative driven strictly by values
        causes = []
        if status["eff_delta"] is not None and status["eff_delta"] < -0.005:
            causes.append("reduced efficiency (higher losses)")
        if status["thermal_stress"] is not None and status["thermal_stress"] > 1.0:
            causes.append("elevated thermal stress")
        if status["load_percent"] is not None and status["load_percent"] > 100.0:
            causes.append("overloading")
        if status["is_anomaly"] and status["fault_class"]:
            causes.append(f"a detected {status['fault_class']} pattern")
        if causes:
            lines.append("Likely drivers: " + ", ".join(causes) + ".")
        else:
            lines.append("No abnormal drivers are present in current evidence.")
        if isinstance(plan, dict) and "priority" in plan:
            lines.append(f"Draft maintenance priority: {plan['priority']} — {plan['next_action']}")
        lines.append(
            "Note: Health Index and RUL are model-based Stage 2 metrics, not industry-certified."
        )
        return "\n".join(lines)

    @staticmethod
    def _compose_scenario(asset_id, result) -> str:
        if "error" in result:
            return f"Cannot run scenario on {asset_id}: {result['error']}."
        cur, proj, d = result["current"], result["projected"], result["deltas"]
        lines = [
            f"What-if for {asset_id} — scenario {result['scenario']} "
            f"(params={result['parameters']}). Projection only; live state unchanged.",
            f"- Health index: {cur['health_index']} -> {proj['health_index']} "
            f"(delta {d.get('health_index')}).",
            f"- RUL hours: {cur['rul_hours']} -> {proj['rul_hours']} (delta {d.get('rul_hours')}).",
            f"- Efficiency: {cur['efficiency']} -> {proj['efficiency']} (delta {d.get('efficiency')}).",
            f"- Temperature: {cur['temperature']} -> {proj['temperature']} (delta {d.get('temperature')}).",
            f"- Anomaly score: {cur['anomaly_score']} -> {proj['anomaly_score']} "
            f"(delta {d.get('anomaly_score')}).",
        ]
        if result["warnings"]:
            lines.append("Warnings: " + "; ".join(result["warnings"]) + ".")
        return "\n".join(lines)

    @staticmethod
    def _compose_maintenance(asset_id, status, plan) -> str:
        if "error" in plan:
            return f"No data available for {asset_id}: {plan['error']}."
        lines = [
            f"Draft maintenance plan for {asset_id} (priority {plan['priority']}):",
            f"- Reason: {plan['reason']}",
            "- Evidence: " + "; ".join(plan["evidence"]),
            "- Recommended checks: " + "; ".join(plan["recommended_checks"]),
            f"- Next action: {plan['next_action']}",
            f"- {plan['disclaimer']}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _compose_fleet(fleet, risk) -> str:
        lines = [
            "Fleet summary (evidence from system tools):",
            f"- Assets: {fleet.get('asset_count')} "
            f"(normal={fleet.get('healthy_count')}, warning={fleet.get('warning_count')}, "
            f"critical={fleet.get('critical_count')}).",
            f"- Average health: {fleet.get('average_health')}; "
            f"minimum health: {fleet.get('minimum_health')}.",
            f"- Average efficiency: {fleet.get('average_efficiency')}; "
            f"total energy loss: {fleet.get('total_energy_loss_w')} W; "
            f"anomalies: {fleet.get('anomaly_count')}.",
        ]
        if risk.get("critical_assets"):
            ids = ", ".join(a["asset_id"] for a in risk["critical_assets"])
            lines.append(f"- Critical assets: {ids}.")
        if risk.get("warning_assets"):
            ids = ", ".join(a["asset_id"] for a in risk["warning_assets"])
            lines.append(f"- Warning assets: {ids}.")
        return "\n".join(lines)
