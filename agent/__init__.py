"""
Edge-Guardian agent layer (P5).

GPT-5 is a reasoning/orchestration layer only. It inspects system state
through the structured AgentTools surface and never computes physics,
detects anomalies, or issues control actions. The core pipeline runs
without the GPT-5 API (deterministic fallback backend).
"""

from agent.tools import AgentTools
from agent.agent import EdgeGuardianAgent

__all__ = ["AgentTools", "EdgeGuardianAgent"]
