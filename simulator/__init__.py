"""
Edge-Guardian What-If Simulator (P4).

Orchestrates hypothetical scenario testing by applying fault effects
to current asset states and propagating them through the existing
physics, AI, and health pipelines without mutating real state.
"""

from simulator.what_if import WhatIfSimulator

__all__ = ["WhatIfSimulator"]
