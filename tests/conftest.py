"""
Shared pytest fixtures for P5 integration tests.

Training the Edge AI engine (TensorFlow autoencoder + RandomForest) is the
expensive step, so it is done ONCE per session. Each test then gets a fresh,
isolated fleet/agent built cheaply from the shared trained engine, avoiding
cross-test state contamination.
"""

import pytest

from main import build_backend, run_operation, Backend
from data.models import FaultScenario
from digital_shadow.fleet import FleetManager
from digital_shadow.shadow import DigitalShadow
from simulator.what_if import WhatIfSimulator
from agent.tools import AgentTools
from agent.agent import EdgeGuardianAgent
from cloud.base import LocalSink

# Fleet of 23 so the plan's primary target T023 exists.
_ASSET_COUNT = 23
_TARGET = "T023"
_SEED = 42


@pytest.fixture(scope="session")
def trained_backend():
    """A fully-trained backend (expensive). Treat as read-only for the engine."""
    return build_backend(
        asset_count=_ASSET_COUNT,
        seed=_SEED,
        train_normal_s=30,
        train_fault_s=15,
        force_fallback=True,
    )


def _fresh_from(trained: Backend) -> Backend:
    """Build an isolated backend reusing the trained engine (cheap)."""
    fleet = FleetManager()
    for cfg in trained.configs:
        fleet.add_shadow(DigitalShadow(cfg))
    simulator = WhatIfSimulator(trained.ai_engine, trained.health_calculator)
    tools = AgentTools(fleet, simulator)
    agent = EdgeGuardianAgent(tools, force_fallback=True)
    return Backend(
        configs=trained.configs,
        ai_engine=trained.ai_engine,
        health_calculator=trained.health_calculator,
        fleet=fleet,
        simulator=simulator,
        tools=tools,
        agent=agent,
        sink=LocalSink(),
    )


@pytest.fixture
def fresh_backend(trained_backend):
    """A cheap, isolated backend per test (fresh fleet/shadows/agent)."""
    return _fresh_from(trained_backend)


@pytest.fixture
def normal_backend(fresh_backend):
    """Backend after normal operation across the whole fleet (fleet healthy)."""
    for i, cfg in enumerate(fresh_backend.configs):
        run_operation(fresh_backend, cfg, steps=10,
                      scenario=FaultScenario.NORMAL, seed=_SEED + i)
    return fresh_backend


# Expose constants for tests.
TARGET = _TARGET
ASSET_COUNT = _ASSET_COUNT
SEED = _SEED
