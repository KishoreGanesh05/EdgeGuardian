"""
Edge-Guardian — Main Pipeline & End-to-End Demo (P5).

This module is the integration layer that wires P1–P4 together and exposes
the `python main.py --demo` command required by the implementation plan
(§25, §26, §40).

Orchestration only — all calculations live in their P1–P4 modules:

    Telemetry → Physics → Features → Edge AI → Health → RUL → Digital Shadow
                                                                    ↓
                                                         Fleet / What-If / Agent

The demo runs fully locally: no hardware, no AWS credentials, no GPT-5 API.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config.settings import setup_logging, DEMO_SEED
from data.models import (
    TransformerConfig,
    TelemetrySample,
    PhysicsResult,
    FeatureVector,
    AIResult,
    HealthResult,
    RULResult,
    FaultScenario,
    HealthState,
)
from data.generator import (
    generate_fleet_configs,
    generate_telemetry,
    generate_training_dataset,
    calculate_physics,
    extract_feature_vector,
)
from edge_ai.features import EdgeAIEngine
from health.calculator import HealthCalculator
from digital_shadow.shadow import DigitalShadow
from digital_shadow.fleet import FleetManager
from simulator.what_if import WhatIfSimulator
from agent.tools import AgentTools
from agent.agent import EdgeGuardianAgent
from cloud.base import TelemetrySink, LocalSink

logger = logging.getLogger("edge_guardian.main")


# Demo defaults. The plan's primary story injects a fault into T023, so the
# demo fleet must be large enough to contain it (T023 => >= 23 assets).
#
# Fault scenario note: the plan's narrative names FEEDER_RESISTANCE_INCREASE,
# but in the Stage-2 P1 physics that fault is numerically weak at 11 kV
# (efficiency is insensitive to feeder resistance; only temperature responds,
# and only slightly), so it does not reliably trip the autoencoder at realistic
# severities. The demo therefore injects OVERLOAD into T023, which genuinely
# exercises the full detection -> health -> RUL cascade. FEEDER_RESISTANCE_INCREASE
# remains a fully supported scenario (see README "Known limitations").
DEMO_ASSET_COUNT = 25
DEMO_TARGET_ASSET = "T023"
DEMO_FAULT_SCENARIO = FaultScenario.OVERLOAD
DEMO_FAULT_PARAMS = {"overload_percent": 160.0}
DEMO_FAULT_SEVERITY = 1.0
DEMO_NORMAL_STEPS = 20
DEMO_FAULT_STEPS = 20


# ═══════════════════════════════════════════════════════════════════════════
# Core orchestration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Backend:
    """Container for the fully-wired Edge-Guardian backend."""
    configs: List[TransformerConfig]
    ai_engine: EdgeAIEngine
    health_calculator: HealthCalculator
    fleet: FleetManager
    simulator: WhatIfSimulator
    tools: AgentTools
    agent: EdgeGuardianAgent
    sink: TelemetrySink


def process_sample(
    config: TransformerConfig,
    telemetry: TelemetrySample,
    ai_engine: EdgeAIEngine,
    health_calculator: HealthCalculator,
    shadow: DigitalShadow,
    sink: Optional[TelemetrySink] = None,
) -> Tuple[PhysicsResult, FeatureVector, AIResult, HealthResult, RULResult]:
    """
    Process one telemetry sample through the full pipeline and update the shadow.

    This is the canonical orchestration function (plan §25). It performs no
    physics or ML itself — it only sequences the P1–P4 components and records
    the result on the asset's Digital Shadow.
    """
    if sink is not None:
        sink.send(telemetry)

    physics = calculate_physics(telemetry, config)
    features = extract_feature_vector(telemetry, physics)
    ai_result = ai_engine.infer(features)

    # Accumulate degradation across samples using the shadow's prior state.
    prev_degradation = shadow.state.degradation_score
    health = health_calculator.calculate_health(
        telemetry, physics, ai_result, previous_degradation=prev_degradation
    )
    rul = health_calculator.estimate_rul(health)

    shadow.update(telemetry, physics, ai_result, health, rul)
    return physics, features, ai_result, health, rul


def build_backend(
    asset_count: int = DEMO_ASSET_COUNT,
    seed: int = DEMO_SEED,
    train_normal_s: int = 60,
    train_fault_s: int = 30,
    sink: Optional[TelemetrySink] = None,
    api_key: Optional[str] = None,
    force_fallback: bool = False,
) -> Backend:
    """
    Construct and train the full Edge-Guardian backend.

    Steps:
        1. Generate a fleet of transformer configs.
        2. Train the Edge AI engine on a P1 training dataset.
        3. Create a Digital Shadow per asset and register it with the fleet.
        4. Wire the what-if simulator, agent tools, and agent.
    """
    sink = sink or LocalSink()

    logger.info("Building Edge-Guardian backend: %d assets (seed=%d)", asset_count, seed)
    configs = generate_fleet_configs(count=asset_count, seed=seed)

    # ── Train Edge AI on genuine P1 pipeline output ─────────────────────
    ai_engine = EdgeAIEngine()
    dataset = generate_training_dataset(
        configs, normal_duration_s=train_normal_s, fault_duration_s=train_fault_s, seed=seed
    )
    ai_engine.train(dataset)

    # ── Fleet of Digital Shadows ────────────────────────────────────────
    health_calculator = HealthCalculator()
    fleet = FleetManager()
    for config in configs:
        fleet.add_shadow(DigitalShadow(config))

    simulator = WhatIfSimulator(ai_engine, health_calculator)
    tools = AgentTools(fleet, simulator)
    agent = EdgeGuardianAgent(tools, api_key=api_key, force_fallback=force_fallback)

    return Backend(
        configs=configs,
        ai_engine=ai_engine,
        health_calculator=health_calculator,
        fleet=fleet,
        simulator=simulator,
        tools=tools,
        agent=agent,
        sink=sink,
    )


def run_operation(
    backend: Backend,
    config: TransformerConfig,
    steps: int,
    scenario: FaultScenario = FaultScenario.NORMAL,
    severity: Optional[float] = None,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Generate and process `steps` telemetry samples for one asset.

    Reuses the P1 telemetry generator (including scenario injection) and the
    canonical process_sample orchestration to update the asset's shadow.
    """
    shadow = backend.fleet.get_shadow(config.asset_id)
    if shadow is None:
        raise ValueError(f"No shadow registered for asset '{config.asset_id}'")

    samples = generate_telemetry(
        config=config,
        duration_seconds=steps,
        sample_interval_seconds=1,
        seed=seed,
        scenario=scenario,
        scenario_params=params,
        scenario_severity=severity,
    )
    for sample in samples:
        process_sample(
            config, sample, backend.ai_engine, backend.health_calculator, shadow, backend.sink
        )


# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

def _fleet_line(metrics: dict) -> str:
    return (
        f"Normal: {metrics['healthy_count']}  |  "
        f"Warning: {metrics['warning_count']}  |  "
        f"Critical: {metrics['critical_count']}  |  "
        f"Avg health: {metrics['average_health']:.1f}"
    )


def run_demo(
    asset_count: int = DEMO_ASSET_COUNT,
    target_asset: str = DEMO_TARGET_ASSET,
    seed: int = DEMO_SEED,
    run_agent: bool = True,
    force_fallback: bool = False,
) -> Backend:
    """
    Execute the required end-to-end demo and print a concise diagnostic.

    Returns the Backend so callers (and tests) can inspect final state.
    """
    print("\n" + "=" * 60)
    print("EDGE-GUARDIAN DEMO  (Stage 2 — software-only)")
    print("=" * 60)

    backend = build_backend(asset_count=asset_count, seed=seed, force_fallback=force_fallback)
    target_cfg = next((c for c in backend.configs if c.asset_id == target_asset), None)
    if target_cfg is None:
        raise ValueError(
            f"Target asset '{target_asset}' not in fleet of {asset_count}. "
            f"Increase --asset-count."
        )

    print(f"\nFleet: {len(backend.configs)} assets "
          f"({backend.configs[0].asset_id} – {backend.configs[-1].asset_id})")

    # ── 1. Normal operation for the whole fleet ─────────────────────────
    print("\nRunning normal operation...")
    for i, config in enumerate(backend.configs):
        run_operation(backend, config, steps=DEMO_NORMAL_STEPS,
                      scenario=FaultScenario.NORMAL, seed=seed + i)

    initial = backend.fleet.get_fleet_metrics()
    # Snapshot a COPY before the fault — the live state object is mutated in place.
    target_before = backend.fleet.get_shadow(target_asset).state.model_copy(deep=True)
    print("\nInitial fleet status:")
    print("  " + _fleet_line(initial))

    # ── 2. Inject fault into the target asset ───────────────────────────
    severity_desc = ", ".join(f"{k}={v}" for k, v in DEMO_FAULT_PARAMS.items())
    print(f"\nInjecting fault:")
    print(f"  Asset:    {target_asset}")
    print(f"  Scenario: {DEMO_FAULT_SCENARIO.value}")
    print(f"  Params:   {severity_desc} (severity={DEMO_FAULT_SEVERITY})")

    run_operation(
        backend, target_cfg, steps=DEMO_FAULT_STEPS,
        scenario=DEMO_FAULT_SCENARIO, severity=DEMO_FAULT_SEVERITY,
        params=DEMO_FAULT_PARAMS, seed=seed + 999,
    )

    # ── 3. Diagnostic (real computed values only) ───────────────────────
    target_after = backend.fleet.get_shadow(target_asset).state
    print(f"\nResult for {target_asset}:")
    print(f"  Anomaly score: {target_after.anomaly_score:.4f} "
          f"(is_anomaly={target_after.is_anomaly})")
    print(f"  Fault class:   {target_after.fault_class} "
          f"(confidence={target_after.fault_confidence})")
    print(f"  Efficiency:    {target_before.efficiency:.4f} -> {target_after.efficiency:.4f}")
    print(f"  Temperature:   {target_before.temperature:.1f}°C -> {target_after.temperature:.1f}°C")
    print(f"  Health:        {target_before.health_index:.1f} ({target_before.health_state.value})"
          f" -> {target_after.health_index:.1f} ({target_after.health_state.value})")
    print(f"  RUL:           {target_before.rul_hours:.0f} h -> {target_after.rul_hours:.0f} h")

    # ── 4. Fleet status after fault ─────────────────────────────────────
    final = backend.fleet.get_fleet_metrics()
    print("\nFleet status after fault:")
    print("  " + _fleet_line(final))

    # ── 5. Agent investigation (deterministic fallback if no API key) ───
    if run_agent:
        print("\n" + "-" * 60)
        print(f"ENGINEERING ASSISTANT  (backend: {backend.agent.backend})")
        print("-" * 60)

        q1 = f"Why is {target_asset} unhealthy?"
        print(f"\nQ: {q1}")
        print(backend.agent.ask(q1)["answer"])

        q2 = f"What if {target_asset} load increases another 20%?"
        print(f"\nQ: {q2}")
        print(backend.agent.ask(q2)["answer"])

        q3 = f"Generate a maintenance plan for {target_asset}."
        print(f"\nQ: {q3}")
        print(backend.agent.ask(q3)["answer"])

    print("\n" + "=" * 60)
    print(f"Telemetry samples routed to sink '{backend.sink.name}': "
          f"{getattr(backend.sink, 'total_sent', 'n/a')}")
    print("Demo complete. (No hardware, AWS, or GPT-5 API required.)")
    print("=" * 60 + "\n")
    return backend


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Edge-Guardian Stage 2 pipeline")
    parser.add_argument("--demo", action="store_true", help="Run the end-to-end demo.")
    parser.add_argument("--asset-count", type=int, default=DEMO_ASSET_COUNT,
                        help=f"Number of fleet assets (default {DEMO_ASSET_COUNT}).")
    parser.add_argument("--target", type=str, default=DEMO_TARGET_ASSET,
                        help=f"Asset to inject the fault into (default {DEMO_TARGET_ASSET}).")
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help="Random seed.")
    parser.add_argument("--no-agent", action="store_true", help="Skip the agent Q&A section.")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Force the deterministic agent backend even if an API key is set.")
    parser.add_argument("--ask", type=str, default=None,
                        help="Ask the agent one question after running the demo.")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        help="Logging level (default WARNING to keep demo output clean).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    if not args.demo and args.ask is None:
        parser.print_help()
        return 0

    backend = run_demo(
        asset_count=args.asset_count,
        target_asset=args.target,
        seed=args.seed,
        run_agent=not args.no_agent,
        force_fallback=args.force_fallback,
    )

    if args.ask:
        print(f"\nQ: {args.ask}")
        print(backend.agent.ask(args.ask)["answer"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
