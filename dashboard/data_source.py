"""
Edge-Guardian Dashboard Data Source.

Provides mock initialized fleet and simulator for the Streamlit dashboard demo.
"""

import streamlit as st
import numpy as np

from data.models import FaultScenario
from data.generator import (
    generate_fleet_configs,
    _generate_single_sample,
    _generate_sample_with_physics,
    calculate_physics,
    extract_feature_vector,
    generate_training_dataset,
)
from digital_shadow.shadow import DigitalShadow
from digital_shadow.fleet import FleetManager
from edge_ai.features import EdgeAIEngine
from health.calculator import HealthCalculator
from simulator.what_if import WhatIfSimulator
from data.scenarios import apply_scenario


@st.cache_resource
def initialize_demo_backend():
    """Initialize the fleet and Edge AI engine using genuine pipeline outputs."""

    # 1. Initialize logic units
    ai_engine = EdgeAIEngine()
    health_calculator = HealthCalculator()
    simulator = WhatIfSimulator(ai_engine, health_calculator)
    fleet = FleetManager()

    # 2. Generate 25 transformer assets
    configs = generate_fleet_configs(count=25, seed=42)
    rng = np.random.default_rng(42)

    # 3. Train the AI model on generated P1 training data
    dataset = generate_training_dataset(
        configs,
        normal_duration_s=60,
        fault_duration_s=30,
        seed=42,
    )
    ai_engine.train(dataset)

    # 4. Populate FleetManager with Digital Shadows
    for i, config in enumerate(configs):
        shadow = DigitalShadow(config)

        # Historical samples for dashboard trends
        for step in range(50):
            # Vary load over time
            load = (
                50.0
                + 30.0 * np.sin(step / 10.0)
                + float(rng.normal(0, 2))
            )

            # Inject historical fault scenarios
            scenario = FaultScenario.NORMAL
            severity = 0.0

            if step > 30:
                if i == 0:
                    scenario = FaultScenario.OVERLOAD
                    severity = 1.0 + ((step - 30) / 20.0)

                elif i == 1:
                    scenario = FaultScenario.THERMAL_STRESS
                    severity = 1.0

                elif i == 22:
                    # T023: validated Stage-2 OVERLOAD demonstration asset
                    scenario = FaultScenario.OVERLOAD
                    severity = 1.0

            effect = apply_scenario(
                scenario,
                config,
                severity,
            )

            load = max(
                5.0,
                min(
                    200.0,
                    load + effect.load_adjustment_percent,
                ),
            )

            # P1: Generate telemetry + physics
            sample = _generate_single_sample(
                config,
                load,
                rng,
            )

            sample = _generate_sample_with_physics(
                config=config,
                timestamp=sample.timestamp,
                load_pct=load,
                effect=effect,
                rng=rng,
            )

            physics = calculate_physics(
                sample,
                config,
            )

            features = extract_feature_vector(
                sample,
                physics,
            )

            # P2: Edge AI inference
            ai = ai_engine.infer(features)

            # P3: Health + RUL
            previous_degradation = shadow.state.degradation_score

            health = health_calculator.calculate_health(
                sample,
                physics,
                ai,
                previous_degradation=previous_degradation,
            )

            rul = health_calculator.estimate_rul(
                health,
            )

            # Update Digital Shadow
            shadow.update(
                sample,
                physics,
                ai,
                health,
                rul,
            )

        # ------------------------------------------------------------
        # T023 LIVE DEMO STATE
        # ------------------------------------------------------------
        #
        # Keep the historical trend above, then explicitly make the
        # CURRENT T023 state the validated Stage-2 OVERLOAD case.
        #
        # This is intentionally dashboard/demo-layer behavior only.
        # P1, P2 and P3 remain unchanged.
        # ------------------------------------------------------------

        if config.asset_id == "T023":

            demo_scenario = FaultScenario.OVERLOAD
            demo_severity = 1.0

            demo_effect = apply_scenario(
                demo_scenario,
                config,
                demo_severity,
            )

            # High-load demonstration point.
            demo_load = 123.0

            # P1: Generate final T023 telemetry
            demo_sample = _generate_sample_with_physics(
                config=config,
                timestamp=sample.timestamp,
                load_pct=demo_load,
                effect=demo_effect,
                rng=rng,
            )

            # P1: Physics
            demo_physics = calculate_physics(
                demo_sample,
                config,
            )

            # P1: Physics-derived feature vector
            demo_features = extract_feature_vector(
                demo_sample,
                demo_physics,
            )

            # P2: Edge AI
            demo_ai = ai_engine.infer(
                demo_features,
            )

            # P3: Health
            previous_degradation = shadow.state.degradation_score

            demo_health = health_calculator.calculate_health(
                demo_sample,
                demo_physics,
                demo_ai,
                previous_degradation=previous_degradation,
            )

            # P3: RUL
            demo_rul = health_calculator.estimate_rul(
                demo_health,
            )

            # Final T023 Digital Shadow state
            shadow.update(
                demo_sample,
                demo_physics,
                demo_ai,
                demo_health,
                demo_rul,
            )

        fleet.add_shadow(shadow)

    return fleet, simulator


def get_fleet_manager() -> FleetManager:
    fleet, _ = initialize_demo_backend()
    return fleet


def get_simulator() -> WhatIfSimulator:
    _, simulator = initialize_demo_backend()
    return simulator


def get_available_assets() -> list[str]:
    fleet = get_fleet_manager()
    return list(fleet._shadows.keys())