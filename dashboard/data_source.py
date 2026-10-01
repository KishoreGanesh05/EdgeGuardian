"""
Edge-Guardian Dashboard Data Source.

Provides mock initialized fleet and simulator for the Streamlit dashboard demo.
"""

import streamlit as st
import numpy as np

from data.models import HealthState, FaultScenario
from data.generator import generate_fleet_configs, _generate_single_sample, calculate_physics, extract_feature_vector, generate_training_dataset
from digital_shadow.shadow import DigitalShadow
from digital_shadow.fleet import FleetManager
from edge_ai.features import EdgeAIEngine
from health.calculator import HealthCalculator
from simulator.what_if import WhatIfSimulator
from data.scenarios import apply_scenario

@st.cache_resource
def initialize_demo_backend():
    """Initializes the fleet and Edge AI engine using genuine pipeline outputs."""
    # 1. Initialize logic units
    ai_engine = EdgeAIEngine()
    health_calculator = HealthCalculator()
    simulator = WhatIfSimulator(ai_engine, health_calculator)
    fleet = FleetManager()

    # 2. Generate a fleet of 20 configs
    configs = generate_fleet_configs(count=20, seed=42)
    rng = np.random.default_rng(42)

    # 3. Train the AI model on a generated dataset (Genuine P1 -> P2 training)
    # Using small durations for fast startup
    dataset = generate_training_dataset(configs, normal_duration_s=60, fault_duration_s=30, seed=42)
    ai_engine.train(dataset)

    # 4. Populate FleetManager with DigitalShadows
    for i, config in enumerate(configs):
        shadow = DigitalShadow(config)

        # We simulate some historical steps to populate the graphs
        for step in range(50):
            # Vary load over time
            load = 50.0 + 30.0 * np.sin(step / 10.0) + float(rng.normal(0, 2))

            # Inject a real physical fault for the first couple of assets late in history
            scenario = FaultScenario.NORMAL
            severity = 0.0
            if step > 30:
                if i == 0:
                    scenario = FaultScenario.OVERLOAD
                    severity = 1.0 + ((step - 30) / 20.0) # progressive overload
                elif i == 1:
                    scenario = FaultScenario.THERMAL_STRESS
                    severity = 1.0

            effect = apply_scenario(scenario, config, severity)
            load = max(5.0, min(200.0, load + effect.load_adjustment_percent))

            # P1: Generate Telemetry and Physics
            sample = _generate_single_sample(config, load, rng)

            # If there's an effect, we should ideally use _generate_sample_with_physics, but _generate_single_sample
            # hardcodes NORMAL. Let's use the proper function:
            from data.generator import _generate_sample_with_physics
            from datetime import datetime, timezone

            sample = _generate_sample_with_physics(
                config=config,
                timestamp=datetime.now(timezone.utc),
                load_pct=load,
                effect=effect,
                rng=rng
            )

            physics = calculate_physics(sample, config)
            features = extract_feature_vector(sample, physics)

            # P2: AI Inference
            ai = ai_engine.infer(features)

            # P3: Health & RUL
            # Retrieve previous degradation to accurately step health state
            prev_deg = shadow.state.degradation_score
            health = health_calculator.calculate_health(sample, physics, ai, previous_degradation=prev_deg)
            rul = health_calculator.estimate_rul(health)

            # Update Shadow
            shadow.update(sample, physics, ai, health, rul)

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
