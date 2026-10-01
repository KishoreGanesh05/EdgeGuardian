"""
Edge-Guardian What-If Simulator (P4).
"""

from datetime import datetime, timezone
import numpy as np

from data.models import FaultScenario, ScenarioResult, DigitalShadowState
from data.scenarios import apply_scenario
from data.generator import _generate_sample_with_physics, calculate_physics, extract_feature_vector
from digital_shadow.shadow import DigitalShadow
from edge_ai.features import EdgeAIEngine
from health.calculator import HealthCalculator

class WhatIfSimulator:
    """
    Simulates hypothetical scenarios on transformer assets without mutating
    the live Digital Shadow.
    """

    def __init__(self, ai_engine: EdgeAIEngine, health_calculator: HealthCalculator):
        self.ai_engine = ai_engine
        self.health_calculator = health_calculator

    def run_scenario(
        self,
        shadow: DigitalShadow,
        scenario: FaultScenario,
        severity: float = 1.0,
        parameters: dict = None
    ) -> ScenarioResult:
        """
        Run a what-if scenario on the given Digital Shadow.

        Args:
            shadow: The source DigitalShadow (will not be mutated).
            scenario: The fault scenario to apply.
            severity: Severity multiplier for the fault.
            parameters: Scenario-specific parameter overrides.

        Returns:
            ScenarioResult containing current and projected states and deltas.
        """
        params = parameters or {}
        rng = np.random.default_rng(42) # Deterministic for simulation predictability

        # 1. Snapshot current state for comparison
        current_state = shadow.state.model_dump()

        # 2. Apply scenario to get the physics effect
        effect = apply_scenario(scenario, shadow.state.config, severity, params)

        # 3. Calculate adjusted load based on current state + scenario effect
        load_pct = max(5.0, min(200.0, shadow.state.load_percent + effect.load_adjustment_percent))

        # 4. Generate new telemetry and physics using existing pipeline
        sample = _generate_sample_with_physics(
            config=shadow.state.config,
            timestamp=datetime.now(timezone.utc),
            load_pct=load_pct,
            effect=effect,
            rng=rng,
        )

        physics = calculate_physics(sample, shadow.state.config)
        features = extract_feature_vector(sample, physics)

        # 5. Run AI Inference
        ai_result = self.ai_engine.infer(features)

        # 6. Calculate Health & RUL
        # For simulation, we use the current degradation score as the history penalty
        health_result = self.health_calculator.calculate_health(
            sample, physics, ai_result, previous_degradation=shadow.state.degradation_score
        )
        rul_result = self.health_calculator.estimate_rul(health_result)

        # 7. Construct projected state dict
        projected_state = current_state.copy()

        # Update projected metrics
        projected_state["load_percent"] = sample.load_percent
        projected_state["temperature"] = sample.transformer_temperature_c
        projected_state["efficiency"] = physics.efficiency
        projected_state["power_factor"] = physics.power_factor
        projected_state["health_index"] = health_result.health_index
        projected_state["health_state"] = health_result.health_state.value
        projected_state["anomaly_score"] = ai_result.anomaly_score
        projected_state["is_anomaly"] = ai_result.is_anomaly
        projected_state["fault_class"] = ai_result.fault_class
        projected_state["fault_confidence"] = ai_result.fault_confidence
        projected_state["rul_hours"] = rul_result.rul_hours
        projected_state["rms_v"] = physics.rms_v
        projected_state["rms_i"] = physics.rms_i
        projected_state["power_w"] = physics.power_w
        projected_state["loss_w"] = physics.loss_w
        projected_state["eff_delta"] = physics.eff_delta
        projected_state["thermal_stress"] = physics.thermal_stress

        # 8. Calculate deltas
        deltas = {
            "health_index": projected_state["health_index"] - current_state["health_index"],
            "rul_hours": projected_state["rul_hours"] - current_state["rul_hours"],
            "anomaly_score": projected_state["anomaly_score"] - current_state["anomaly_score"],
            "efficiency": projected_state["efficiency"] - current_state["efficiency"],
            "temperature": projected_state["temperature"] - current_state["temperature"],
        }

        # 9. Warnings
        warnings = []
        if health_result.health_index < 40.0:
            warnings.append("CRITICAL: Health Index below 40.0")
        if ai_result.is_anomaly:
            warnings.append(f"ANOMALY: {ai_result.fault_class or 'Unknown'} detected")

        return ScenarioResult(
            asset_id=shadow.state.asset_id,
            scenario=scenario.value,
            parameters={"severity": severity, **params},
            current_state=current_state,
            projected_state=projected_state,
            deltas=deltas,
            warnings=warnings
        )
