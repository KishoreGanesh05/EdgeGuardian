"""
Edge-Guardian Synthetic Telemetry Generator.

Generates physically plausible telemetry for virtual transformers using
deterministic equations plus controlled noise. The generator follows
the pipeline:

    1. Generate load profile (time-varying)
    2. Convert load to current
    3. Calculate voltage behavior (with feeder drop)
    4. Apply scenario effects (resistance increase, overload, etc.)
    5. Calculate physics (power, losses, efficiency, temperature)
    6. Add measurement noise
    7. Package as TelemetrySample + PhysicsResult + FeatureVector

This module is the "virtual industrial data source" for Stage 2.
It replaces the physical CT/VT sensors that would exist in Stage 3.

NOTE: This is a physics-informed simulation for Stage 2 validation,
not a validated transformer simulation model.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.models import (
    FeatureVector,
    FaultScenario,
    PhysicsResult,
    TelemetrySample,
    TransformerConfig,
)
from data.scenarios import ScenarioEffect, apply_scenario
from physics.electrical import (
    calculate_apparent_power,
    calculate_effective_resistance,
    calculate_feeder_voltage_drop,
    calculate_input_power,
    calculate_output_power,
    calculate_power_factor,
    calculate_real_power,
    calculate_resistive_loss,
    calculate_rms,
)
from physics.efficiency import calculate_efficiency, calculate_efficiency_delta
from physics.thermal import (
    calculate_thermal_state,
    estimate_loss_induced_heating,
)
from config.settings import (
    MEASUREMENT_NOISE_PERCENT,
    THERMAL_RISE_COEFFICIENT,
    SAMPLE_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Industrial Data Source Abstraction
# ═══════════════════════════════════════════════════════════════════════════

class IndustrialDataSource:
    """
    Abstract interface for industrial data acquisition.

    Stage 2: SimulatedIndustrialDataSource (this module)
    Stage 3: A physical adapter connecting to real CT/VT, SCADA, or
             Emerson/NI gateway would implement this interface.
    """

    def get_telemetry(self, asset_id: str) -> Optional[TelemetrySample]:
        """Get the latest telemetry sample for an asset."""
        raise NotImplementedError

    def get_batch(
        self, asset_id: str, count: int
    ) -> List[TelemetrySample]:
        """Get a batch of telemetry samples."""
        raise NotImplementedError


class SimulatedIndustrialDataSource(IndustrialDataSource):
    """
    Stage 2 simulated data source.

    Generates physics-based synthetic telemetry that would come from
    real CT/VT sensors and temperature probes in a physical deployment.
    """

    def __init__(self, configs: Dict[str, TransformerConfig], seed: int = 42):
        self._configs = configs
        self._rng = np.random.default_rng(seed)
        self._sample_index: Dict[str, int] = {}

    def get_telemetry(self, asset_id: str) -> Optional[TelemetrySample]:
        """Generate one simulated telemetry sample."""
        config = self._configs.get(asset_id)
        if config is None:
            return None
        idx = self._sample_index.get(asset_id, 0)
        self._sample_index[asset_id] = idx + 1
        load_pct = _generate_load_profile_value(idx, self._rng)
        return _generate_single_sample(config, load_pct, self._rng)

    def get_batch(self, asset_id: str, count: int) -> List[TelemetrySample]:
        """Generate a batch of simulated samples."""
        return [self.get_telemetry(asset_id) for _ in range(count)]


# ═══════════════════════════════════════════════════════════════════════════
# Fleet Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_fleet_configs(
    count: int = 20,
    seed: int = 42,
    id_prefix: str = "T",
) -> List[TransformerConfig]:
    """
    Generate a fleet of transformer configurations with realistic variety.

    Each transformer gets slightly different parameters to simulate a
    real fleet with varying ages, sizes, and installation conditions.

    Args:
        count:     Number of transformers to generate.
        seed:      Random seed for reproducibility.
        id_prefix: Prefix for asset IDs (e.g. "T" → T001, T002, ...).

    Returns:
        List of TransformerConfig objects.
    """
    rng = np.random.default_rng(seed)
    configs = []

    for i in range(1, count + 1):
        asset_id = f"{id_prefix}{i:03d}"

        # Vary parameters within realistic bounds
        rated_power = rng.choice([250.0, 500.0, 750.0, 1000.0], p=[0.2, 0.4, 0.25, 0.15])
        efficiency = rng.uniform(0.955, 0.985)
        resistance = rng.uniform(0.10, 0.25)
        age_factor = rng.uniform(0.8, 2.5)
        ambient = rng.uniform(28.0, 42.0)
        initial_health = rng.uniform(75.0, 100.0)

        configs.append(TransformerConfig(
            asset_id=asset_id,
            rated_power_kva=float(rated_power),
            nominal_voltage_v=11000.0,
            nominal_frequency_hz=50.0,
            baseline_efficiency=round(float(efficiency), 4),
            base_resistance_ohm=round(float(resistance), 4),
            thermal_coefficient=0.004,
            ambient_temperature_c=round(float(ambient), 1),
            initial_health=round(float(initial_health), 1),
            age_factor=round(float(age_factor), 2),
        ))

    logger.info(f"Generated fleet of {count} transformers ({configs[0].asset_id} – {configs[-1].asset_id})")
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_telemetry(
    config: TransformerConfig,
    duration_seconds: int = 300,
    sample_interval_seconds: int = SAMPLE_INTERVAL_SECONDS,
    seed: Optional[int] = None,
    scenario: FaultScenario = FaultScenario.NORMAL,
    scenario_params: Optional[Dict[str, float]] = None,
    scenario_severity: Optional[float] = None,
) -> List[TelemetrySample]:
    """
    Generate a sequence of physically plausible telemetry samples.

    The generator follows this pipeline for each sample:
        1. Load profile → current
        2. Apply scenario effects (if any)
        3. Calculate voltages with feeder drop
        4. Add measurement noise
        5. Package as TelemetrySample

    Args:
        config:                  Transformer configuration.
        duration_seconds:        Total simulation duration.
        sample_interval_seconds: Time between samples.
        seed:                    Random seed for deterministic output.
        scenario:                Fault scenario to inject.
        scenario_params:         Scenario-specific parameters.
        scenario_severity:       Severity factor (0 to 1).

    Returns:
        List of TelemetrySample objects.
    """
    rng = np.random.default_rng(seed)
    num_samples = max(1, duration_seconds // sample_interval_seconds)

    # Get scenario effects
    effect = apply_scenario(scenario, config, scenario_severity, scenario_params)

    samples = []
    base_time = datetime.now(timezone.utc)

    for i in range(num_samples):
        timestamp = base_time + timedelta(seconds=i * sample_interval_seconds)

        # 1. Generate load profile (40-90% with slow variation + noise)
        load_pct = _generate_load_profile_value(i, rng)

        # 2. Apply scenario load adjustment
        load_pct += effect.load_adjustment_percent
        load_pct = max(5.0, min(200.0, load_pct))

        # 3. Generate the sample with physics
        sample = _generate_sample_with_physics(
            config=config,
            timestamp=timestamp,
            load_pct=load_pct,
            effect=effect,
            rng=rng,
        )
        samples.append(sample)

    logger.info(
        f"Generated {len(samples)} telemetry samples for {config.asset_id} "
        f"[scenario={effect.scenario_id}, severity={effect.severity}]"
    )
    return samples


def _generate_load_profile_value(sample_index: int, rng: np.random.Generator) -> float:
    """
    Generate a realistic load value for a given time step.

    Uses a slow sinusoidal base (diurnal pattern) plus smaller
    random variation to simulate real industrial loading.

    Returns:
        Load as percentage of rated capacity (typically 40-90%).
    """
    # Base load: slow sinusoidal variation (simulates daily cycle)
    base_load = 65.0 + 20.0 * math.sin(2 * math.pi * sample_index / 3600)

    # Random variation: ±5%
    noise = float(rng.normal(0, 2.5))

    return max(10.0, min(100.0, base_load + noise))


def _generate_sample_with_physics(
    config: TransformerConfig,
    timestamp: datetime,
    load_pct: float,
    effect: ScenarioEffect,
    rng: np.random.Generator,
) -> TelemetrySample:
    """
    Generate a single telemetry sample with full physics.

    This is where the physical model turns load + scenario into
    physically consistent voltage, current, temperature, and power readings.
    """
    load_fraction = load_pct / 100.0

    # ── Current from load ──────────────────────────────────────────────
    # I_rated = S_rated / V_nominal (single-phase equivalent)
    rated_current = (config.rated_power_kva * 1000.0) / config.nominal_voltage_v
    feeder_current = rated_current * load_fraction

    # ── Resistance with scenario adjustment ────────────────────────────
    effective_resistance = config.base_resistance_ohm * effect.resistance_multiplier

    # ── Voltage calculations ───────────────────────────────────────────
    # Feeder voltage with drop
    voltage_drop = calculate_feeder_voltage_drop(feeder_current, effective_resistance)
    feeder_voltage = config.nominal_voltage_v  # Source voltage is constant

    # Secondary voltage accounts for transformer ratio and feeder losses
    # Simplified: secondary voltage slightly reduced by feeder losses
    turns_ratio = 0.038  # ~11kV to ~415V (typical distribution)
    secondary_voltage = (feeder_voltage - voltage_drop) * turns_ratio

    # Secondary current (power balance approximation)
    if secondary_voltage > 0:
        secondary_current = (feeder_current * feeder_voltage * config.baseline_efficiency
                            * effect.efficiency_multiplier) / secondary_voltage
    else:
        secondary_current = 0.0

    # ── Power factor ───────────────────────────────────────────────────
    # Normal industrial PF: 0.85-0.95 with slight variation
    base_pf = 0.90 + 0.05 * math.sin(2 * math.pi * load_fraction)
    pf = max(0.70, min(0.99, base_pf + float(rng.normal(0, 0.01))))

    # ── Losses ─────────────────────────────────────────────────────────
    feeder_loss = calculate_resistive_loss(feeder_current, effective_resistance)

    # ── Temperature ────────────────────────────────────────────────────
    # Ambient with scenario adjustment
    ambient_temp = config.ambient_temperature_c + effect.ambient_temp_adjustment_c

    # Additional heating from excess losses (above normal baseline)
    normal_loss = calculate_resistive_loss(feeder_current, config.base_resistance_ohm)
    excess_loss = max(0.0, feeder_loss - normal_loss)
    extra_heating = estimate_loss_induced_heating(excess_loss, thermal_resistance=0.01)

    thermal = calculate_thermal_state(
        ambient_temperature_c=ambient_temp,
        load_fraction=load_fraction,
        thermal_rise_coefficient=THERMAL_RISE_COEFFICIENT,
        additional_loss_heating_c=extra_heating,
    )

    transformer_temp = thermal["temperature_c"]

    # ── Add measurement noise ──────────────────────────────────────────
    noise_factor = MEASUREMENT_NOISE_PERCENT / 100.0
    feeder_voltage = _add_noise(feeder_voltage, noise_factor, rng)
    feeder_current = _add_noise(feeder_current, noise_factor, rng)
    secondary_voltage = _add_noise(secondary_voltage, noise_factor, rng)
    secondary_current = _add_noise(secondary_current, noise_factor, rng)
    transformer_temp = _add_noise(transformer_temp, noise_factor * 0.5, rng)

    # ── Package as TelemetrySample ─────────────────────────────────────
    return TelemetrySample(
        timestamp=timestamp,
        asset_id=config.asset_id,
        voltage_feeder_v=round(feeder_voltage, 2),
        current_feeder_a=round(max(0.0, feeder_current), 4),
        voltage_secondary_v=round(max(0.0, secondary_voltage), 2),
        current_secondary_a=round(max(0.0, secondary_current), 4),
        frequency_hz=50.0 + float(rng.normal(0, 0.02)),
        ambient_temperature_c=round(ambient_temp, 2),
        transformer_temperature_c=round(transformer_temp, 2),
        load_percent=round(load_pct, 2),
        scenario_id=effect.scenario_id if effect.scenario_id != "NORMAL" else None,
        scenario_severity=effect.severity if effect.severity > 0 else None,
    )


def _generate_single_sample(
    config: TransformerConfig,
    load_pct: float,
    rng: np.random.Generator,
) -> TelemetrySample:
    """Generate a single sample under normal conditions."""
    effect = ScenarioEffect()
    return _generate_sample_with_physics(
        config=config,
        timestamp=datetime.now(timezone.utc),
        load_pct=load_pct,
        effect=effect,
        rng=rng,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Physics Pipeline — Processes raw telemetry into PhysicsResult
# ═══════════════════════════════════════════════════════════════════════════

def calculate_physics(
    sample: TelemetrySample,
    config: TransformerConfig,
) -> PhysicsResult:
    """
    Run the complete physics pipeline on a telemetry sample.

    This is the central function that converts raw telemetry into
    physics-derived engineering quantities. The Edge AI team should
    NOT bypass this — raw telemetry goes to physics, physics outputs
    go to feature extraction.

    Pipeline:
        TelemetrySample → PhysicsResult
            - RMS voltage / current
            - Real power, apparent power, power factor
            - Input power, output power
            - Losses
            - Efficiency + efficiency delta
            - Thermal stress

    Args:
        sample: Raw telemetry from the transformer.
        config: Transformer configuration (for baseline values).

    Returns:
        PhysicsResult with all derived engineering quantities.
    """
    # ── RMS values (for single-sample, RMS ≈ the instantaneous value) ──
    rms_v = abs(sample.voltage_feeder_v)
    rms_i = abs(sample.current_feeder_a)

    # ── Power calculations ─────────────────────────────────────────────
    apparent = calculate_apparent_power(rms_v, rms_i)

    # Estimate power factor from feeder-side and secondary-side power
    input_power = calculate_apparent_power(
        sample.voltage_feeder_v, sample.current_feeder_a
    )
    output_power = calculate_apparent_power(
        sample.voltage_secondary_v, sample.current_secondary_a
    )

    # Power factor from the real/apparent relationship
    # For simulation: derive from secondary power balance
    if apparent > 0 and output_power > 0:
        pf = min(1.0, output_power / input_power) if input_power > 0 else 0.95
        pf = max(0.0, min(1.0, pf))
    else:
        pf = 0.95  # Default industrial PF

    real_power = calculate_real_power(rms_v, rms_i, pf)

    # ── Input / Output power ───────────────────────────────────────────
    p_in = input_power  # Using apparent as approximation for simulation
    p_out = output_power

    # ── Losses ─────────────────────────────────────────────────────────
    loss = max(0.0, p_in - p_out)

    # ── Efficiency ─────────────────────────────────────────────────────
    efficiency = calculate_efficiency(p_in, p_out)
    eff_delta = calculate_efficiency_delta(efficiency, config.baseline_efficiency)

    # ── Thermal stress ─────────────────────────────────────────────────
    from physics.thermal import calculate_thermal_stress
    thermal_stress = calculate_thermal_stress(
        transformer_temperature_c=sample.transformer_temperature_c,
        ambient_temperature_c=sample.ambient_temperature_c,
    )

    return PhysicsResult(
        rms_v=round(rms_v, 2),
        rms_i=round(rms_i, 4),
        power_w=round(real_power, 2),
        apparent_power_va=round(apparent, 2),
        power_factor=round(pf, 4),
        input_power_w=round(p_in, 2),
        output_power_w=round(p_out, 2),
        loss_w=round(loss, 2),
        efficiency=round(efficiency, 6),
        eff_delta=round(eff_delta, 6),
        thermal_stress=round(thermal_stress, 4),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction Interface
# ═══════════════════════════════════════════════════════════════════════════

def extract_feature_vector(
    sample: TelemetrySample,
    physics: PhysicsResult,
) -> FeatureVector:
    """
    Extract the seven-value feature vector from telemetry + physics.

    This is the INTERFACE for the Edge AI team. They should call this
    function to get the physics-derived features, NOT read raw telemetry.

    Feature order (contractually fixed):
        [RMS_V, RMS_I, Power, Power_Factor, Efficiency, Temperature, Eff_Delta]

    Args:
        sample:  Raw telemetry sample.
        physics: Physics calculation results.

    Returns:
        FeatureVector ready for ML preprocessing and inference.
    """
    return FeatureVector(
        rms_v=physics.rms_v,
        rms_i=physics.rms_i,
        power=physics.power_w,
        power_factor=physics.power_factor,
        efficiency=physics.efficiency,
        temperature=sample.transformer_temperature_c,
        eff_delta=physics.eff_delta,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Generation (for training Edge AI models)
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_dataset(
    configs: List[TransformerConfig],
    normal_duration_s: int = 300,
    fault_duration_s: int = 120,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Generate a labeled training dataset for Edge AI model training.

    Produces:
        - Normal samples from all transformers (for autoencoder training)
        - Fault samples for each scenario (for classifier training)

    Each record contains the telemetry, physics result, feature vector,
    and the scenario label.

    Args:
        configs:           List of transformer configurations.
        normal_duration_s: Duration of normal operation per transformer.
        fault_duration_s:  Duration of each fault scenario per transformer.
        seed:              Random seed.

    Returns:
        Dict with keys:
            "normal":  List of normal-operation records
            "faults":  List of fault-scenario records (labeled)
    """
    rng_seed = seed
    dataset = {"normal": [], "faults": []}

    # ── Normal operation from all transformers ─────────────────────────
    for config in configs:
        samples = generate_telemetry(
            config=config,
            duration_seconds=normal_duration_s,
            seed=rng_seed,
            scenario=FaultScenario.NORMAL,
        )
        for sample in samples:
            physics = calculate_physics(sample, config)
            features = extract_feature_vector(sample, physics)
            dataset["normal"].append({
                "asset_id": config.asset_id,
                "sample": sample,
                "physics": physics,
                "features": features,
                "label": "NORMAL",
            })
        rng_seed += 1

    # ── Fault scenarios from a subset of transformers ──────────────────
    fault_scenarios = [
        FaultScenario.FEEDER_RESISTANCE_INCREASE,
        FaultScenario.OVERLOAD,
        FaultScenario.THERMAL_STRESS,
        FaultScenario.EFFICIENCY_DEGRADATION,
    ]

    for scenario in fault_scenarios:
        # Use a subset of transformers for fault training
        fault_configs = configs[:max(3, len(configs) // 4)]
        for config in fault_configs:
            samples = generate_telemetry(
                config=config,
                duration_seconds=fault_duration_s,
                seed=rng_seed,
                scenario=scenario,
            )
            for sample in samples:
                physics = calculate_physics(sample, config)
                features = extract_feature_vector(sample, physics)
                dataset["faults"].append({
                    "asset_id": config.asset_id,
                    "sample": sample,
                    "physics": physics,
                    "features": features,
                    "label": scenario.value,
                })
            rng_seed += 1

    logger.info(
        f"Generated training dataset: {len(dataset['normal'])} normal, "
        f"{len(dataset['faults'])} fault samples"
    )
    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_noise(value: float, noise_fraction: float, rng: np.random.Generator) -> float:
    """Add Gaussian measurement noise as a fraction of the value."""
    if abs(value) < 1e-10:
        return value
    noise = float(rng.normal(0, abs(value) * noise_fraction))
    return value + noise
