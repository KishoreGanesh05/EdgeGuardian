"""
Edge-Guardian Data Models.

Strongly-typed Pydantic models defining the data contracts between
all system modules. These models enforce type safety and validation
throughout the Edge-Guardian pipeline.

NOTE: This is a physics-informed simulation model for Stage 2 validation,
not a validated transformer digital twin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pydantic import ConfigDict
from enum import Enum
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


# ── Enumerations ───────────────────────────────────────────────────────────

class HealthState(str, Enum):
    """Asset health classification states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class FaultScenario(str, Enum):
    """Supported fault injection scenarios."""
    NORMAL = "NORMAL"
    FEEDER_RESISTANCE_INCREASE = "FEEDER_RESISTANCE_INCREASE"
    OVERLOAD = "OVERLOAD"
    THERMAL_STRESS = "THERMAL_STRESS"
    EFFICIENCY_DEGRADATION = "EFFICIENCY_DEGRADATION"


# ── Transformer Configuration ─────────────────────────────────────────────

class TransformerConfig(BaseModel):
    """
    Configuration for a virtual transformer asset.

    Defines the physical and operational parameters for a single
    transformer unit in the Edge-Guardian simulation.
    """
    asset_id: str = Field(..., description="Unique asset identifier (e.g. T001)")
    rated_power_kva: float = Field(500.0, gt=0, description="Rated power in kVA")
    nominal_voltage_v: float = Field(11000.0, gt=0, description="Nominal voltage in volts")
    nominal_frequency_hz: float = Field(50.0, gt=0, description="Nominal frequency in Hz")
    baseline_efficiency: float = Field(0.97, ge=0.5, le=1.0, description="Baseline efficiency (0-1)")
    base_resistance_ohm: float = Field(0.15, ge=0, description="Base feeder resistance in ohms")
    thermal_coefficient: float = Field(0.004, ge=0, description="Thermal coefficient for winding resistance")
    ambient_temperature_c: float = Field(35.0, description="Ambient temperature in °C")
    initial_health: float = Field(95.0, ge=0, le=100, description="Initial health index (0-100)")
    age_factor: float = Field(1.0, ge=0.1, le=5.0, description="Age degradation multiplier")


# ── Telemetry ──────────────────────────────────────────────────────────────

class TelemetrySample(BaseModel):
    """
    A single telemetry measurement from a transformer.

    Represents raw (simulated) electrical and thermal readings
    as would be captured by CT/VT sensors and temperature probes.
    """
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    asset_id: str
    voltage_feeder_v: float = Field(..., description="Feeder-side RMS voltage (V)")
    current_feeder_a: float = Field(..., description="Feeder-side RMS current (A)")
    voltage_secondary_v: float = Field(..., description="Secondary-side RMS voltage (V)")
    current_secondary_a: float = Field(..., description="Secondary-side RMS current (A)")
    frequency_hz: float = Field(50.0, gt=0, description="System frequency (Hz)")
    ambient_temperature_c: float = Field(..., description="Ambient temperature (°C)")
    transformer_temperature_c: float = Field(..., description="Transformer winding temperature (°C)")
    load_percent: float = Field(..., ge=0, description="Load as percentage of rated capacity")
    # Optional scenario metadata
    scenario_id: Optional[str] = Field(None, description="Active fault scenario if any")
    scenario_severity: Optional[float] = Field(None, description="Fault severity parameter")


# ── Physics Results ────────────────────────────────────────────────────────

class PhysicsResult(BaseModel):
    """
    Results from the physics engine calculations.

    All values are deterministically derived from telemetry
    using explicit engineering equations.
    """
    rms_v: float = Field(..., description="RMS voltage (V)")
    rms_i: float = Field(..., description="RMS current (A)")
    power_w: float = Field(..., description="Real power (W)")
    apparent_power_va: float = Field(..., description="Apparent power (VA)")
    power_factor: float = Field(..., ge=0, le=1.0, description="Power factor (0-1)")
    input_power_w: float = Field(..., description="Input power at feeder (W)")
    output_power_w: float = Field(..., description="Output power at secondary (W)")
    loss_w: float = Field(..., ge=0, description="Total losses (W)")
    efficiency: float = Field(..., ge=0, le=1.0, description="Calculated efficiency (0-1)")
    eff_delta: float = Field(..., description="Efficiency delta from baseline")
    thermal_stress: float = Field(..., ge=0, description="Thermal stress factor (0+)")


# ── Feature Vector ─────────────────────────────────────────────────────────

class FeatureVector(BaseModel):
    """
    The seven-value physics-derived feature vector for Edge AI.

    This is the interface between the physics engine and the ML models.
    The order is contractually fixed and must not be silently changed.

    Order: [RMS_V, RMS_I, Power, Power_Factor, Efficiency, Temperature, Eff_Delta]
    """
    rms_v: float
    rms_i: float
    power: float
    power_factor: float
    efficiency: float
    temperature: float
    eff_delta: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in the canonical feature order."""
        return np.array([
            self.rms_v,
            self.rms_i,
            self.power,
            self.power_factor,
            self.efficiency,
            self.temperature,
            self.eff_delta,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "FeatureVector":
        """Create FeatureVector from numpy array in canonical order."""
        if len(arr) != 7:
            raise ValueError(f"Expected 7 features, got {len(arr)}")
        return cls(
            rms_v=float(arr[0]),
            rms_i=float(arr[1]),
            power=float(arr[2]),
            power_factor=float(arr[3]),
            efficiency=float(arr[4]),
            temperature=float(arr[5]),
            eff_delta=float(arr[6]),
        )


# ── AI Results ─────────────────────────────────────────────────────────────

class AIResult(BaseModel):
    """
    Combined result from Edge AI anomaly detection and fault classification.
    """
    anomaly_score: float = Field(..., ge=0, description="Reconstruction error (MSE)")
    is_anomaly: bool = Field(..., description="Whether score exceeds threshold")
    fault_class: Optional[str] = Field(None, description="Classified fault type or UNKNOWN")
    fault_confidence: Optional[float] = Field(None, ge=0, le=1.0, description="Classification confidence")
    model_version: str = Field("v1.0-stage2", description="Model version identifier")


# ── Health Results ─────────────────────────────────────────────────────────

class HealthResult(BaseModel):
    """
    Asset health assessment result.

    The Health Index is a project-defined composite score (0-100),
    NOT an industry-standard or certified metric.
    """
    health_index: float = Field(..., ge=0, le=100, description="Health index (0-100)")
    health_state: HealthState = Field(..., description="Categorical health state")
    degradation_score: float = Field(..., ge=0, le=1.0, description="Normalized degradation (0-1)")


# ── RUL Results ────────────────────────────────────────────────────────────

class RULResult(BaseModel):
    """
    Remaining Useful Life estimation.

    NOTE: This is a model-based/synthetic estimate for Stage 2,
    NOT validated against real-world failure data.
    """
    rul_hours: float = Field(..., ge=0, description="Estimated remaining useful life (hours)")
    critical_threshold: float = Field(..., description="Health threshold for critical state")
    current_degradation: float = Field(..., ge=0, le=1.0, description="Current degradation level (0-1)")
    model_based: bool = Field(True, description="Always True for Stage 2 — synthetic model")


# ── Digital Shadow ─────────────────────────────────────────────────────────

class DigitalShadowState(BaseModel):
    """
    The authoritative current-state object for a single transformer asset.

    The dashboard and agent should read this rather than independently
    recomputing state. History is bounded to prevent unbounded memory growth.
    """
    asset_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: TransformerConfig
    # Current operational state
    load_percent: float = 0.0
    temperature: float = 35.0
    efficiency: float = 0.97
    power_factor: float = 0.95
    # Health & AI state
    health_index: float = 95.0
    health_state: HealthState = HealthState.NORMAL
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    fault_class: Optional[str] = None
    fault_confidence: Optional[float] = None
    rul_hours: float = 200000.0
    degradation_score: float = 0.0
    # Physics
    rms_v: float = 0.0
    rms_i: float = 0.0
    power_w: float = 0.0
    loss_w: float = 0.0
    eff_delta: float = 0.0
    thermal_stress: float = 0.0
    # History (bounded)
    health_history: List[float] = Field(default_factory=list)
    anomaly_history: List[float] = Field(default_factory=list)
    efficiency_history: List[float] = Field(default_factory=list)
    temperature_history: List[float] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Scenario Simulation ───────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    """Request for a what-if scenario simulation."""
    asset_id: str
    scenario: str
    parameters: dict = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """
    Result of a what-if scenario simulation.

    This is a projection only — the real Digital Shadow is NOT mutated.
    """
    asset_id: str
    scenario: str
    parameters: dict
    current_state: dict = Field(..., description="State before simulation")
    projected_state: dict = Field(..., description="Projected state after scenario")
    deltas: dict = Field(..., description="Differences between current and projected")
    warnings: List[str] = Field(default_factory=list)
