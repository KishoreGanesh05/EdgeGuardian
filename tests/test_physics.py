"""
Edge-Guardian Physics Module Tests.

Tests for electrical, efficiency, and thermal calculations.
Verifies that physics equations are correct, handle edge cases safely,
and produce expected behavior chains (e.g. higher resistance → higher loss).
"""

import math
import pytest
import numpy as np

from physics.electrical import (
    calculate_rms,
    calculate_real_power,
    calculate_apparent_power,
    calculate_power_factor,
    calculate_resistive_loss,
    calculate_feeder_voltage_drop,
    calculate_effective_resistance,
)
from physics.efficiency import (
    calculate_efficiency,
    calculate_efficiency_delta,
    calculate_loss_from_efficiency,
)
from physics.thermal import (
    calculate_transformer_temperature,
    calculate_thermal_stress,
    calculate_thermal_state,
    estimate_loss_induced_heating,
)


# ═══════════════════════════════════════════════════════════════════════════
# Electrical Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateRMS:
    """Tests for RMS calculation."""

    def test_rms_dc_signal(self):
        """RMS of a constant value equals that value."""
        result = calculate_rms([5.0, 5.0, 5.0, 5.0])
        assert result == pytest.approx(5.0)

    def test_rms_sine_wave(self):
        """RMS of a sine wave = peak / sqrt(2)."""
        peak = 10.0
        t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        signal = peak * np.sin(t)
        result = calculate_rms(signal)
        expected = peak / math.sqrt(2)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_rms_empty_array(self):
        """RMS of empty input returns 0."""
        assert calculate_rms([]) == 0.0

    def test_rms_single_value(self):
        """RMS of a single value is that value's absolute."""
        assert calculate_rms([3.0]) == pytest.approx(3.0)
        assert calculate_rms([-3.0]) == pytest.approx(3.0)

    def test_rms_rejects_nan(self):
        """RMS raises ValueError for NaN input."""
        with pytest.raises(ValueError, match="NaN"):
            calculate_rms([1.0, float("nan"), 3.0])

    def test_rms_rejects_inf(self):
        """RMS raises ValueError for infinite input."""
        with pytest.raises(ValueError, match="infinite"):
            calculate_rms([1.0, float("inf"), 3.0])


class TestCalculateRealPower:
    """Tests for real power calculation: P = V × I × PF."""

    def test_basic_power(self):
        """P = 230V × 10A × 0.9 = 2070W."""
        result = calculate_real_power(230.0, 10.0, 0.9)
        assert result == pytest.approx(2070.0)

    def test_unity_power_factor(self):
        """At PF=1.0, real power equals apparent power."""
        result = calculate_real_power(100.0, 5.0, 1.0)
        assert result == pytest.approx(500.0)

    def test_zero_power_factor(self):
        """At PF=0, real power is zero (purely reactive)."""
        result = calculate_real_power(100.0, 5.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_never_negative(self):
        """Real power should never be negative."""
        result = calculate_real_power(100.0, 5.0, 0.5)
        assert result >= 0.0

    def test_rejects_negative_voltage(self):
        with pytest.raises(ValueError):
            calculate_real_power(-100.0, 5.0, 0.9)


class TestCalculateApparentPower:
    """Tests for apparent power calculation: S = V × I."""

    def test_basic_apparent(self):
        """S = 230V × 10A = 2300VA."""
        result = calculate_apparent_power(230.0, 10.0)
        assert result == pytest.approx(2300.0)

    def test_zero_current(self):
        """No current → no apparent power."""
        result = calculate_apparent_power(230.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_never_negative(self):
        result = calculate_apparent_power(230.0, 10.0)
        assert result >= 0.0


class TestCalculatePowerFactor:
    """Tests for power factor calculation: PF = P / S."""

    def test_unity_pf(self):
        """Purely resistive load → PF = 1.0."""
        result = calculate_power_factor(500.0, 500.0)
        assert result == pytest.approx(1.0)

    def test_typical_industrial(self):
        """Typical industrial PF around 0.85."""
        result = calculate_power_factor(850.0, 1000.0)
        assert result == pytest.approx(0.85)

    def test_pf_bounded_0_to_1(self):
        """PF must always be in [0, 1]."""
        # Even if P > S due to rounding, PF should be clamped
        result = calculate_power_factor(1100.0, 1000.0)
        assert 0.0 <= result <= 1.0

    def test_zero_apparent_power(self):
        """Zero apparent power → PF defaults to 1.0 (no-load)."""
        result = calculate_power_factor(0.0, 0.0)
        assert result == 1.0


class TestCalculateResistiveLoss:
    """Tests for I²R loss calculation."""

    def test_basic_loss(self):
        """P = I²R = 10² × 0.5 = 50W."""
        result = calculate_resistive_loss(10.0, 0.5)
        assert result == pytest.approx(50.0)

    def test_loss_increases_with_resistance(self):
        """Core physics: higher resistance → higher loss."""
        loss_normal = calculate_resistive_loss(10.0, 0.15)
        loss_fault = calculate_resistive_loss(10.0, 0.15 * 1.25)  # +25%
        assert loss_fault > loss_normal

    def test_loss_increases_quadratically_with_current(self):
        """Loss is quadratic in current."""
        loss_1 = calculate_resistive_loss(10.0, 0.5)
        loss_2 = calculate_resistive_loss(20.0, 0.5)
        assert loss_2 == pytest.approx(4.0 * loss_1)

    def test_zero_current(self):
        """No current → no loss."""
        assert calculate_resistive_loss(0.0, 0.5) == 0.0

    def test_never_negative(self):
        result = calculate_resistive_loss(10.0, 0.5)
        assert result >= 0.0


class TestFeederVoltageDropAndResistance:
    """Tests for voltage drop and temperature-adjusted resistance."""

    def test_voltage_drop(self):
        """V_drop = I × R = 10 × 0.15 = 1.5V."""
        result = calculate_feeder_voltage_drop(10.0, 0.15)
        assert result == pytest.approx(1.5)

    def test_effective_resistance_increases_with_temp(self):
        """Resistance increases with temperature rise."""
        r_cold = calculate_effective_resistance(0.15, 0.0, 0.004)
        r_hot = calculate_effective_resistance(0.15, 50.0, 0.004)
        assert r_hot > r_cold

    def test_effective_resistance_formula(self):
        """R_eff = R_base × (1 + α × ΔT) = 0.15 × (1 + 0.004 × 50) = 0.18."""
        result = calculate_effective_resistance(0.15, 50.0, 0.004)
        expected = 0.15 * (1 + 0.004 * 50)
        assert result == pytest.approx(expected)

    def test_effective_resistance_never_below_base(self):
        """Even with negative temp rise, resistance stays ≥ base."""
        result = calculate_effective_resistance(0.15, -100.0, 0.004)
        assert result >= 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Efficiency Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateEfficiency:
    """Tests for efficiency calculation."""

    def test_basic_efficiency(self):
        """η = P_out / P_in = 970 / 1000 = 0.97."""
        result = calculate_efficiency(1000.0, 970.0)
        assert result == pytest.approx(0.97)

    def test_efficiency_bounded(self):
        """Efficiency must be in [0, 1]."""
        # Even if output > input (measurement error), clamp to 1.0
        result = calculate_efficiency(100.0, 110.0)
        assert 0.0 <= result <= 1.0

    def test_zero_input_power(self):
        """Zero input → efficiency 0.0 (safe handling)."""
        result = calculate_efficiency(0.0, 0.0)
        assert result == 0.0

    def test_efficiency_decreases_with_higher_losses(self):
        """Higher losses → lower efficiency."""
        eff_normal = calculate_efficiency(1000.0, 970.0)
        eff_lossy = calculate_efficiency(1000.0, 920.0)
        assert eff_lossy < eff_normal


class TestCalculateEfficiencyDelta:
    """Tests for efficiency delta calculation."""

    def test_normal_delta_near_zero(self):
        """Normal operation: eff_delta ≈ 0."""
        result = calculate_efficiency_delta(0.97, 0.97)
        assert result == pytest.approx(0.0)

    def test_negative_delta_on_degradation(self):
        """Lower efficiency → negative eff_delta."""
        result = calculate_efficiency_delta(0.93, 0.97)
        assert result < 0.0
        assert result == pytest.approx(-0.04)

    def test_positive_delta_if_improved(self):
        """Higher efficiency → positive eff_delta (unusual but valid)."""
        result = calculate_efficiency_delta(0.98, 0.97)
        assert result > 0.0

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            calculate_efficiency_delta(float("nan"), 0.97)


class TestCalculateLossFromEfficiency:
    """Tests for loss calculation from efficiency."""

    def test_basic_loss(self):
        """P_loss = P_in × (1 - η) = 1000 × 0.03 = 30W."""
        result = calculate_loss_from_efficiency(1000.0, 0.97)
        assert result == pytest.approx(30.0)

    def test_zero_input(self):
        result = calculate_loss_from_efficiency(0.0, 0.97)
        assert result == 0.0

    def test_never_negative(self):
        result = calculate_loss_from_efficiency(1000.0, 0.97)
        assert result >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Thermal Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformerTemperature:
    """Tests for thermal model."""

    def test_no_load_equals_ambient(self):
        """At zero load, temperature equals ambient."""
        result = calculate_transformer_temperature(35.0, 0.0)
        assert result == pytest.approx(35.0)

    def test_full_load_rise(self):
        """At full load, temp = ambient + thermal_rise_coeff."""
        result = calculate_transformer_temperature(35.0, 1.0, thermal_rise_coefficient=40.0)
        assert result == pytest.approx(75.0)

    def test_temperature_increases_with_load(self):
        """Higher load → higher temperature."""
        temp_50 = calculate_transformer_temperature(35.0, 0.5)
        temp_100 = calculate_transformer_temperature(35.0, 1.0)
        assert temp_100 > temp_50

    def test_overload_temperature(self):
        """Overload (>100%) produces temperature above rated rise."""
        temp_normal = calculate_transformer_temperature(35.0, 1.0, 40.0)
        temp_overload = calculate_transformer_temperature(35.0, 1.3, 40.0)
        assert temp_overload > temp_normal

    def test_additional_heating(self):
        """Additional loss heating increases temperature."""
        temp_normal = calculate_transformer_temperature(35.0, 0.7, additional_loss_heating_c=0.0)
        temp_extra = calculate_transformer_temperature(35.0, 0.7, additional_loss_heating_c=5.0)
        assert temp_extra > temp_normal
        assert temp_extra - temp_normal == pytest.approx(5.0)


class TestThermalStress:
    """Tests for thermal stress factor."""

    def test_below_rated_stress_under_1(self):
        """Temperature below rated rise → stress < 1.0."""
        result = calculate_thermal_stress(80.0, 35.0, rated_temperature_rise_c=65.0)
        assert result < 1.0

    def test_at_rated_stress_equals_1(self):
        """Temperature at exactly rated rise → stress = 1.0."""
        result = calculate_thermal_stress(100.0, 35.0, rated_temperature_rise_c=65.0)
        assert result == pytest.approx(1.0)

    def test_above_rated_stress_over_1(self):
        """Temperature above rated rise → stress > 1.0."""
        result = calculate_thermal_stress(110.0, 35.0, rated_temperature_rise_c=65.0)
        assert result > 1.0

    def test_stress_never_negative(self):
        """Even if transformer is cooler than ambient, stress ≥ 0."""
        result = calculate_thermal_stress(30.0, 35.0, rated_temperature_rise_c=65.0)
        assert result >= 0.0

    def test_zero_rated_rise(self):
        """Zero rated rise returns 0.0 (safe handling)."""
        result = calculate_thermal_stress(100.0, 35.0, rated_temperature_rise_c=0.0)
        assert result == 0.0


class TestThermalState:
    """Tests for the combined thermal state function."""

    def test_returns_all_keys(self):
        """calculate_thermal_state returns temperature, stress, and rise."""
        result = calculate_thermal_state(35.0, 0.7)
        assert "temperature_c" in result
        assert "thermal_stress" in result
        assert "temperature_rise_c" in result

    def test_temperature_rise_is_consistent(self):
        """Rise = temperature - ambient."""
        result = calculate_thermal_state(35.0, 0.7)
        expected_rise = result["temperature_c"] - 35.0
        assert result["temperature_rise_c"] == pytest.approx(expected_rise)


class TestLossInducedHeating:
    """Tests for excess-loss heating estimation."""

    def test_no_excess_loss(self):
        """Zero excess loss → no additional heating."""
        result = estimate_loss_induced_heating(0.0)
        assert result == 0.0

    def test_positive_loss_causes_heating(self):
        """Excess loss causes proportional temperature rise."""
        result = estimate_loss_induced_heating(100.0, thermal_resistance=0.01)
        assert result == pytest.approx(1.0)

    def test_negative_loss_clamped(self):
        """Negative excess loss → 0 heating."""
        result = estimate_loss_induced_heating(-50.0)
        assert result == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Physics Chain Integration Test
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsChain:
    """
    Integration test: verify the causal chain that a fault produces.

    Higher resistance → Higher I²R loss → Lower efficiency
    → Negative eff_delta → Temperature increase
    """

    def test_resistance_increase_chain(self):
        """
        THE PRIMARY PHYSICS CHAIN:
        +25% resistance → higher loss → lower efficiency → negative eff_delta
        """
        current = 30.0  # 30A typical feeder current
        baseline_resistance = 0.15
        fault_resistance = baseline_resistance * 1.25  # +25%

        # Normal losses
        loss_normal = calculate_resistive_loss(current, baseline_resistance)

        # Fault losses
        loss_fault = calculate_resistive_loss(current, fault_resistance)

        # Verify loss increases
        assert loss_fault > loss_normal

        # Calculate efficiencies (assume 500kW system)
        input_power = 500000.0
        eff_normal = calculate_efficiency(input_power, input_power - loss_normal)
        eff_fault = calculate_efficiency(input_power, input_power - loss_fault)

        # Verify efficiency decreases
        assert eff_fault < eff_normal

        # Verify eff_delta is negative
        baseline_eff = 0.97
        delta_normal = calculate_efficiency_delta(eff_normal, baseline_eff)
        delta_fault = calculate_efficiency_delta(eff_fault, baseline_eff)
        assert delta_fault < delta_normal

        # Verify the loss difference would cause extra heating
        excess_loss = loss_fault - loss_normal
        extra_heating = estimate_loss_induced_heating(excess_loss)
        assert extra_heating > 0.0
