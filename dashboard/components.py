"""
Edge-Guardian Dashboard Components.
"""

import streamlit as st
import pandas as pd
from digital_shadow.fleet import FleetManager
from digital_shadow.shadow import DigitalShadow
from simulator.what_if import WhatIfSimulator
from data.models import FaultScenario

def render_fleet_overview(fleet: FleetManager):
    st.header("Fleet Overview")

    metrics = fleet.get_fleet_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Assets", metrics["asset_count"])
    col2.metric("Healthy", metrics["healthy_count"])
    col3.metric("Warning", metrics["warning_count"])
    col4.metric("Critical", metrics["critical_count"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Health", f"{metrics['average_health']:.1f} / 100")
    col2.metric("Minimum Health", f"{metrics['minimum_health']:.1f} / 100")
    col3.metric("Average RUL", f"{metrics['average_rul_hours']:.0f} hrs")

def render_asset_detail(shadow: DigitalShadow):
    st.header(f"Asset Detail: {shadow.state.asset_id}")

    s = shadow.state

    # ── Telemetry & Physics ──
    st.subheader("Operational & Physics State")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Load", f"{s.load_percent:.1f} %")
    c2.metric("Temperature", f"{s.temperature:.1f} °C")
    c3.metric("Efficiency", f"{s.efficiency * 100:.2f} %", f"{s.eff_delta * 100:.2f} %")
    c4.metric("Power Factor", f"{s.power_factor:.3f}")

    # ── Edge AI ──
    st.subheader("Edge AI Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Score", f"{s.anomaly_score:.2f}")
    c2.metric("Status", "ANOMALY" if s.is_anomaly else "NORMAL", delta_color="inverse")

    if s.fault_class:
        c3.metric("Classified Fault", s.fault_class, f"{s.fault_confidence * 100:.1f}% conf")
    else:
        c3.metric("Classified Fault", "None")

    # ── Health & RUL ──
    st.subheader("Health & RUL")
    c1, c2, c3 = st.columns(3)

    color = "green" if s.health_state == "NORMAL" else "orange" if s.health_state == "WARNING" else "red"
    c1.markdown(f"**Health Index:** <span style='color:{color}; font-size:24px;'>{s.health_index:.1f}</span>", unsafe_allow_html=True)
    c2.markdown(f"**State:** <span style='color:{color}; font-size:24px;'>{s.health_state}</span>", unsafe_allow_html=True)
    c3.markdown(f"**Est. RUL:** <span style='font-size:24px;'>{s.rul_hours:.0f} hrs</span>", unsafe_allow_html=True)
    st.caption("ℹ️ RUL is a Model-based / synthetic Stage 2 estimate.")

    # ── Trends ──
    st.subheader("Trends")

    if s.health_history:
        st.line_chart(s.health_history, height=150, use_container_width=True)
        st.caption("Health Index Trend")

    if s.temperature_history:
        st.line_chart(s.temperature_history, height=150, use_container_width=True)
        st.caption("Temperature Trend")

def render_what_if_lab(shadow: DigitalShadow, simulator: WhatIfSimulator):
    st.header(f"What-If Lab: {shadow.state.asset_id}")
    st.write("Inject hypothetical fault scenarios to observe downstream effects on physics, AI, and health.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Controls")
        scenario = st.selectbox(
            "Fault Scenario",
            [s for s in FaultScenario if s != FaultScenario.NORMAL]
        )
        severity = st.slider("Severity Multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        run_btn = st.button("Run Simulation", type="primary")

    with col2:
        if run_btn:
            with st.spinner("Running simulation through Physics → Edge AI → Health pipelines..."):
                result = simulator.run_scenario(shadow, scenario, severity)

            st.subheader("WHAT-IF / SIMULATED Result")
            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Current")
                cs = result.current_state
                st.metric("Health Index", f"{cs['health_index']:.1f}")
                st.metric("RUL", f"{cs['rul_hours']:.0f} hrs")
                st.metric("Anomaly Score", f"{cs['anomaly_score']:.2f}")
                st.metric("Fault", cs['fault_class'] or "None")
                st.metric("Efficiency", f"{cs['efficiency']*100:.2f} %")
                st.metric("Temperature", f"{cs['temperature']:.1f} °C")

            with c2:
                st.markdown("### Simulated")
                ps = result.projected_state
                ds = result.deltas
                st.metric("Health Index", f"{ps['health_index']:.1f}", f"{ds['health_index']:.1f}")
                st.metric("RUL", f"{ps['rul_hours']:.0f} hrs", f"{ds['rul_hours']:.0f} hrs")
                st.metric("Anomaly Score", f"{ps['anomaly_score']:.2f}", f"{ds['anomaly_score']:.2f}")
                st.metric("Fault", ps['fault_class'] or "None")
                st.metric("Efficiency", f"{ps['efficiency']*100:.2f} %", f"{ds['efficiency']*100:.2f} %")
                st.metric("Temperature", f"{ps['temperature']:.1f} °C", f"{ds['temperature']:.1f} °C")
