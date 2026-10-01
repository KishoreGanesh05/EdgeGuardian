"""
Edge-Guardian Main Dashboard Application.
"""

import streamlit as st
import sys
import os

# Ensure the root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.components import render_fleet_overview, render_asset_detail, render_what_if_lab
from dashboard.data_source import get_fleet_manager, get_simulator, get_available_assets

def main():
    st.set_page_config(
        page_title="EdgeGuardian | Asset Intelligence",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ EdgeGuardian")
    st.subheader("Physics-Gated Edge AI for Transformer Asset Intelligence")

    # Load Backend Data
    fleet_manager = get_fleet_manager()
    simulator = get_simulator()
    assets = get_available_assets()

    if not assets:
        st.error("No assets available in the fleet.")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio(
        "Select View",
        ["Fleet Overview", "Asset Detail & Trends", "What-If Lab"]
    )

    st.sidebar.divider()
    selected_asset = st.sidebar.selectbox("Target Asset", assets)

    st.sidebar.markdown(
        """
        <div style="margin-top: 50px; font-size: 0.8em; color: gray;">
        Telemetry → Physics → Edge AI → Fault → Health → RUL → Digital Shadow → What-if
        </div>
        """,
        unsafe_allow_html=True
    )

    shadow = fleet_manager.get_shadow(selected_asset)

    if view_mode == "Fleet Overview":
        render_fleet_overview(fleet_manager)
    elif view_mode == "Asset Detail & Trends":
        render_asset_detail(shadow)
    elif view_mode == "What-If Lab":
        render_what_if_lab(shadow, simulator)

if __name__ == "__main__":
    main()
