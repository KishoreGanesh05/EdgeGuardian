"""
Edge-Guardian Configuration Module.

Centralizes all configurable parameters for the Edge-Guardian system.
All thresholds, model parameters, and simulation settings are defined here
rather than hard-coded throughout the application.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODEL_DIR = PROJECT_ROOT / "edge_ai" / "model"

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"

def setup_logging(level: Optional[str] = None) -> None:
    """Configure structured logging for the application."""
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    # Suppress noisy third-party loggers
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)

# ── Simulation Defaults ───────────────────────────────────────────────────
DEMO_ASSET_COUNT = int(os.getenv("DEMO_ASSET_COUNT", "20"))
DEMO_SEED = int(os.getenv("DEMO_SEED", "42"))
SAMPLE_INTERVAL_SECONDS = 1
NORMAL_DURATION_SECONDS = 300      # 5 minutes of normal operation
FAULT_DURATION_SECONDS = 120       # 2 minutes of fault operation

# ── Transformer Defaults ──────────────────────────────────────────────────
DEFAULT_TRANSFORMER = {
    "rated_power_kva": 500.0,
    "nominal_voltage_v": 11000.0,
    "nominal_frequency_hz": 50.0,
    "baseline_efficiency": 0.97,
    "base_resistance_ohm": 0.15,
    "thermal_coefficient": 0.004,
    "ambient_temperature_c": 35.0,
    "initial_health": 95.0,
    "age_factor": 1.0,
}

# ── Physics Engine ─────────────────────────────────────────────────────────
MEASUREMENT_NOISE_PERCENT = 0.5    # ±0.5% measurement noise
THERMAL_TIME_CONSTANT = 600.0      # seconds — simplified thermal inertia
THERMAL_RISE_COEFFICIENT = 40.0    # °C rise at full load above ambient

# ── Edge AI ────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "rms_v", "rms_i", "power", "power_factor",
    "efficiency", "temperature", "eff_delta"
]
FEATURE_COUNT = len(FEATURE_ORDER)

# Autoencoder architecture
AUTOENCODER_ENCODING_DIM = 4
AUTOENCODER_HIDDEN_DIM = 16
AUTOENCODER_EPOCHS = 100
AUTOENCODER_BATCH_SIZE = 32
AUTOENCODER_LEARNING_RATE = 0.001

# Anomaly detection
ANOMALY_THRESHOLD_METHOD = "percentile"
ANOMALY_THRESHOLD_PERCENTILE = 97.0

# Classifier
CLASSIFIER_UNKNOWN_CONFIDENCE = 0.60
CLASSIFIER_N_ESTIMATORS = 100
CLASSIFIER_RANDOM_STATE = 42

# ── Health Index ───────────────────────────────────────────────────────────
HEALTH_WARNING_THRESHOLD = 70.0
HEALTH_CRITICAL_THRESHOLD = 40.0

# Penalty weights for health calculation
HEALTH_WEIGHTS = {
    "anomaly_severity": 0.30,
    "thermal_stress": 0.20,
    "overload_exposure": 0.15,
    "efficiency_degradation": 0.20,
    "degradation_history": 0.15,
}

# ── RUL ────────────────────────────────────────────────────────────────────
RUL_CRITICAL_THRESHOLD = 20.0      # Health index below which RUL = 0
RUL_BASE_LIFETIME_HOURS = 200000.0 # ~23 years nominal lifetime

# ── Digital Shadow ─────────────────────────────────────────────────────────
HISTORY_MAX_LENGTH = 500           # Max history entries per asset

# ── Agent / GPT-5 ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"           # Fallback model if gpt-5 unavailable

# ── AWS (optional) ─────────────────────────────────────────────────────────
AWS_ENABLED = bool(os.getenv("AWS_ACCESS_KEY_ID", ""))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_IOT_ENDPOINT = os.getenv("AWS_IOT_ENDPOINT", "")


def load_scenarios_config() -> dict:
    """Load scenario configurations from YAML."""
    scenarios_path = CONFIG_DIR / "scenarios.yaml"
    if scenarios_path.exists():
        with open(scenarios_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}
