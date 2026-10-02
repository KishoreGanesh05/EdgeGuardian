# Edge-Guardian — Stage 2 Software Prototype

A layered, physics-first intelligence pipeline for transformer and feeder
condition monitoring. **Stage 2 is software-only** — it emulates the telemetry
that ESP32 / CT-VT / industrial-gateway hardware would eventually provide, and
runs entirely locally with **no hardware, no AWS, and no GPT-5 API required**.

```
Telemetry → Physics → Features → Edge AI → Health → RUL → Digital Shadow
                                                               ↓
                                                  Fleet · What-If · Agent
```

The central principle: *Physics creates trustworthy features. Edge AI detects
abnormal behavior. Health/RUL converts behavior into asset condition. Digital
Shadows organize state. Simulation tests possible futures. GPT-5 explains and
orchestrates for engineers.*

---

## Quick start

```bash
pip install -r requirements.txt
python main.py --demo
```

The demo builds a 25-transformer virtual fleet, runs normal operation, injects
a fault into **T023**, and prints the detection → health → RUL cascade, then
runs an engineering-assistant Q&A — all offline.

### Useful flags

```bash
python main.py --demo --asset-count 30 --target T023   # custom fleet / target
python main.py --demo --no-agent                        # skip agent Q&A
python main.py --demo --ask "Compare T023 and T024"     # one extra question
python main.py --demo --force-fallback                  # never use the LLM
```

---

## Architecture (layers)

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Simulation | `data/` | Virtual fleet configs, physically-consistent telemetry, fault scenarios |
| Physics | `physics/` | Electrical, efficiency, thermal equations (pure functions) |
| Edge AI | `edge_ai/` | 7-feature vector → autoencoder anomaly score + RandomForest fault class |
| Health | `health/` | Project-defined Health Index (0–100) and model-based RUL |
| Digital Shadow | `digital_shadow/` | Authoritative per-asset state + bounded history; fleet aggregation |
| Simulator | `simulator/` | What-if projections that never mutate live state |
| **Agent (P5)** | `agent/` | Structured tools + GPT-5/deterministic engineering assistant |
| **Cloud (P5)** | `cloud/` | Optional telemetry sinks (Local / AWS IoT / SiteWise) |
| **Integration (P5)** | `main.py` | Orchestration + `--demo` end-to-end story |
| Dashboard | `dashboard/` | Streamlit fleet/asset/fault-injection/what-if views |

The seven-value feature vector (fixed order):
`[RMS_V, RMS_I, Power, Power_Factor, Efficiency, Temperature, Eff_Delta]`.

---

## The engineering assistant (agent)

The agent inspects system state **only** through the structured tool layer in
[`agent/tools.py`](agent/tools.py) and never fabricates measurements. It has two
interchangeable backends:

- **Deterministic fallback** (default, always available): a rule-based
  investigator that calls the same tools and composes an evidence-based answer
  from real values. This is what the demo uses — no API key needed.
- **GPT-5 / GPT-4o** (optional): set `OPENAI_API_KEY` and install `openai`
  (`pip install openai`) to enable a tool-calling LLM loop.

Tools (all return JSON-serializable, summarized evidence):
`get_transformer_status`, `get_asset_history`, `get_fleet_health`,
`get_high_risk_assets`, `get_anomaly_details`, `calculate_rul`, `run_scenario`,
`compare_assets`, `generate_maintenance_plan`.

The agent cannot issue breaker commands or any safety-critical control action,
and labels Health Index / RUL as model-based (not industry-certified).

---

## Optional cloud sinks

`cloud/` provides a `TelemetrySink` interface. `LocalSink` is the default and
requires nothing. `AwsIoTSink` and `SiteWiseSink` import `boto3` lazily and
raise a clear error only if actually used without the dependency — the core
pipeline never depends on them.

```python
from cloud.base import LocalSink
from cloud.aws_iot import AwsIoTSink      # optional
sink = LocalSink()                         # default; used by the demo
```

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Fleet overview, asset detail with health/anomaly trends, fault injection, and
what-if views (what-if never mutates real state).

---

## Testing

```bash
python -m pytest tests/ -q
```

The suite covers physics, generator, Edge AI, health/RUL, digital shadow,
simulator, dashboard, **cloud sinks, agent tools/assistant, and the mandatory
end-to-end cascade**. Training the TensorFlow autoencoder makes the full suite
take a few minutes; the P5 integration tests train the engine once per session.

---

## Configuration

All thresholds and parameters live in [`config/settings.py`](config/settings.py)
and [`config/scenarios.yaml`](config/scenarios.yaml). Copy `.env.example` to
`.env` to override via environment variables (all optional).

---

## Known limitations (Stage 2)

- **Feeder-resistance fault is numerically weak.** In the Stage-2 physics,
  efficiency is insensitive to feeder resistance at 11 kV (feeder I²R loss is
  tiny relative to throughput); only temperature responds, and only slightly.
  `FEEDER_RESISTANCE_INCREASE` is fully supported and propagates through the
  pipeline, but does not reliably trip the autoencoder at realistic severities.
  **The demo and end-to-end test therefore inject `OVERLOAD` into T023**, which
  genuinely exercises the detection → health → RUL cascade.
- Health Index and RUL are **project-defined, model-based** metrics for Stage 2
  demonstration — **not** industry-certified or validated against real failure
  data.
- The thermal and electrical models are simplified simulation models, not
  certified transformer models (e.g. IEEE C57.91).
- Edge-Guardian is for condition monitoring and analysis only. It does **not**
  replace protection relays or perform any safety-critical trip/control logic.
