# Edge-Guardian — Stage 2 Coding-Agent Implementation Plan

## 0. Purpose

This file is the implementation contract for a coding agent building the Stage 2 software prototype of **Edge-Guardian**.

**Stage 2 is software-first and hardware-free.**

Do NOT require or assume:
- ESP32-S3 hardware
- physical CT/VT sensors
- physical transformer
- ADS1115 hardware
- physical industrial gateway
- real SCADA/PLC/relay connection
- AWS-connected industrial hardware

Instead, emulate the telemetry those systems would eventually provide. The software architecture must remain compatible with a later Stage 3 physical implementation.

---

# 1. Product Definition

Edge-Guardian is a layered intelligence pipeline for transformer and feeder condition monitoring:

```text
Virtual Transformer / Industrial OT Telemetry
                    |
                    v
          Data Acquisition Layer
                    |
                    v
             Physics Engine
                    |
                    v
           Feature Extraction
                    |
                    v
             Edge AI / TinyML
              /           \
             v             v
      Anomaly Score    Fault Class
              \           /
               v         v
              Health Index
                    |
                    v
                   RUL
                    |
                    v
             Digital Shadow
                    |
                    v
            Fleet Intelligence
                    |
                    v
            Scenario Simulator
                    |
                    v
           GPT-5 Agent / Tools
                    |
                    v
          Engineer / Maintenance
```

AWS is an optional cloud/fleet integration layer. It must not become a dependency for the local Edge AI pipeline.

---

# 2. Primary Engineering Goal

Build an end-to-end software demo where:

1. Multiple virtual transformers generate physically consistent telemetry.
2. A controlled fault can be injected into one transformer.
3. The physics engine calculates the physical consequences.
4. Physics-derived features are passed to Edge AI.
5. Edge AI detects anomalous behavior.
6. A classifier identifies a modeled fault scenario.
7. Health Index changes.
8. RUL/degradation state changes.
9. The asset Digital Shadow updates.
10. Fleet metrics update.
11. A what-if scenario can be simulated.
12. A GPT-5 agent can inspect structured state through tools and produce an evidence-based explanation.
13. The complete pipeline runs locally without AWS or physical hardware.

---

# 3. Non-Negotiable Architecture Rules

## 3.1 Physics first

Do not feed arbitrary raw telemetry directly into ML.

```text
Telemetry
   |
   v
Physics calculations
   |
   v
Engineering features
   |
   v
ML inference
```

The required seven-value Edge AI feature vector is:

```text
RMS_V
RMS_I
Power
Power_Factor
Efficiency
Temperature
Eff_Delta
```

The internal system may maintain more variables, but the baseline ML interface must support these seven features in exactly this order.

## 3.2 Edge AI is the core

Do not redesign the project as:

```text
Cloud -> LLM -> prediction
```

Use:

```text
Physics -> Edge AI -> Health/RUL -> Digital Shadow -> Agent
```

GPT-5 is a reasoning/orchestration layer. It is not the physics engine, anomaly detector, or protection relay.

## 3.3 Protection remains deterministic

Edge-Guardian must not replace protection relays or safety-critical trip logic.

It is for:
- condition monitoring
- anomaly detection
- predictive maintenance
- engineering analysis
- asset intelligence

It must not directly control breakers, protection relays, emergency trips, or safety interlocks.

---

# 4. Technology Direction

Use Python for the Stage 2 core.

Recommended stack:

```text
Python 3.11+
NumPy
Pandas
Scikit-learn
TensorFlow / Keras
TensorFlow Lite
Pydantic
Streamlit
Pytest
```

Optional:

```text
Boto3
AWS IoT / SiteWise integration
OpenAI SDK for GPT-5 agent
```

Optional dependencies must not prevent the local demo from running.

---

# 5. Repository Structure

Create this logical structure:

```text
edge-guardian/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   ├── settings.py
│   └── scenarios.yaml
│
├── data/
│   ├── __init__.py
│   ├── models.py
│   ├── generator.py
│   ├── scenarios.py
│   └── datasets/
│
├── physics/
│   ├── __init__.py
│   ├── electrical.py
│   ├── thermal.py
│   └── efficiency.py
│
├── edge_ai/
│   ├── __init__.py
│   ├── features.py
│   ├── anomaly.py
│   ├── classifier.py
│   ├── preprocessing.py
│   └── model/
│       ├── autoencoder.keras
│       ├── autoencoder.tflite
│       └── metadata.json
│
├── health/
│   ├── __init__.py
│   ├── health_index.py
│   └── rul.py
│
├── digital_shadow/
│   ├── __init__.py
│   ├── asset.py
│   ├── fleet.py
│   └── history.py
│
├── simulator/
│   ├── __init__.py
│   └── what_if.py
│
├── agent/
│   ├── __init__.py
│   ├── tools.py
│   ├── prompts.py
│   └── agent.py
│
├── cloud/
│   ├── __init__.py
│   ├── aws_iot.py
│   └── sitewise.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_physics.py
│   ├── test_generator.py
│   ├── test_features.py
│   ├── test_anomaly.py
│   ├── test_health.py
│   ├── test_rul.py
│   ├── test_digital_shadow.py
│   ├── test_simulator.py
│   └── test_end_to_end.py
│
└── main.py
```

---

# 6. Data Contracts

Use typed models. Pydantic is preferred.

## 6.1 TransformerConfig

Required:

```text
asset_id: str
rated_power_kva: float
nominal_voltage_v: float
nominal_frequency_hz: float
baseline_efficiency: float
base_resistance_ohm: float
thermal_coefficient: float
ambient_temperature_c: float
initial_health: float
age_factor: float
```

Keep physical parameters configurable.

## 6.2 TelemetrySample

Required:

```text
timestamp
asset_id
voltage_feeder_v
current_feeder_a
voltage_secondary_v
current_secondary_a
frequency_hz
ambient_temperature_c
transformer_temperature_c
load_percent
```

Optional:

```text
scenario_id
scenario_severity
```

Do not put ML outputs into raw telemetry.

## 6.3 PhysicsResult

Required:

```text
rms_v
rms_i
power_w
apparent_power_va
power_factor
input_power_w
output_power_w
loss_w
efficiency
eff_delta
thermal_stress
```

## 6.4 FeatureVector

Required:

```text
rms_v
rms_i
power
power_factor
efficiency
temperature
eff_delta
```

Implement:

```python
to_array() -> np.ndarray
```

It must return exactly:

```python
[
    rms_v,
    rms_i,
    power,
    power_factor,
    efficiency,
    temperature,
    eff_delta
]
```

## 6.5 AIResult

```text
anomaly_score: float
is_anomaly: bool
fault_class: str | None
fault_confidence: float | None
model_version: str
```

## 6.6 HealthResult

```text
health_index: float
health_state: str
degradation_score: float
```

Allowed states:

```text
NORMAL
WARNING
CRITICAL
```

Thresholds must be configurable.

## 6.7 RULResult

```text
rul_hours: float
critical_threshold: float
current_degradation: float
model_based: bool
```

For Stage 2, `model_based` must be `True`.

## 6.8 DigitalShadow

Required:

```text
asset_id
timestamp
rated_power
load_percent
temperature
efficiency
power_factor
health_index
health_state
anomaly_score
fault_class
rul_hours
health_history
anomaly_history
```

Use bounded/configurable history.

---

# 7. Module 1 — Synthetic Telemetry Generator

File:

```text
data/generator.py
```

## Responsibility

Generate physically plausible telemetry for virtual transformers.

Do NOT use pure random values as the primary model. Use deterministic equations plus controlled noise.

Required interface:

```python
generate_telemetry(
    config,
    duration_seconds,
    sample_interval_seconds,
    seed=None
)
```

Normal operation should:

1. Generate load percentage.
2. Convert load to current.
3. Calculate voltage behavior.
4. Calculate power.
5. Calculate power factor.
6. Calculate losses.
7. Calculate efficiency.
8. Calculate temperature.
9. Add small measurement noise.

The generator must be deterministic when a seed is supplied.

---

# 8. Module 2 — Fault Scenarios

File:

```text
data/scenarios.py
```

Minimum scenarios:

```text
NORMAL
FEEDER_RESISTANCE_INCREASE
OVERLOAD
THERMAL_STRESS
EFFICIENCY_DEGRADATION
```

Optional:

```text
SENSOR_INCONSISTENCY
VOLTAGE_ABNORMALITY
POWER_FACTOR_DEGRADATION
```

## Feeder resistance increase

Concept:

```text
R_fault = R_base * (1 + resistance_increase_percent / 100)
```

The scenario must produce the chain:

```text
Higher resistance
      ->
Higher I²R loss
      ->
Lower efficiency
      ->
Potential temperature increase
      ->
Feature deviation
      ->
Anomaly
```

## Overload

Increase load beyond the normal envelope and affect:

- current
- power
- losses
- temperature
- degradation

## Thermal stress

Use elevated ambient temperature and/or sustained loading.

## Efficiency degradation

Reduce effective efficiency in a parameterized way.

All scenarios must be configurable rather than hard-coded.

---

# 9. Module 3 — Electrical Physics

File:

```text
physics/electrical.py
```

Implement pure functions:

```python
calculate_rms(values)
calculate_real_power(voltage, current, power_factor)
calculate_apparent_power(voltage, current)
calculate_power_factor(real_power, apparent_power)
calculate_resistive_loss(current, resistance)
```

Handle zero denominators safely.

Never silently return NaN.

---

# 10. Module 4 — Efficiency Physics

File:

```text
physics/efficiency.py
```

Implement:

```python
calculate_efficiency(input_power, output_power)
calculate_efficiency_delta(feeder_efficiency, secondary_efficiency)
```

Document assumptions and handle invalid/zero input power safely.

---

# 11. Module 5 — Thermal Model

File:

```text
physics/thermal.py
```

Implement a simple deterministic thermal model.

Inputs:

```text
ambient temperature
load/current
age factor
thermal coefficient
```

Outputs:

```text
transformer temperature
thermal stress
```

Higher sustained load must produce higher thermal stress.

This is a project simulation model, not a certified transformer thermal model.

---

# 12. Module 6 — Feature Extraction

File:

```text
edge_ai/features.py
```

Input:

```text
TelemetrySample + PhysicsResult
```

Output:

```text
FeatureVector
```

Mandatory features:

```text
RMS_V
RMS_I
Power
Power_Factor
Efficiency
Temperature
Eff_Delta
```

Never silently reorder them.

---

# 13. Module 7 — Preprocessing

File:

```text
edge_ai/preprocessing.py
```

Implement:

```python
fit_scaler(normal_training_data)
transform(features)
save_scaler(...)
load_scaler(...)
```

The scaler must be fitted only on normal training data.

Never fit preprocessing on the complete dataset containing test anomalies.

Store preprocessing metadata.

---

# 14. Module 8 — TinyML Anomaly Detector

File:

```text
edge_ai/anomaly.py
```

Use a lightweight autoencoder.

Architecture:

```text
7 inputs
   |
small Dense layer
   |
small latent layer
   |
Dense layer
   |
7 outputs
```

Train primarily on normal operating windows.

Anomaly score:

```text
MSE(input_vector, reconstructed_vector)
```

Choose the threshold from validation data and save it in model metadata.

Implement:

```python
predict(features) -> AIResult
```

Return:

```text
anomaly_score
is_anomaly
model_version
```

Keep the model small enough to support later TFLite Micro conversion.

---

# 15. Module 9 — Fault Classifier

File:

```text
edge_ai/classifier.py
```

Recommended first model:

```text
RandomForestClassifier
```

Input:

```text
seven-value feature vector
```

Output:

```text
fault_class
confidence
```

Train only on explicitly labeled synthetic scenarios.

If confidence is below a configurable threshold, return:

```text
UNKNOWN
```

Do not force a class for every anomaly.

---

# 16. Module 10 — Health Index

File:

```text
health/health_index.py
```

Create a project-defined 0–100 Health Index.

Conceptual penalty inputs:

```text
anomaly severity
thermal stress
overload exposure
efficiency degradation
degradation history
```

Do not describe it as an industry-standard score.

Implement:

```python
calculate_health(...)
```

Return:

```text
health_index
health_state
degradation_score
```

Expected behavior:

```text
normal -> high health -> NORMAL
moderate sustained abnormality -> lower health -> WARNING
severe/sustained abnormality -> low health -> CRITICAL
```

---

# 17. Module 11 — RUL

File:

```text
health/rul.py
```

Implement a physics-informed degradation model using:

```text
current degradation
thermal stress
overload exposure
anomaly persistence
age factor
```

Project degradation until a configurable critical threshold.

Do NOT claim it is learned from real failure records.

Output:

```text
rul_hours
```

Rules:

```text
critical/already failed -> RUL = 0
RUL must never be negative
model_based = True
```

---

# 18. Module 12 — Digital Shadow

Files:

```text
digital_shadow/asset.py
digital_shadow/history.py
digital_shadow/fleet.py
```

Each transformer has one Digital Shadow.

Update order:

```text
Telemetry
    ->
Physics
    ->
Features
    ->
AI
    ->
Health
    ->
RUL
    ->
Digital Shadow
```

The Digital Shadow is the authoritative current-state object.

Implement:

```python
get_current_state(asset_id)
get_history(asset_id)
update(asset_id, result)
```

The dashboard and agent should read the Digital Shadow rather than independently recomputing state.

---

# 19. Module 13 — Fleet Intelligence

File:

```text
digital_shadow/fleet.py
```

Implement:

```python
get_fleet_health()
get_critical_assets()
get_warning_assets()
get_average_efficiency()
get_total_energy_loss()
get_anomaly_count()
get_asset(asset_id)
```

The fleet layer aggregates individual Digital Shadows.

It must not bypass the individual asset pipeline.

---

# 20. Module 14 — What-If Scenario Engine

File:

```text
simulator/what_if.py
```

Implement:

```python
run_scenario(asset_id, scenario, parameters)
```

Required:

```text
LOAD_INCREASE
AMBIENT_TEMPERATURE_INCREASE
FEEDER_RESISTANCE_INCREASE
```

Optional:

```text
ASSET_OUTAGE
DEMAND_REDISTRIBUTION
```

## Critical rule

A scenario must NOT mutate the real Digital Shadow by default.

Correct:

```text
current state
    ->
temporary copy
    ->
scenario simulation
    ->
projected result
```

This allows the agent to answer:

```text
What happens if T023 load increases another 20%?
```

without changing T023's actual state.

---

# 21. Module 15 — Agent Tools

File:

```text
agent/tools.py
```

Required tools:

```python
get_transformer_status(asset_id)
get_asset_history(asset_id)
get_fleet_health()
get_high_risk_assets()
get_anomaly_details(asset_id)
calculate_rul(asset_id)
run_scenario(asset_id, scenario, parameters)
compare_assets(asset_a, asset_b)
generate_maintenance_plan(asset_id)
```

All tools must return structured JSON-serializable objects.

Do not return huge raw datasets to the LLM.

Return summarized evidence.

---

# 22. Agent Behavior

For:

```text
Why is T023 unhealthy?
```

Use:

```text
get_transformer_status("T023")
        |
        v
get_anomaly_details("T023")
        |
        v
get_asset_history("T023")
        |
        v
optional run_scenario(...)
        |
        v
evidence-based explanation
```

The agent must not invent:

- temperatures
- anomaly scores
- RUL
- fault classes
- maintenance history
- sensor readings

If data is unavailable, state that it is unavailable.

---

# 23. Maintenance Plan

Implement:

```python
generate_maintenance_plan(asset_id)
```

Example structure:

```json
{
  "asset_id": "T023",
  "priority": "HIGH",
  "reason": "...",
  "evidence": [],
  "recommended_checks": [],
  "next_action": "..."
}
```

For Stage 2 this is a draft engineering action only.

Do not automatically dispatch real maintenance.

---

# 24. Dashboard

File:

```text
dashboard/app.py
```

Use Streamlit.

Required views:

## Fleet overview

Show:

```text
Total assets
Normal
Warning
Critical
Average health
Average efficiency
Total estimated energy loss
Anomaly count
```

## Asset detail

Show:

```text
Asset ID
Health
Health state
Temperature
Efficiency
Power factor
Load
Anomaly score
Fault class
RUL
```

Also show:

```text
health trend
anomaly trend
```

## Fault injection

Controls:

```text
asset
scenario
severity
duration
```

## What-if

Controls:

```text
asset
scenario
parameter
```

Show:

```text
current value
projected value
difference
```

Never mutate real state for a what-if operation.

---

# 25. Main Pipeline

File:

```text
main.py
```

Implement a clear orchestration function similar to:

```python
def process_sample(asset_config, telemetry):
    physics = calculate_physics(telemetry, asset_config)

    features = extract_features(
        telemetry,
        physics
    )

    ai_result = edge_ai.predict(features)

    health = calculate_health(
        features=features,
        ai_result=ai_result
    )

    rul = calculate_rul(
        features=features,
        health=health
    )

    digital_shadow.update(
        telemetry=telemetry,
        physics=physics,
        features=features,
        ai=ai_result,
        health=health,
        rul=rul
    )
```

Keep orchestration separate from individual calculations.

---

# 26. End-to-End Demo Command

Required command:

```bash
python main.py --demo
```

It must:

1. Create a fleet.
2. Run normal operation.
3. Inject a fault into one asset.
4. Process resulting telemetry.
5. Update Digital Shadow.
6. Print a concise diagnostic.
7. Show changed fleet status.

Example structure:

```text
EDGE-GUARDIAN DEMO

Fleet: 20 assets

Initial:
Normal: 20
Warning: 0
Critical: 0

Injecting:
Asset: T023
Scenario: FEEDER_RESISTANCE_INCREASE
Severity: 25%

Result:
T023 anomaly score: ...
T023 fault class: FEEDER_RESISTANCE_INCREASE
T023 health: ... -> ...
T023 RUL: ... hours

Fleet:
Normal: ...
Warning: ...
Critical: ...
```

Use real computed values; never hard-code the displayed result.

---

# 27. Dataset and Training Workflow

Provide reproducible commands such as:

```bash
python -m data.generator --generate
python -m edge_ai.anomaly --train
python -m edge_ai.classifier --train
```

Or a documented equivalent such as:

```bash
python scripts/train_models.py
```

Training must be reproducible with fixed seeds.

Save:

```text
model
scaler
threshold
training metadata
feature order
model version
```

---

# 28. Dataset Splitting

For classifier:

```text
train
validation
test
```

For anomaly detector:

```text
normal training
normal validation
mixed test
```

Never contaminate normal training with fault scenarios.

---

# 29. Required Tests

## Physics

Verify:

```text
RMS is correct
Power calculation is correct
PF is bounded
I²R loss increases with resistance
Efficiency is sensible
Efficiency delta is correct
```

## Generator

Verify:

```text
same seed -> same output
different seed -> different noise
normal load remains plausible
fault scenario changes expected physical quantities
```

## Features

Verify:

```text
exactly seven features
correct ordering
no unexpected NaN
```

## Anomaly

Verify:

```text
normal sample generally has lower anomaly score
known anomaly produces increased anomaly score
model loads correctly
```

## Classifier

Verify:

```text
known scenario can be classified
low confidence can become UNKNOWN
model loads correctly
```

## Health

Verify:

```text
normal -> high health
moderate stress -> lower health
severe stress -> lower health
health remains 0..100
```

## RUL

Verify:

```text
higher degradation -> lower RUL
critical degradation -> RUL 0
RUL never negative
```

## Digital Shadow

Verify:

```text
asset creation
asset update
history recording
bounded history
fleet aggregation
```

## Scenario

Verify:

```text
scenario returns projection
real state is unchanged by default
projected values move in expected direction
```

## Mandatory end-to-end test

`tests/test_end_to_end.py` must test:

```text
create fleet
    ->
generate normal telemetry
    ->
process
    ->
inject feeder resistance fault into T023
    ->
process
    ->
T023 anomaly increases
    ->
T023 health decreases
    ->
Digital Shadow updates
    ->
fleet metrics update
    ->
scenario engine works
```

---

# 30. Configuration

Use configuration rather than hard-coded constants.

Example:

```yaml
health:
  warning_threshold: 70
  critical_threshold: 40

anomaly:
  threshold_method: percentile
  threshold_percentile: 99

classifier:
  unknown_confidence_threshold: 0.60

simulation:
  default_assets: 20
  sample_interval_seconds: 1
```

Validate thresholds against generated data.

---

# 31. Logging and Errors

Use Python logging.

Log:

```text
simulation start
fault injection
model loading
anomaly detection
health transition
scenario execution
agent tool calls
```

Never log secrets.

Explicitly handle:

```text
missing values
NaN
infinite values
zero apparent power
invalid asset IDs
unknown scenarios
missing models
missing configuration
```

Do not silently propagate invalid data.

---

# 32. AWS Integration Boundary

Files:

```text
cloud/aws_iot.py
cloud/sitewise.py
```

These are optional adapters.

The core application must work without them.

Use an interface such as:

```python
TelemetrySink
```

Implement:

```text
LocalSink
AwsIoTSink
SiteWiseSink
```

Default:

```text
LocalSink
```

Do not require AWS credentials for:

```bash
python main.py --demo
```

---

# 33. Emerson/NI Integration Boundary

Do not attempt to reproduce proprietary Emerson/NI functionality.

Create a conceptual interface:

```python
IndustrialDataSource
```

Stage 2 implementation:

```text
SimulatedIndustrialDataSource
```

A future Stage 3 adapter can connect to real industrial infrastructure.

---

# 34. GPT-5 Integration Boundary

The agent must depend on tool interfaces, not internal implementation details.

Correct:

```text
GPT-5
  |
  +--> get_transformer_status()
  +--> get_asset_history()
  +--> get_anomaly_details()
  +--> run_scenario()
```

Incorrect:

```text
GPT-5 directly reads random CSV files
GPT-5 directly calculates electrical physics
GPT-5 invents asset state
```

The core application must run without the GPT-5 API.

---

# 35. Agent System Prompt Requirements

Use a system prompt with these rules:

```text
You are the Edge-Guardian engineering assistant.

Analyze transformer condition using structured system tools.

Do not invent measurements.

Use tool results as the source of truth for current asset state.

Distinguish:
- simulated/ingested telemetry
- physics-derived values
- ML predictions
- model-based RUL
- recommendations

Do not claim the Edge-Guardian Health Index or RUL is industry-certified.

Do not issue breaker commands or safety-critical control actions.

When evidence is insufficient, state what is missing.
```

---

# 36. Required Demo Story

The implementation must support exactly this primary story:

```text
1. Fleet starts healthy.
2. T023 receives a feeder resistance increase.
3. Electrical losses increase.
4. Efficiency decreases.
5. Temperature/degradation changes.
6. Physics features move outside normal behavior.
7. TinyML detects anomaly.
8. Classifier identifies the modeled fault.
9. Health Index decreases.
10. RUL decreases.
11. Digital Shadow updates.
12. Fleet dashboard highlights T023.
13. Engineer asks:
       "Why is T023 unhealthy?"
14. GPT-5 retrieves evidence using tools.
15. GPT-5 explains the chain of evidence.
16. Engineer asks:
       "What if T023 load increases another 20%?"
17. Agent calls scenario simulator.
18. Scenario result returns without mutating real asset state.
19. Agent explains projected impact.
20. Agent generates a draft maintenance plan.
```

---

# 37. Acceptance Criteria

## Core

- [ ] Local application runs.
- [ ] No hardware required.
- [ ] No AWS credentials required for main demo.
- [ ] Virtual transformers can be generated.
- [ ] At least 20 assets can run.
- [ ] Normal telemetry is physically consistent enough for the chosen simplified model.
- [ ] Fault scenarios are parameterized.
- [ ] Physics calculations work.
- [ ] Seven-feature vector works.
- [ ] Autoencoder anomaly detection works.
- [ ] Fault classifier works.
- [ ] Health Index works.
- [ ] RUL works as a model-based estimate.
- [ ] Digital Shadow works.
- [ ] Fleet aggregation works.
- [ ] What-if simulation works.
- [ ] Dashboard works.
- [ ] End-to-end demo works.

## Agent

- [ ] Tools return structured state.
- [ ] Agent does not invent measurements.
- [ ] Agent explains a known anomaly.
- [ ] Agent can call scenario simulation.
- [ ] Agent produces a draft maintenance plan.
- [ ] Agent cannot directly control safety-critical equipment.

## Quality

- [ ] Unit tests pass.
- [ ] End-to-end test passes.
- [ ] README explains setup.
- [ ] Model training is reproducible.
- [ ] Configuration is documented.
- [ ] No secrets are committed.
- [ ] Synthetic/model-based claims are clearly labeled.

---

# 38. Implementation Order — Follow This Order

## Phase 1 — Foundation

1. Create repository structure.
2. Create configuration.
3. Create typed data models.
4. Create logging.
5. Create test scaffolding.

## Phase 2 — Physics

6. Implement electrical equations.
7. Implement efficiency calculations.
8. Implement thermal model.
9. Write physics tests.

## Phase 3 — Simulation

10. Implement transformer configuration.
11. Implement normal telemetry generator.
12. Implement fault scenarios.
13. Generate datasets.
14. Write generator/scenario tests.

## Phase 4 — Edge AI

15. Implement feature extraction.
16. Implement preprocessing.
17. Train autoencoder.
18. Implement anomaly scoring.
19. Train classifier.
20. Implement UNKNOWN confidence handling.
21. Write AI tests.

## Phase 5 — Health

22. Implement Health Index.
23. Implement degradation state.
24. Implement RUL.
25. Write health/RUL tests.

## Phase 6 — Digital Shadow

26. Implement asset shadow.
27. Implement history.
28. Implement fleet manager.
29. Write digital-shadow tests.

## Phase 7 — Scenario Engine

30. Implement what-if engine.
31. Ensure scenario execution does not mutate real state.
32. Write scenario tests.

## Phase 8 — Dashboard

33. Build fleet dashboard.
34. Build asset detail.
35. Build fault injection.
36. Build what-if view.

## Phase 9 — Agent

37. Implement tool layer.
38. Implement GPT-5 integration.
39. Implement system prompt.
40. Implement maintenance-plan generation.
41. Test tool-driven investigation.

## Phase 10 — Cloud adapters

42. Add AWS interfaces.
43. Add optional AWS implementation.
44. Keep local execution independent.

## Phase 11 — Integration

45. Implement `main.py --demo`.
46. Run end-to-end tests.
47. Fix numerical/logic inconsistencies.
48. Produce final demo dataset.
49. Update README.
50. Run the final acceptance checklist.

---

# 39. Explicitly Do NOT Implement

Do not:

- build physical hardware drivers
- require ESP32
- require CT/VT wiring
- build a real protection relay
- implement breaker control
- make AWS mandatory
- make GPT-5 mandatory for Edge AI inference
- replace physics with an LLM
- use random numbers as the entire simulation model
- claim synthetic RUL is real-world validated
- claim Health Index is an industry standard
- build a huge deep-learning architecture
- build 3D factory visualization before the core pipeline works
- create a multi-agent architecture
- add WhatsApp automation
- create autonomous real-world maintenance dispatch
- mutate real Digital Shadow during what-if simulation
- allow the LLM to fabricate missing telemetry
- over-engineer before the end-to-end demo works

---

# 40. Definition of Done

This command:

```bash
python main.py --demo
```

must run from a clean environment and demonstrate:

```text
Virtual fleet
      |
      v
Physics
      |
      v
Features
      |
      v
Edge AI
      |
      v
Health + RUL
      |
      v
Digital Shadow
      |
      v
Fleet intelligence
      |
      v
What-if simulation
      |
      v
GPT-5 engineering assistant
```

with **T023** as the primary fault-injection example.

---

# 41. Final Architecture Contract

Preserve this separation:

```text
                 EDGE-GUARDIAN
                       |
       +---------------+---------------+
       |                               |
   PHYSICAL/OT                     VIRTUAL
       |                               |
       +---------------+---------------+
                       |
                DATA ACQUISITION
                       |
                       v
                PHYSICS ENGINE
                       |
                       v
              FEATURE EXTRACTION
                       |
                       v
                 EDGE AI
                /       \
               /         \
        ANOMALY       CLASSIFIER
               \         /
                \       /
                 HEALTH
                    |
                    v
                   RUL
                    |
                    v
             DIGITAL SHADOW
                    |
             +------+------+
             |             |
             v             v
           FLEET        SIMULATOR
             |             |
             +------+------+
                    |
                    v
             AGENT TOOLS
                    |
                    v
                  GPT-5
                    |
                    v
                ENGINEER
```

The central engineering principle is:

> **Physics creates trustworthy features. Edge AI detects abnormal behavior. Health/RUL converts that behavior into asset condition. Digital Shadows organize the state. Simulation tests possible futures. GPT-5 explains and orchestrates the information for engineers.**
