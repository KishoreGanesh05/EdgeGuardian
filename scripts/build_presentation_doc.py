"""
Generate the Edge-Guardian Stage 2 presentation CONTENT as a .docx, mapped
1:1 to the InnoVent-27 Stage 2 template (slides 2-13 + 15).

This is a content draft to paste into the official .pptx template. It follows
the template's brand rules (Arial, title/body sizes, brand colors) and keeps
every quantitative claim consistent with reports/metrics.md and the running
`python main.py --demo`.

Run:  python scripts/build_presentation_doc.py
Out:  Edge-Guardian_Stage2_Presentation_Content.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "Edge-Guardian_Stage2_Presentation_Content.docx"

# Brand palette (from the template guidelines slide)
BLUE = RGBColor(0x00, 0x00, 0xB3)
TEAL = RGBColor(0x12, 0xC6, 0xB3)
ORANGE = RGBColor(0xFF, 0x9C, 0x00)
BLACK = RGBColor(0x00, 0x00, 0x00)
GRAY = RGBColor(0x66, 0x66, 0x66)


def _set_base_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    style.font.color.rgb = BLACK


def slide_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.space_before = Pt(18)
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(18)
    run.font.color.rgb = BLUE


def subhead(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(12)
    run.font.color.rgb = TEAL


def bullet(doc: Document, text: str, level: int = 0, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.name = "Arial"
        r.font.size = Pt(11)
    r2 = p.add_run(text)
    r2.font.name = "Arial"
    r2.font.size = Pt(11)


def body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(11)


def note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.name = "Arial"
    r.font.size = Pt(9.5)
    r.font.color.rgb = GRAY


def todo(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run("[TEAM TO SUPPLY] ")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = ORANGE
    r2 = p.add_run(text)
    r2.font.name = "Arial"
    r2.font.size = Pt(10)
    r2.font.color.rgb = GRAY


def table(doc: Document, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = "Arial"
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLUE
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.name = "Arial"
            run.font.size = Pt(9.5)
    doc.add_paragraph()
    return t


def hr(doc: Document) -> None:
    p = doc.add_paragraph()
    r = p.add_run("—" * 40)
    r.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    r.font.size = Pt(8)


def build() -> None:
    doc = Document()
    _set_base_font(doc)

    # ── Cover ────────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Edge-Guardian — Stage 2 Presentation Content")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(22)
    r.font.color.rgb = BLUE
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Draft to paste into the InnoVent-27 Stage 2 template "
                    "(slides 2–13, 15). Arial · brand colors · ≤10-min budget.")
    r.italic = True
    r.font.size = Pt(11)
    r.font.color.rgb = GRAY
    note(doc, "Format rules: Arial; Title 28–32pt, Body 16–18pt, Captions 10–12pt. "
              "Colors #0000B3 / #12C6B3 / #FF9C00 / #000000. 10 min present + 5 min Q&A, "
              "online (MS Teams). Keep slide 1 (guidelines) out of the final deck.")
    note(doc, "All metrics below match reports/metrics.md and `python main.py --demo` "
              "(reproducible via scripts/evaluate.py).")
    doc.add_page_break()

    # ── SLIDE 2 — Title ──────────────────────────────────────────────────
    slide_title(doc, "SLIDE 2 — Title")
    bullet(doc, "Edge-Guardian", bold_lead="Project name: ")
    bullet(doc, "Physics-Gated Edge AI for Transformer & Feeder Condition Intelligence",
           bold_lead="Tagline: ")
    bullet(doc, "Track 3.2.2.2 — Edge AI · Co-sponsored by Tata Technologies · AWS · Emerson NI")
    todo(doc, "Month/Year of presentation, and your team/college name on the title slide.")
    hr(doc)

    # ── SLIDE 3 — Team ───────────────────────────────────────────────────
    slide_title(doc, "SLIDE 3 — Team Members Overview")
    todo(doc, "College name.")
    todo(doc, "Team leader + each member: Name, Branch, Photo, Role in project "
              "(e.g., Physics/Simulation, Edge AI/ML, Dashboard/Agent, Hardware/Stage-3).")
    note(doc, "Suggested role split to show breadth: Simulation & Physics · Edge AI/TinyML · "
              "Digital Shadow & Dashboard · Agent & Cloud · Hardware (Stage 3).")
    hr(doc)

    # ── SLIDE 4 — Agenda (fixed) ─────────────────────────────────────────
    slide_title(doc, "SLIDE 4 — Agenda (keep template list)")
    for item in ["Problem Statement", "Objective and Approach", "Solution Overview",
                 "Solution Architecture", "Challenges Faced", "Technical Implementation",
                 "Results and Achievements", "Demonstration video",
                 "Project plan for POC completion (Stage-3, Jan 2027)"]:
        bullet(doc, item)
    hr(doc)

    # ── SLIDE 5 — Problem Statement ──────────────────────────────────────
    slide_title(doc, "SLIDE 5 — Problem Statement")
    subhead(doc, "The problem")
    bullet(doc, "Transformer and feeder failures are detectable weeks ahead, yet still go "
                "unprevented. The gap is not sensing — it is turning telemetry into asset-level "
                "diagnosis, health, and remaining-useful-life.")
    bullet(doc, "Where data already exists (advanced DAQ / SCADA / historians), it is siloed in "
                "vendor-locked systems, used for threshold alerting rather than fault "
                "diagnosis/prognosis, and often cloud-dependent — which air-gapped OT networks reject.")
    bullet(doc, "Where data does not exist (the long tail of smaller units), there is no "
                "asset-level monitoring at all.")
    subhead(doc, "Market size & relevance")
    bullet(doc, "~13 million distribution transformers in India; ~10% annual failure (~1.3M/yr); "
                "each failure can cost a facility 5–10% of monthly revenue.")
    bullet(doc, "Primary segment: data-rich industrial plants & utilities that already collect "
                "telemetry but lack physics-grounded, explainable asset intelligence.")
    bullet(doc, "Long-tail segment: 63M MSMEs with little/no instrumentation; RDSS (₹12,585 cr, "
                "FY25) funds metering at scale; peak demand 243 → 400 GW by 2030.")
    bullet(doc, "Sustainability: one unhealthy transformer wastes ~71 tonnes CO₂/yr.")
    note(doc, "Figures from your Stage-1 pre-read — keep primary sources handy; jurors verify.")
    hr(doc)

    # ── SLIDE 6 — Objective & Approach ───────────────────────────────────
    slide_title(doc, "SLIDE 6 — Objective and Approach")
    subhead(doc, "Objective")
    bullet(doc, "A data-source-agnostic asset-intelligence layer that turns a site's EXISTING "
                "transformer/feeder telemetry into physics-gated edge AI diagnostics, health & RUL — "
                "vendor-neutral, deployable on existing OT/edge infrastructure, able to run offline.")
    bullet(doc, "Where no instrumentation exists, an optional low-cost edge node provides the same "
                "telemetry — extending coverage to the long tail with no change to the software.")
    subhead(doc, "Approach (physics first, AI second)")
    body(doc, "Any data source → PHYSICS → 7-feature vector → Edge AI (anomaly + fault class) "
              "→ Health Index → RUL → Digital Shadow → Fleet · What-If · Agent")
    bullet(doc, "The 7-feature physics contract is the universal interface: any source that can "
                "provide the raw electrical/thermal channels plugs in unchanged.")
    bullet(doc, "Physics converts raw measurements into trustworthy features; the AI never sees raw "
                "samples and never controls safety-critical equipment.")
    subhead(doc, "Why this approach (vs alternatives)")
    table(doc,
          ["Approach", "Limitation", "Edge-Guardian"],
          [["Vendor DAQ/SCADA alerting", "Thresholds only, siloed, vendor-locked", "Physics + AI diagnosis & prognosis"],
           ["Cloud-only analytics", "Rejected by air-gapped OT", "Runs on existing edge, offline"],
           ["Raw-telemetry ML", "Noisy, hard to trust/explain", "Physics-gated features"],
           ["Pure-LLM 'AI monitor'", "Fabricates, unsafe for control", "Evidence-only, deterministic core"]])
    subhead(doc, "Phases")
    bullet(doc, "Stage 1: concept → Stage 2: software POC (this submission, held-out metrics) → "
                "Stage 3: integrate a real data feed + edge-gateway deployment, pilot validation (Jan 2027).")
    hr(doc)

    # ── SLIDE 7 — Solution Overview ──────────────────────────────────────
    slide_title(doc, "SLIDE 7 — Solution Overview")
    subhead(doc, "A pluggable intelligence layer (Stage 2 implements the software pipeline)")
    bullet(doc, "Data acquisition (pluggable) — ingests existing DAQ/SCADA/historian/SiteWise feeds, "
                "or an optional low-cost node where none exist. Stage-2 emulates this via a simulated "
                "industrial data source.")
    bullet(doc, "TinyML fault detection — autoencoder anomaly score on a clean 7-value feature vector.")
    bullet(doc, "Remaining Useful Life — model-based degradation projection to a critical threshold.")
    bullet(doc, "Energy-loss analytics — differential efficiency / loss per asset.")
    bullet(doc, "Asset Digital Shadow — authoritative live state + bounded health/anomaly history.")
    bullet(doc, "Agentic assistant — evidence-only engineering Q&A, draft maintenance plans.")
    subhead(doc, "Unique aspects")
    bullet(doc, "Data-source-agnostic: the 7-feature physics contract decouples the AI from any "
                "specific sensor, DAQ, or vendor.")
    bullet(doc, "Fixed 7-feature contract: [RMS_V, RMS_I, Power, Power_Factor, Efficiency, "
                "Temperature, Eff_Delta].")
    bullet(doc, "What-if simulator projects futures without mutating live state; deterministic "
                "fallback agent needs no GPT-5 API and no network.")
    hr(doc)

    # ── SLIDE 8 — Technical Implementation ───────────────────────────────
    slide_title(doc, "SLIDE 8 — Technical Implementation")
    subhead(doc, "Stack")
    bullet(doc, "Python 3.12 · NumPy · Pydantic (typed data contracts) · scikit-learn "
                "(RandomForest classifier) · TensorFlow/Keras + TFLite (autoencoder) · "
                "Streamlit (dashboard) · pytest (181 tests).")
    bullet(doc, "Optional, non-blocking: OpenAI SDK (GPT-5/4o agent backend) · boto3 "
                "(AWS IoT Core / SiteWise sinks).")
    subhead(doc, "How it's applied")
    bullet(doc, "Pure-function physics (electrical/thermal/efficiency) feed a single feature "
                "extractor; scaler fit on NORMAL data only; anomaly threshold set from a held-out "
                "validation set.")
    bullet(doc, "Autoencoder (7→16→4→16→7) converts to TFLite — a direct path to TFLite-Micro on "
                "ESP32-class hardware in Stage 3.")
    subhead(doc, "Difficulties resolved")
    bullet(doc, "Prevented train/test contamination with explicit normal 70/15/15 and fault 70/30 "
                "stratified splits.")
    bullet(doc, "Removed any hard dependency on an LLM via a deterministic, tool-driven agent.")
    subhead(doc, "Scalability & flexibility")
    bullet(doc, "Pluggable data sources: an IndustrialDataSource interface (SimulatedIndustrialDataSource "
                "today) accepts Stage-3 adapters for OPC-UA / Modbus / MQTT / historian / AWS SiteWise / "
                "CSV replay — with no change to the AI pipeline.")
    bullet(doc, "New asset class = new 'physics pack'; the Edge AI, Health/RUL, Digital Shadow, "
                "simulator and agent layers stay unchanged. Edge↔cloud boundary is optional.")
    hr(doc)

    # ── SLIDE 9 — Novelty ────────────────────────────────────────────────
    slide_title(doc, "SLIDE 9 — Novelty (prior art + competitor benchmarking)")
    subhead(doc, "What is novel / hard to copy")
    bullet(doc, "Physics-gated TinyML — raw data never reaches the model; a deterministic physics "
                "engine computes RMS, power factor, efficiency and thermal stress first.")
    bullet(doc, "Deterministic-AI hybrid — the AI is architecturally barred from safety-critical "
                "decisions; protection logic stays deterministic (aviation/nuclear-style isolation).")
    bullet(doc, "Data-source-agnostic deployment — one 7-feature physics contract lets the same AI "
                "run on existing DAQ/SCADA/historian feeds OR an optional low-cost node; vendor-neutral, "
                "brownfield, offline.")
    bullet(doc, "Differential physics — where feeder and secondary channels exist, the engine "
                "classifies fault TYPE (not just alerts) from the efficiency/loss differential.")
    bullet(doc, "Evidence-only agent — answers cite real tool values and never fabricate; "
                "maintenance output is a human-approved draft, not autonomous dispatch.")

    subhead(doc, "Prior-art search (public literature) — and the gap we fill")
    bullet(doc, "Transformer PHM / IoT condition monitoring is well studied but is largely "
                "server/cloud-side and parameter-threshold alerting, not on-device physics-gated ML.")
    note(doc, "• Power Transformer Prognostics & Health Management using ML — MDPI Machines, 2025. "
              "https://www.mdpi.com/2075-1702/13/2/125")
    note(doc, "• Condition Monitoring of Electrical Transformers using IoT — MDPI Applied Sciences, "
              "2024. https://www.mdpi.com/2076-3417/14/21/9690")
    bullet(doc, "TinyML autoencoder anomaly detection on MCUs is maturing, but published work "
                "targets vibration/acoustic/generic signals — not a physics-derived transformer "
                "feature vector feeding fault classification.")
    note(doc, "• Unsupervised TinyML anomaly detection for extreme industrial environments — "
              "MDPI Sensors, 2023. https://www.mdpi.com/1424-8220/23/4/2344")
    note(doc, "• Multimodal TinyML predictive maintenance (INT8 autoencoder, F1≈0.998 on Cortex-M4F) "
              "— Preprints, 2026. https://www.preprints.org/manuscript/202606.1304")
    bullet(doc, "Physics-informed / physics-guided ML for power-system fault diagnosis exists, but "
                "is predominantly offline/server-side, not edge-deployed TinyML with an evidence-only agent.")
    note(doc, "• Robust Fault Detection & Classification in Power Systems via Physics-Informed and "
              "Data-Driven Learning — arXiv:2608.08785. https://arxiv.org/abs/2608.08785")
    note(doc, "• Physics-informed, cost-aware fault classification for predictive maintenance — "
              "Nature Scientific Reports, 2026. https://www.nature.com/articles/s41598-026-64374-2")

    subhead(doc, "Competitor benchmarking")
    table(doc,
          ["Capability", "SCADA", "IoT-cloud CM", "Generic TinyML AD", "Physics-informed server ML", "Edge-Guardian"],
          [["Physics-gated features", "n/a", "✗", "✗", "✓", "✓"],
           ["On-device / on-edge TinyML", "✗", "partial", "✓", "✗", "✓"],
           ["Fully offline / air-gap OK", "partial", "✗", "✓", "✗", "✓"],
           ["Works on existing DAQ/SCADA (brownfield)", "n/a", "partial", "✗", "partial", "✓"],
           ["Fault TYPE classification", "✓", "partial", "partial", "✓", "✓"],
           ["Evidence-only explanations", "✗", "partial", "✗", "partial", "✓"],
           ["AI barred from trip logic", "n/a", "n/a", "n/a", "n/a", "✓"]])
    note(doc, "Jurors research novelty independently — keep these citations in the pre-read and be "
              "ready to state the one-line gap: 'physics-gated TinyML + deterministic-AI safety "
              "boundary + evidence-only agent, running data-source-agnostically on existing OT "
              "infrastructure, on-edge and offline.'")
    hr(doc)

    # ── SLIDE 10 — Challenges Faced ──────────────────────────────────────
    slide_title(doc, "SLIDE 10 — Challenges Faced")
    table(doc,
          ["Challenge", "Solution", "Lesson"],
          [["No real fault data for training",
            "Physics-informed synthetic generation with fixed seeds",
            "Reproducibility earns trust"],
           ["Train/test contamination risk",
            "Explicit normal 70/15/15 & fault 70/30 stratified splits; threshold from validation",
            "Report held-out numbers only"],
           ["Feeder-resistance barely detectable (0.7%)",
            "Diagnosed to the LOSS MODEL (not features); prototyped corrected copper-loss physics "
            "→ feeder 0.7%→49%, ROC-AUC 0.83→0.94 with the same 7 features",
            "Localize gaps with evidence before adding complexity"],
           ["Avoiding LLM dependency / hallucination",
            "Deterministic tool-driven fallback agent; evidence-only tools",
            "Core must run with no API/network"],
           ["Keeping AI out of the safety path",
            "Layered deterministic-AI boundary",
            "Trust boundary is the real moat"]])
    hr(doc)

    # ── SLIDE 11 — Results & Achievements ────────────────────────────────
    slide_title(doc, "SLIDE 11 — Results and Achievements (held-out, reproducible)")
    subhead(doc, "Fault classifier (RandomForest)")
    bullet(doc, "99.6% accuracy on 504 held-out samples; per-class F1 ≥ 0.99; only 2 errors "
                "(feeder ↔ thermal).")
    subhead(doc, "Anomaly detector (autoencoder)")
    bullet(doc, "Precision 0.96 · false-positive rate 4% · ROC-AUC 0.81.")
    table(doc,
          ["Fault scenario (shipped)", "Detection rate"],
          [["Efficiency degradation", "99%"],
           ["Overload", "69%"],
           ["Thermal stress", "34%"],
           ["Feeder resistance increase", "<1% (weak physical signal — documented)"]])
    subhead(doc, "Prototyped Stage-3 improvement (same 7 features)")
    table(doc,
          ["Prototyped loss-model fix", "Current → Fixed"],
          [["Feeder-resistance detection", "0.7% → 49%"],
           ["Overall ROC-AUC", "0.83 → 0.94"],
           ["Normal efficiency / FPR", "unchanged"]])
    note(doc, "Prototyped only (reports/physics_fix_experiment.md) — correcting the loss model "
              "recovers feeder detection using the SAME 7 features; the gap was the physics, not "
              "the feature count. Not part of the shipped Stage-2 numbers.")
    subhead(doc, "End-to-end demo (live, offline)")
    bullet(doc, "T023 OVERLOAD → anomaly score 17.4 (detected) → fault class OVERLOAD @ 100% "
                "confidence → health 95.9 → 56.1 → RUL ~190k → ~90k h → fleet metrics update → "
                "agent explains with evidence.")
    subhead(doc, "Engineering quality")
    bullet(doc, "181-test suite (180 passing); reproducible metrics via scripts/evaluate.py; "
                "TFLite conversion verified; fixed seeds throughout.")
    note(doc, "Full tables: reports/metrics.md and reports/metrics.json. Health Index and RUL are "
              "project-defined, model-based Stage-2 metrics — not industry-certified.")
    hr(doc)

    # ── SLIDE 12 — Demonstration ─────────────────────────────────────────
    slide_title(doc, "SLIDE 12 — Demonstration (virtual simulation)")
    subhead(doc, "What to show")
    bullet(doc, "Live `python main.py --demo` (runs offline): healthy fleet → inject fault → "
                "detection → health → RUL → digital shadow → agent explanation → what-if → draft plan.")
    bullet(doc, "Streamlit dashboard: Fleet Overview, Asset Detail with health/anomaly trends, "
                "What-If Lab (never mutates live state).")
    bullet(doc, "Agent Q&A: 'Why is T023 unhealthy?', 'What if load increases 20%?', "
                "'Generate a maintenance plan.'")
    todo(doc, "Record a 1–2 min screen-capture video of the demo + dashboard for this slide "
              "(the template requires a demonstration video / virtual simulation).")
    note(doc, "Keep the narrative aligned with the software build: the agent produces an "
              "evidence-based DRAFT work order (human-approved); autonomous WhatsApp/email dispatch "
              "is a Stage-3 roadmap item, not a Stage-2 claim.")
    hr(doc)

    # ── SLIDE 13 — Project Plan ──────────────────────────────────────────
    slide_title(doc, "SLIDE 13 — Project Plan for POC Completion (to Stage-3, Jan 2027)")
    subhead(doc, "Timeline & milestones")
    table(doc,
          ["Milestone", "Target", "Output"],
          [["Software POC + held-out metrics", "Complete (now)", "This submission, reports/"],
           ["Live data-feed integration", "Mid-December 2026", "OPC-UA/Modbus/historian or AWS SiteWise adapter + edge-gateway deploy"],
           ["Pilot on real telemetry", "Dec–Jan", "Calibrated Health/RUL on a real asset feed"],
           ["Optional low-cost node (long tail)", "Parallel track", "Node for un-instrumented sites"],
           ["Final demo", "Jan 2027", "End-to-end on real data"]])
    subhead(doc, "Resources & procurement")
    bullet(doc, "Primary: access to a real telemetry feed (DAQ/SCADA/historian export, OPC-UA/Modbus, "
                "or AWS SiteWise) + a pilot asset-owner partner; an on-prem edge gateway / industrial PC; "
                "AWS account (adapters ready, no code change needed).")
    bullet(doc, "Optional (long tail): low-cost edge node + non-invasive CT/VT sensors for sites with "
                "no existing instrumentation.")
    subhead(doc, "Future improvements & roadmap")
    bullet(doc, "Adopt the prototyped corrected loss model — feeder-resistance detection rose "
                "0.7% → 49% and overall ROC-AUC 0.83 → 0.94 using the SAME 7 features "
                "(reports/physics_fix_experiment.md).")
    bullet(doc, "Add temporal/windowed features to lift the harder faults (thermal stress, "
                "slow-onset feeder degradation).")
    bullet(doc, "Calibrate Health Index / RUL against real failure data from the pilot.")
    bullet(doc, "Cross-domain physics packs — EV traction motors (Automotive), actuators/power "
                "buses (Aerospace), drives/pumps (IHM) — same pipeline, new physics.")
    bullet(doc, "Human-approved auto work-order dispatch once field-validated.")
    hr(doc)

    # ── SLIDE 15 — Additional Information ────────────────────────────────
    slide_title(doc, "SLIDE 15 — Additional Information")
    bullet(doc, "Reproducibility: `pip install -r requirements.txt` → `python main.py --demo` "
                "(offline) → `python scripts/evaluate.py` regenerates all metrics.")
    bullet(doc, "Testing: `python -m pytest -q` (181 tests incl. mandatory end-to-end cascade).")
    bullet(doc, "Disclaimers: Health Index & RUL are model-based Stage-2 metrics, not certified; "
                "Edge-Guardian is condition-monitoring only and does not replace protection relays "
                "or perform trip/control actions.")
    note(doc, "Repo layout: data/ physics/ edge_ai/ health/ digital_shadow/ simulator/ agent/ "
              "cloud/ dashboard/ · main.py · reports/")

    try:
        doc.save(OUT)
        print(f"Wrote {OUT}")
    except PermissionError:
        alt = OUT.with_name(OUT.stem + "_updated.docx")
        doc.save(alt)
        print(f"Target locked (open in Word?). Wrote {alt} instead.")


if __name__ == "__main__":
    build()
