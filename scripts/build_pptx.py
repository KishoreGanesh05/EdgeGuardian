"""
Populate the InnoVent-27 Stage 2 template with Edge-Guardian content.

Opens the official template (preserving its branding/layouts), replaces the
guidance text in each content slide's body textbox with our content, adds
native tables where helpful, and puts prior-art citations in speaker notes.

Follows template guidelines: Arial; brand colors #0000B3 / #12C6B3 / #FF9C00.
[TEAM TO SUPPLY] placeholders mark where the team must add content.

Run:  python scripts/build_pptx.py
Out:  Edge-Guardian_Stage2.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "InnoVent-27_Stage_2_Presentation_Template.pptx"
OUT = REPO / "Edge-Guardian_Stage2.pptx"

BLUE = RGBColor(0x00, 0x00, 0xB3)
TEAL = RGBColor(0x12, 0xC6, 0xB3)
ORANGE = RGBColor(0xFF, 0x9C, 0x00)
BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ── text helpers ─────────────────────────────────────────────────────────

def _style_run(run, size, color, bold=False):
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def set_body(shape, items, base_size=14):
    """
    Fill a textbox with content. `items` is a list of (kind, text):
      kind: 'sub' (teal bold subhead), 'bul' (bullet), 'sub2' (indent bullet),
            'flow' (plain line), 'todo' (orange placeholder), 'note' (gray caption).
    """
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    first = True
    for kind, text in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if kind == "sub":
            r = p.add_run(); _style_run(r, base_size + 1, TEAL, bold=True)
            r.text = text
            p.space_before = Pt(6)
        elif kind == "bul":
            r = p.add_run(); _style_run(r, base_size, BLACK)
            r.text = "•  " + text
        elif kind == "sub2":
            r = p.add_run(); _style_run(r, base_size - 1, BLACK)
            r.text = "    –  " + text
            p.level = 1
        elif kind == "flow":
            r = p.add_run(); _style_run(r, base_size, BLACK, bold=True)
            r.text = text
        elif kind == "todo":
            r = p.add_run(); _style_run(r, base_size - 1, ORANGE, bold=True)
            r.text = "[TEAM TO SUPPLY]  "
            r2 = p.add_run(); _style_run(r2, base_size - 1, BLACK)
            r2.text = text
        elif kind == "note":
            r = p.add_run(); _style_run(r, 11, RGBColor(0x66, 0x66, 0x66))
            r.text = text
            r.font.italic = True


def set_title_text(shape, text, tagline=None):
    """Replace a title placeholder's text while preserving its styling."""
    tf = shape.text_frame
    runs = [r for para in tf.paragraphs for r in para.runs]
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r.text = ""
    else:
        tf.paragraphs[0].add_run().text = text
    if tagline:
        p = tf.add_paragraph()
        r = p.add_run(); r.text = tagline
        r.font.name = "Arial"; r.font.size = Pt(16)


def add_table(slide, left, top, width, height, headers, rows, col_widths=None, font=9):
    gf = slide.shapes.add_table(1 + len(rows), len(headers),
                                Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = gf.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for i, h in enumerate(headers):
        cell = tbl.cell(0, i)
        cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
        cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
        p = cell.text_frame.paragraphs[0]
        r = p.add_run(); r.text = h
        _style_run(r, font, WHITE, bold=True)
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri % 2 else RGBColor(0xED, 0xF6, 0xF5)
            cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            _style_run(r, font, BLACK, bold=(ci == len(row) - 1))
    return gf


def set_notes(slide, lines):
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.add_run().text = ln


def build():
    prs = Presentation(str(TEMPLATE))
    S = prs.slides

    # ── SLIDE 2 — Cover ─────────────────────────────────────────────────
    cover = S[1]
    for sh in cover.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 0:
            set_title_text(sh, "Edge-Guardian",
                           "Physics-Gated, Data-Source-Agnostic Edge AI for "
                           "Transformer & Feeder Intelligence")
        elif sh.has_text_frame and "Month" in sh.text_frame.text:
            sh.text_frame.paragraphs[0].runs and None
            tf = sh.text_frame; tf.clear()
            r = tf.paragraphs[0].add_run()
            r.text = "[Month YYYY]   ·   [Team Name]   ·   [College]"
            _style_run(r, 14, ORANGE, bold=True)

    # ── SLIDE 3 — Team ──────────────────────────────────────────────────
    team_body = S[2].shapes[2]
    set_body(team_body, [
        ("todo", "College name."),
        ("todo", "Team leader + each member: Name, Branch, Photo, Role."),
        ("note", "Suggested role split: Simulation & Physics · Edge AI/TinyML · "
                 "Digital Shadow & Dashboard · Agent & Cloud · Hardware (Stage 3)."),
    ], base_size=16)

    # ── SLIDE 5 — Problem Statement ─────────────────────────────────────
    set_body(S[4].shapes[1], [
        ("sub", "The problem"),
        ("bul", "Transformer & feeder failures are detectable weeks ahead, yet still go "
                "unprevented. The gap is turning telemetry into diagnosis, health & RUL."),
        ("bul", "Where data exists (DAQ/SCADA/historians): siloed, vendor-locked, alert-only, "
                "and often cloud-dependent — which air-gapped OT rejects."),
        ("bul", "Where it does not (the long tail of smaller units): no asset-level monitoring at all."),
        ("sub", "Market size & relevance"),
        ("bul", "~13M distribution transformers in India; ~10% annual failure (~1.3M/yr); each "
                "failure ≈ 5–10% of monthly revenue."),
        ("bul", "Primary: data-rich industrial plants & utilities. Long tail: 63M MSMEs "
                "(RDSS ₹12,585 cr, FY25). Peak demand 243 → 400 GW by 2030."),
        ("bul", "Sustainability: ~71 tonnes CO₂/yr wasted per unhealthy transformer."),
        ("note", "Figures from the Stage-1 pre-read — keep primary sources handy; jurors verify."),
    ], base_size=14)

    # ── SLIDE 6 — Objective & Approach ──────────────────────────────────
    set_body(S[5].shapes[1], [
        ("sub", "Objective"),
        ("bul", "A data-source-agnostic asset-intelligence layer: turn a site's EXISTING telemetry "
                "into physics-gated edge-AI diagnostics, health & RUL. Vendor-neutral, runs offline."),
        ("bul", "Optional low-cost node extends the same software to un-instrumented sites."),
        ("sub", "Approach — physics first, AI second, data source last"),
        ("flow", "Any data source → PHYSICS → 7-feature vector → Edge AI → Health → RUL → "
                 "Digital Shadow → Agent"),
        ("bul", "Physics builds trustworthy features; the AI never sees raw samples and never "
                "controls safety-critical equipment."),
    ], base_size=14)
    add_table(S[5], 0.28, 4.35, 12.77, 1.9,
              ["Approach", "Limitation", "Edge-Guardian"],
              [["Vendor DAQ/SCADA alerting", "Thresholds only, siloed, vendor-locked", "Physics + AI diagnosis & prognosis"],
               ["Cloud-only analytics", "Rejected by air-gapped OT", "Runs on existing edge, offline"],
               ["Raw-telemetry ML", "Noisy, hard to trust/explain", "Physics-gated features"],
               ["Pure-LLM 'AI monitor'", "Fabricates, unsafe for control", "Evidence-only, deterministic core"]],
              col_widths=[3.3, 5.2, 4.27], font=10)

    # ── SLIDE 7 — Solution Overview ─────────────────────────────────────
    set_body(S[6].shapes[1], [
        ("sub", "A pluggable intelligence layer (Stage 2 = the software pipeline)"),
        ("bul", "Data acquisition (pluggable) — existing DAQ/SCADA/historian/SiteWise, or an "
                "optional node. Stage-2 emulates this via a simulated industrial data source."),
        ("bul", "TinyML fault detection — autoencoder anomaly score on a clean 7-value feature vector."),
        ("bul", "Remaining Useful Life — model-based degradation projection to a critical threshold."),
        ("bul", "Energy-loss analytics — differential efficiency / loss per asset."),
        ("bul", "Asset Digital Shadow — authoritative live state + bounded history; fleet aggregation."),
        ("bul", "Agentic assistant — evidence-only engineering Q&A + draft maintenance plans."),
        ("sub", "Unique aspects"),
        ("bul", "Data-source-agnostic 7-feature contract decouples the AI from any sensor/DAQ/vendor; "
                "what-if simulation never mutates live state; deterministic agent needs no API/network."),
    ], base_size=13)

    # ── SLIDE 8 — Technical Implementation ──────────────────────────────
    set_body(S[7].shapes[1], [
        ("sub", "Stack"),
        ("bul", "Python 3.12 · NumPy · Pydantic (typed contracts) · scikit-learn (RandomForest) · "
                "TensorFlow/Keras + TFLite (autoencoder) · Streamlit · pytest (181 tests)."),
        ("bul", "Optional, non-blocking: OpenAI SDK (agent) · boto3 (AWS IoT / SiteWise)."),
        ("sub", "How it's applied & difficulties resolved"),
        ("bul", "Pure-function physics → one feature extractor; scaler fit on NORMAL only; anomaly "
                "threshold set from a held-out validation set (no train/test leakage)."),
        ("bul", "Autoencoder 7→16→4→16→7 converts to TFLite — a path to TFLite-Micro on an edge "
                "gateway/MCU. Deterministic fallback removes any hard LLM dependency."),
        ("sub", "Scalability & flexibility"),
        ("bul", "Pluggable IndustrialDataSource (OPC-UA/Modbus/MQTT/historian/SiteWise/CSV); a new "
                "asset class = a new 'physics pack' — the AI/Health/Shadow/agent layers stay unchanged."),
    ], base_size=13)

    # ── SLIDE 9 — Novelty ───────────────────────────────────────────────
    set_body(S[8].shapes[1], [
        ("sub", "What is novel / hard to copy"),
        ("bul", "Physics-gated TinyML — raw data never reaches the model (physics computes RMS, PF, "
                "power, thermal first)."),
        ("bul", "Deterministic-AI hybrid — AI architecturally barred from safety-critical/trip decisions."),
        ("bul", "Data-source-agnostic / brownfield — same AI on existing DAQ/SCADA OR an optional node; "
                "vendor-neutral, offline."),
        ("bul", "Differential physics classifies fault TYPE (not just alerts); evidence-only agent cites "
                "real values (human-approved draft plans)."),
        ("note", "Prior-art gap: transformer PHM/IoT CM is cloud/alert-centric; TinyML anomaly detection "
                 "is generic (vibration/acoustic); physics-informed ML is server-side — none combine all "
                 "three on existing OT data. Full citations in the speaker notes."),
    ], base_size=12.5)
    add_table(S[8], 0.28, 4.25, 12.77, 2.2,
              ["Capability", "DAQ/SCADA", "Cloud ML", "Generic TinyML", "Physics server ML", "Edge-Guardian"],
              [["Physics-gated features", "n/a", "✗", "✗", "✓", "✓"],
               ["On-edge TinyML (TFLite)", "✗", "✗", "✓", "✗", "✓"],
               ["Offline / air-gap OK", "partial", "✗", "✓", "✗", "✓"],
               ["Brownfield on existing DAQ", "✓", "partial", "✗", "partial", "✓"],
               ["Fault-TYPE diagnosis", "partial", "partial", "partial", "✓", "✓"],
               ["Evidence-only / AI off trip logic", "✗", "partial", "✗", "partial", "✓"]],
              col_widths=[3.5, 1.5, 1.4, 1.75, 2.1, 2.52], font=8.5)
    set_notes(S[8], [
        "Prior-art references (keep in pre-read):",
        "1. Power Transformer Prognostics & Health Management using ML — MDPI Machines, 2025. "
        "https://www.mdpi.com/2075-1702/13/2/125",
        "2. Condition Monitoring of Electrical Transformers using IoT — MDPI Applied Sciences, 2024. "
        "https://www.mdpi.com/2076-3417/14/21/9690",
        "3. Unsupervised TinyML anomaly detection for extreme industrial environments — MDPI Sensors, "
        "2023. https://www.mdpi.com/1424-8220/23/4/2344",
        "4. Multimodal TinyML predictive maintenance (INT8 autoencoder on Cortex-M4F) — Preprints, 2026. "
        "https://www.preprints.org/manuscript/202606.1304",
        "5. Robust Fault Detection & Classification in Power Systems via Physics-Informed and Data-Driven "
        "Learning — arXiv:2608.08785. https://arxiv.org/abs/2608.08785",
        "6. Physics-informed, cost-aware fault classification for predictive maintenance — Nature "
        "Scientific Reports, 2026. https://www.nature.com/articles/s41598-026-64374-2",
        "One-line gap: physics-gated TinyML + deterministic-AI safety boundary + evidence-only agent, "
        "running data-source-agnostically on existing OT infrastructure, on-edge and offline.",
    ])

    # ── SLIDE 10 — Challenges ───────────────────────────────────────────
    set_body(S[9].shapes[1], [
        ("note", "Challenge → solution → lesson (details below)."),
    ], base_size=12)
    add_table(S[9], 0.28, 1.5, 12.77, 3.6,
              ["Challenge", "Solution", "Lesson"],
              [["No real fault data for training", "Physics-informed synthetic generation with fixed seeds", "Reproducibility earns trust"],
               ["Train/test contamination risk", "Normal 70/15/15 & fault 70/30 stratified; threshold from validation", "Report held-out numbers only"],
               ["Feeder-resistance barely detectable (0.7%)", "Diagnosed to the loss model (not features); prototyped fix → 49%, ROC-AUC 0.83→0.94 (same 7 features)", "Localize gaps with evidence first"],
               ["Avoiding LLM dependency / hallucination", "Deterministic, tool-driven fallback agent; evidence-only tools", "Core must run with no API/network"],
               ["Keeping AI out of the safety path", "Layered deterministic-AI boundary", "The trust boundary is the real moat"]],
              col_widths=[4.0, 5.6, 3.17], font=10)

    # ── SLIDE 11 — Results ──────────────────────────────────────────────
    set_body(S[10].shapes[1], [
        ("sub", "Held-out, reproducible (reports/metrics.md · scripts/evaluate.py)"),
        ("bul", "Fault classifier: 99.6% held-out accuracy; per-class F1 ≥ 0.99 (504 samples, 2 errors)."),
        ("bul", "Anomaly detector: precision 0.96 · false-positive rate 4% · ROC-AUC 0.81."),
        ("bul", "End-to-end demo: T023 OVERLOAD → anomaly 17.4 → health 95.9→56.1 → RUL ~190k→~90k h "
                "→ fleet updates → agent explains."),
        ("bul", "Quality: 181-test suite; reproducible metrics; TFLite conversion verified; fixed seeds."),
        ("sub", "Shipped detection (left)   ·   Prototyped Stage-3 physics fix (right)"),
    ], base_size=12.5)
    # Left: shipped per-scenario detection (default severities)
    add_table(S[10], 0.28, 4.0, 6.1, 2.0,
              ["Fault scenario (shipped)", "Detection"],
              [["Efficiency degradation", "99%"],
               ["Overload", "69%"],
               ["Thermal stress", "34%"],
               ["Feeder resistance (weak signal)", "<1%"]],
              col_widths=[4.3, 1.8], font=10)
    # Right: prototyped corrected-loss-model result (same 7 features)
    add_table(S[10], 6.9, 4.0, 6.15, 2.0,
              ["Prototyped loss-model fix", "Current → Fixed"],
              [["Feeder-resistance detection", "0.7% → 49%"],
               ["Overall ROC-AUC", "0.83 → 0.94"],
               ["Normal efficiency / FPR", "unchanged"]],
              col_widths=[3.75, 2.4], font=10)
    set_notes(S[10], [
        "Health Index and RUL are project-defined, model-based Stage-2 metrics — not industry-certified.",
        "Full tables and confusion matrices: reports/metrics.md and reports/metrics.json.",
        "Right-hand table is a PROTOTYPED Stage-3 improvement (reports/physics_fix_experiment.md): "
        "correcting the loss model recovers feeder detection using the SAME 7 features — the gap "
        "was the physics, not the feature count. Not part of the shipped Stage-2 numbers.",
    ])

    # ── SLIDE 12 — Demonstration ────────────────────────────────────────
    set_body(S[11].shapes[1], [
        ("sub", "What to show (runs offline)"),
        ("bul", "Live `python main.py --demo`: healthy fleet → inject fault → detection → health → "
                "RUL → digital shadow → agent explanation → what-if → draft plan."),
        ("bul", "Streamlit dashboard: Fleet Overview · Asset Detail trends · What-If Lab "
                "(never mutates live state)."),
        ("bul", "Agent Q&A: 'Why is T023 unhealthy?' · 'What if load +20%?' · 'Generate a maintenance plan.'"),
        ("todo", "Record a 1–2 min screen-capture video of the demo + dashboard for this slide."),
        ("note", "The agent produces an evidence-based DRAFT work order (human-approved); autonomous "
                 "dispatch is a Stage-3 roadmap item, not a Stage-2 claim."),
    ], base_size=13)

    # ── SLIDE 13 — Project Plan ─────────────────────────────────────────
    set_body(S[12].shapes[1], [
        ("sub", "Timeline & milestones"),
    ], base_size=13)
    add_table(S[12], 0.28, 1.15, 12.77, 1.9,
              ["Milestone", "Target", "Output"],
              [["Software POC + held-out metrics", "Complete (now)", "This submission, reports/"],
               ["Live data-feed integration", "Mid-Dec 2026", "OPC-UA/Modbus/historian or SiteWise adapter + edge-gateway deploy"],
               ["Pilot on real telemetry", "Dec–Jan", "Calibrated Health/RUL on a real asset feed"],
               ["Optional low-cost node (long tail)", "Parallel", "Node for un-instrumented sites"],
               ["Final demo", "Jan 2027", "End-to-end on real data"]],
              col_widths=[3.9, 2.0, 6.87], font=9.5)
    body13 = S[12].shapes[1]
    # append resources/future below the (short) subhead by extending its text frame
    for kind, text in [
        ("sub", "Resources & roadmap"),
        ("bul", "Primary: a real telemetry feed (DAQ/SCADA/historian/OPC-UA/Modbus/SiteWise) + pilot "
                "partner + on-prem edge gateway + AWS (adapters ready)."),
        ("bul", "Optional (long tail): low-cost node + non-invasive CT/VT sensors."),
        ("bul", "Future: adopt the prototyped corrected loss model (feeder detection 0.7%→49%, "
                "ROC-AUC 0.83→0.94 with the same 7 features); add temporal features; calibrate "
                "Health/RUL on real data; cross-domain physics packs (Automotive/Aerospace/IHM); "
                "human-approved auto work-orders."),
    ]:
        p = body13.text_frame.add_paragraph()
        if kind == "sub":
            r = p.add_run(); _style_run(r, 14, TEAL, bold=True); r.text = text
            p.space_before = Pt(6)
        else:
            r = p.add_run(); _style_run(r, 12.5, BLACK); r.text = "•  " + text
    # nudge this block below the table
    body13.top = Inches(3.2)
    body13.height = Inches(3.6)

    # ── SLIDE 15 — Additional Information ───────────────────────────────
    s15_body = S[14].shapes.add_textbox(Inches(0.28), Inches(1.2), Inches(12.77), Inches(5.0))
    set_body(s15_body, [
        ("bul", "Reproduce: pip install -r requirements.txt → python main.py --demo (offline) → "
                "python scripts/evaluate.py (regenerates all metrics)."),
        ("bul", "Testing: python -m pytest -q (181 tests incl. mandatory end-to-end cascade)."),
        ("bul", "Disclaimers: Health Index & RUL are model-based Stage-2 metrics, not certified; "
                "Edge-Guardian is condition-monitoring only — no trip/control actions."),
        ("note", "Repo: data/ physics/ edge_ai/ health/ digital_shadow/ simulator/ agent/ cloud/ "
                 "dashboard/ · main.py · reports/"),
    ], base_size=14)

    try:
        prs.save(str(OUT))
        print(f"Wrote {OUT}")
    except PermissionError:
        alt = OUT.with_name(OUT.stem + "_updated.pptx")
        prs.save(str(alt))
        print(f"Target locked (open in PowerPoint?). Wrote {alt} instead.")


if __name__ == "__main__":
    build()
