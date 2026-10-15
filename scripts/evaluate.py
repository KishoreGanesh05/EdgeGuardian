"""
Edge-Guardian Model Evaluation & Metrics Report.

Produces evidence-based, held-out performance metrics for the two Edge AI
models, with NO train/test contamination:

    Anomaly detector (autoencoder)
        - trained ONLY on normal-train
        - threshold set ONLY on normal-validation (configured percentile)
        - evaluated on a MIXED held-out test set (normal-test + fault-test)
        - metrics: confusion matrix, precision/recall/F1, accuracy, ROC-AUC,
          false-positive rate, fault detection rate, score separation

    Fault classifier (RandomForest)
        - trained ONLY on fault-train (stratified)
        - evaluated on held-out fault-test
        - metrics: overall accuracy, per-class precision/recall/F1/support,
          confusion matrix, and UNKNOWN-abstention rate at the configured
          confidence threshold

Everything is seeded for reproducibility. Outputs:
    reports/metrics.json   (machine-readable)
    reports/metrics.md     (human-readable tables for the deck/README)

Run:
    python scripts/evaluate.py
    python scripts/evaluate.py --fleet 15 --normal-seconds 120 --fault-seconds 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is importable when run as `python scripts/evaluate.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config.settings import (
    ANOMALY_THRESHOLD_PERCENTILE,
    AUTOENCODER_BATCH_SIZE,
    AUTOENCODER_ENCODING_DIM,
    AUTOENCODER_HIDDEN_DIM,
    AUTOENCODER_LEARNING_RATE,
    CLASSIFIER_UNKNOWN_CONFIDENCE,
    FEATURE_COUNT,
    FEATURE_ORDER,
    PROJECT_ROOT,
    setup_logging,
)
from data.models import FaultScenario, FeatureVector
from data.generator import (
    generate_fleet_configs,
    generate_telemetry,
    calculate_physics,
    extract_feature_vector,
)
from edge_ai.preprocessing import Preprocessor
from edge_ai.anomaly import AnomalyDetector
from edge_ai.classifier import FaultClassifier

logger = logging.getLogger("edge_guardian.evaluate")

_FAULT_SCENARIOS = [
    FaultScenario.FEEDER_RESISTANCE_INCREASE,
    FaultScenario.OVERLOAD,
    FaultScenario.THERMAL_STRESS,
    FaultScenario.EFFICIENCY_DEGRADATION,
]


# ── Preprocessor helpers (use the public FeatureVector API, no core changes) ─

def _as_feature_vectors(arr: np.ndarray) -> List[FeatureVector]:
    """Convert an (N, 7) array into FeatureVector objects for the Preprocessor."""
    return [FeatureVector.from_array(row) for row in arr]


def _fit_scaler(pp: Preprocessor, arr: np.ndarray) -> None:
    """Fit the preprocessor on an (N, 7) array via the public API."""
    pp.fit(_as_feature_vectors(arr))


def _scale(pp: Preprocessor, arr: np.ndarray) -> np.ndarray:
    """Scale an (N, 7) array to (N, 7) using the fitted preprocessor."""
    return pp.transform_batch(_as_feature_vectors(arr))


# ═══════════════════════════════════════════════════════════════════════════
# Dataset construction with explicit, leakage-free splits
# ═══════════════════════════════════════════════════════════════════════════

def _features_for(config, scenario, duration_s, seed) -> np.ndarray:
    """Generate telemetry for one asset+scenario and return an (N, 7) array."""
    samples = generate_telemetry(
        config=config,
        duration_seconds=duration_s,
        sample_interval_seconds=1,
        seed=seed,
        scenario=scenario,
    )
    rows = []
    for s in samples:
        physics = calculate_physics(s, config)
        fv = extract_feature_vector(s, physics)
        rows.append(fv.to_array())
    return np.asarray(rows, dtype=np.float64)


def build_datasets(
    fleet_size: int,
    normal_seconds: int,
    fault_seconds: int,
    seed: int,
) -> Dict:
    """
    Build normal and fault feature matrices and split them with NO leakage.

    Normal split (per the plan's anomaly-detector protocol):
        normal_train / normal_val / normal_test  = 70 / 15 / 15

    Fault split (per the plan's classifier protocol):
        fault_train / fault_test = 70 / 30, stratified per class
    """
    configs = generate_fleet_configs(count=fleet_size, seed=seed)
    rng = np.random.default_rng(seed)

    # ── Normal samples across the whole fleet ───────────────────────────
    normal_rows: List[np.ndarray] = []
    for i, cfg in enumerate(configs):
        normal_rows.append(_features_for(cfg, FaultScenario.NORMAL, normal_seconds, seed + i))
    X_normal = np.vstack(normal_rows)
    rng.shuffle(X_normal)

    n = X_normal.shape[0]
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    normal_train = X_normal[:n_train]
    normal_val = X_normal[n_train:n_train + n_val]
    normal_test = X_normal[n_train + n_val:]

    # ── Fault samples (subset of fleet, all 4 scenarios) ────────────────
    fault_subset = configs[: max(6, fleet_size // 2)]
    fault_X_by_class: Dict[str, np.ndarray] = {}
    for sc in _FAULT_SCENARIOS:
        rows = []
        for j, cfg in enumerate(fault_subset):
            rows.append(_features_for(cfg, sc, fault_seconds, seed + 1000 + j))
        fault_X_by_class[sc.value] = np.vstack(rows)

    # Stratified 70/30 split per class
    fault_train_X, fault_train_y = [], []
    fault_test_X, fault_test_y = [], []
    for label, X in fault_X_by_class.items():
        Xc = X.copy()
        rng.shuffle(Xc)
        cut = int(0.70 * Xc.shape[0])
        fault_train_X.append(Xc[:cut])
        fault_test_X.append(Xc[cut:])
        fault_train_y += [label] * cut
        fault_test_y += [label] * (Xc.shape[0] - cut)

    return {
        "normal_train": normal_train,
        "normal_val": normal_val,
        "normal_test": normal_test,
        "fault_train_X": np.vstack(fault_train_X),
        "fault_train_y": np.asarray(fault_train_y),
        "fault_test_X": np.vstack(fault_test_X),
        "fault_test_y": np.asarray(fault_test_y),
        "n_assets": fleet_size,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly-detector evaluation (binary: fault = positive)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_anomaly_detector(
    pp: Preprocessor,
    ds: Dict,
    epochs: int,
) -> Tuple[AnomalyDetector, Dict]:
    """Train AE on normal-train, threshold on normal-val, test on mixed set."""
    X_train = _scale(pp, ds["normal_train"])
    X_val = _scale(pp, ds["normal_val"])
    X_normal_test = _scale(pp, ds["normal_test"])
    X_fault_test = _scale(pp, ds["fault_test_X"])

    detector = AnomalyDetector(
        input_dim=FEATURE_COUNT,
        hidden_dim=AUTOENCODER_HIDDEN_DIM,
        encoding_dim=AUTOENCODER_ENCODING_DIM,
    )
    # Train with NO validation split here — we hold out a real normal-val set.
    detector.train(
        X_train,
        epochs=epochs,
        batch_size=AUTOENCODER_BATCH_SIZE,
        learning_rate=AUTOENCODER_LEARNING_RATE,
        threshold_percentile=ANOMALY_THRESHOLD_PERCENTILE,
        validation_split=0.0,
    )

    # Re-set the threshold using the held-out normal VALIDATION set (not train).
    val_scores, _ = detector.predict(X_val)
    threshold = float(np.percentile(val_scores, ANOMALY_THRESHOLD_PERCENTILE))
    detector._threshold = threshold  # set from validation, leakage-free

    # Mixed held-out test: normal (y=0) + fault (y=1)
    normal_scores, _ = detector.predict(X_normal_test)
    fault_scores, _ = detector.predict(X_fault_test)

    # Per-scenario detection rate (fault_scores aligns with fault_test_y order).
    fault_labels = ds["fault_test_y"]
    per_scenario = {}
    for label in sorted(set(fault_labels.tolist())):
        mask = fault_labels == label
        detected = float(np.mean(fault_scores[mask] > threshold)) if mask.any() else 0.0
        per_scenario[label] = {
            "detection_rate": round(detected, 4),
            "mean_score": round(float(np.mean(fault_scores[mask])), 6) if mask.any() else 0.0,
            "support": int(mask.sum()),
        }

    y_true = np.concatenate([
        np.zeros(len(normal_scores), dtype=int),
        np.ones(len(fault_scores), dtype=int),
    ])
    scores = np.concatenate([normal_scores, fault_scores])
    y_pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "threshold": round(threshold, 6),
        "threshold_percentile": ANOMALY_THRESHOLD_PERCENTILE,
        "threshold_source": "held-out normal validation set",
        "test_counts": {
            "normal_test": int(len(normal_scores)),
            "fault_test": int(len(fault_scores)),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": round(float(precision), 4),
        "recall_detection_rate": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(roc_auc, 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "per_scenario_detection_rate": per_scenario,
        "score_separation": {
            "normal_mean": round(float(np.mean(normal_scores)), 6),
            "normal_p95": round(float(np.percentile(normal_scores, 95)), 6),
            "fault_mean": round(float(np.mean(fault_scores)), 6),
            "fault_median": round(float(np.median(fault_scores)), 6),
        },
    }
    return detector, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Classifier evaluation (multiclass) on held-out fault-test
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_classifier(pp: Preprocessor, ds: Dict) -> Tuple[FaultClassifier, Dict]:
    """Train classifier on fault-train, evaluate on held-out fault-test."""
    X_train = _scale(pp, ds["fault_train_X"])
    X_test = _scale(pp, ds["fault_test_X"])
    y_train = ds["fault_train_y"]
    y_test = ds["fault_test_y"]

    clf = FaultClassifier(unknown_threshold=CLASSIFIER_UNKNOWN_CONFIDENCE)
    clf.train(X_train, y_train)
    labels = list(clf.classes)

    # Raw predictions (ignore UNKNOWN abstention) for held-out accuracy.
    raw_pred = clf._classifier.predict(X_test)
    overall_acc = accuracy_score(y_test, raw_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, raw_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_test, raw_pred, labels=labels).tolist()

    # UNKNOWN abstention rate at the configured confidence threshold.
    thresholded = clf.predict_batch(X_test)
    unknown = sum(1 for cls, _ in thresholded if cls == "UNKNOWN")
    # Accuracy counting UNKNOWN as "not correct" (conservative, deployment view).
    thresh_correct = sum(
        1 for (cls, _), truth in zip(thresholded, y_test) if cls == truth
    )
    thresholded_acc = thresh_correct / len(y_test) if len(y_test) else 0.0

    metrics = {
        "labels": labels,
        "test_count": int(len(y_test)),
        "overall_accuracy": round(float(overall_acc), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "unknown_confidence_threshold": CLASSIFIER_UNKNOWN_CONFIDENCE,
        "unknown_abstention_rate": round(unknown / len(y_test), 4) if len(y_test) else 0.0,
        "thresholded_accuracy": round(float(thresholded_acc), 4),
        "feature_importances": {
            k: round(float(v), 4) for k, v in clf.get_feature_importances().items()
        },
    }
    return clf, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Report rendering
# ═══════════════════════════════════════════════════════════════════════════

def _md_table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(meta: Dict, anom: Dict, clf: Dict) -> str:
    cm = anom["confusion_matrix"]
    sep = anom["score_separation"]
    lines = [
        "# Edge-Guardian — Model Evaluation Report",
        "",
        f"_Generated: {meta['generated_utc']} · seed={meta['seed']} · "
        f"fleet={meta['n_assets']} assets_",
        "",
        "All metrics below are on **held-out data** with no train/test "
        "contamination. The anomaly threshold is fixed from a separate normal "
        "**validation** set, then applied unchanged to the test set.",
        "",
        "> Health Index and RUL are project-defined, model-based Stage 2 "
        "metrics — not industry-certified. These results characterize the "
        "synthetic-data models, not a field-validated system.",
        "",
        "## 1. Anomaly Detector (autoencoder, unsupervised)",
        "",
        f"- Threshold: **{anom['threshold']}** "
        f"(P{anom['threshold_percentile']} of {anom['threshold_source']})",
        f"- Test set: {anom['test_counts']['normal_test']} normal + "
        f"{anom['test_counts']['fault_test']} fault samples",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["ROC-AUC", anom["roc_auc"]],
                ["Precision", anom["precision"]],
                ["Recall (fault detection rate)", anom["recall_detection_rate"]],
                ["F1", anom["f1"]],
                ["Accuracy", anom["accuracy"]],
                ["False-positive rate (on normal)", anom["false_positive_rate"]],
            ],
        ),
        "",
        "**Confusion matrix** (positive = fault / anomaly):",
        "",
        _md_table(
            ["", "Pred: Normal", "Pred: Anomaly"],
            [
                ["Actual: Normal", cm["true_negative"], cm["false_positive"]],
                ["Actual: Fault", cm["false_negative"], cm["true_positive"]],
            ],
        ),
        "",
        "**Per-scenario detection rate** (fraction of each fault type caught "
        "at the fixed threshold):",
        "",
        _md_table(
            ["Fault scenario", "Detection rate", "Mean score", "Support"],
            [
                [label, m["detection_rate"], m["mean_score"], m["support"]]
                for label, m in anom["per_scenario_detection_rate"].items()
            ],
        ),
        "",
        "**Score separation** (reconstruction-error MSE):",
        "",
        _md_table(
            ["Group", "Mean", "Other"],
            [
                ["Normal (test)", sep["normal_mean"], f"P95 = {sep['normal_p95']}"],
                ["Fault (test)", sep["fault_mean"], f"median = {sep['fault_median']}"],
            ],
        ),
        "",
        "## 2. Fault Classifier (RandomForest, supervised)",
        "",
        f"- Held-out fault-test samples: **{clf['test_count']}**",
        f"- Overall accuracy (raw): **{clf['overall_accuracy']}**",
        f"- Accuracy with UNKNOWN abstention (conf < "
        f"{clf['unknown_confidence_threshold']}): **{clf['thresholded_accuracy']}** "
        f"(abstention rate {clf['unknown_abstention_rate']})",
        "",
        "**Per-class performance:**",
        "",
        _md_table(
            ["Fault class", "Precision", "Recall", "F1", "Support"],
            [
                [label, m["precision"], m["recall"], m["f1"], m["support"]]
                for label, m in clf["per_class"].items()
            ],
        ),
        "",
        "**Confusion matrix** (rows = actual, columns = predicted):",
        "",
        _md_table(
            [""] + clf["confusion_matrix"]["labels"],
            [
                [clf["confusion_matrix"]["labels"][i]] + row
                for i, row in enumerate(clf["confusion_matrix"]["matrix"])
            ],
        ),
        "",
        "**Feature importances** (classifier):",
        "",
        _md_table(
            ["Feature", "Importance"],
            [[k, v] for k, v in clf["feature_importances"].items()],
        ),
        "",
        "## 3. Reproducibility",
        "",
        f"- Fixed seed: `{meta['seed']}`",
        f"- Feature order: `{', '.join(FEATURE_ORDER)}`",
        "- Splits: normal 70/15/15 (train/val/test), fault 70/30 "
        "(train/test, stratified per class)",
        f"- Regenerate: `python scripts/evaluate.py --seed {meta['seed']}`",
        "",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(fleet: int, normal_seconds: int, fault_seconds: int, seed: int, epochs: int) -> Dict:
    logger.info("Building datasets (fleet=%d, seed=%d)...", fleet, seed)
    ds = build_datasets(fleet, normal_seconds, fault_seconds, seed)

    # Scaler fit ONLY on normal-train (leakage-free baseline).
    pp = Preprocessor()
    _fit_scaler(pp, ds["normal_train"])

    logger.info("Evaluating anomaly detector...")
    _, anom = evaluate_anomaly_detector(pp, ds, epochs=epochs)

    logger.info("Evaluating fault classifier...")
    _, clf = evaluate_classifier(pp, ds)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "seed": seed,
        "n_assets": fleet,
        "normal_seconds": normal_seconds,
        "fault_seconds": fault_seconds,
        "epochs": epochs,
        "split_counts": {
            "normal_train": int(ds["normal_train"].shape[0]),
            "normal_val": int(ds["normal_val"].shape[0]),
            "normal_test": int(ds["normal_test"].shape[0]),
            "fault_train": int(ds["fault_train_X"].shape[0]),
            "fault_test": int(ds["fault_test_X"].shape[0]),
        },
    }

    report = {"meta": meta, "anomaly_detector": anom, "fault_classifier": clf}

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "metrics.md").write_text(render_markdown(meta, anom, clf), encoding="utf-8")

    logger.info("Wrote reports/metrics.json and reports/metrics.md")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Edge-Guardian model evaluation")
    parser.add_argument("--fleet", type=int, default=15, help="Fleet size (default 15).")
    parser.add_argument("--normal-seconds", type=int, default=120,
                        help="Normal telemetry seconds per asset (default 120).")
    parser.add_argument("--fault-seconds", type=int, default=60,
                        help="Fault telemetry seconds per asset/scenario (default 60).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Autoencoder training epochs (default 60).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    report = run(args.fleet, args.normal_seconds, args.fault_seconds, args.seed, args.epochs)

    a = report["anomaly_detector"]
    c = report["fault_classifier"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (held-out)")
    print("=" * 60)
    print(f"Anomaly detector : ROC-AUC={a['roc_auc']}  precision={a['precision']}  "
          f"recall={a['recall_detection_rate']}  F1={a['f1']}  FPR={a['false_positive_rate']}")
    print(f"Fault classifier : accuracy={c['overall_accuracy']}  "
          f"(abstention-adjusted={c['thresholded_accuracy']})  classes={len(c['labels'])}")
    print("Reports written  : reports/metrics.md, reports/metrics.json")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
