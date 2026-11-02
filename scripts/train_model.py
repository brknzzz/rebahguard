"""Pelatihan Random Forest RebahGuard bebas data leakage (Group Split per session_id) + Benchmark Rule-Based.

Contoh penggunaan:
  python -m scripts.train_model --input data/posture_dataset.csv --model-out models/posture_rf.joblib --metrics-out models/metrics.json
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from shared.config import DetectorConfig, LANDSCAPE, PORTRAIT

FEATURE_COLUMNS = ["overhead_deg", "roll_deg", "lean_deg", "motion_rms", "gyro_rms"]
VALID_LABELS = ["POSISI_DUDUK_SEHAT", "DITARUH_TELUNGKUP", "REBAHAN_AKTIF"]
MIN_SAMPLES_PER_CLASS = 30
MIN_SESSIONS_PER_CLASS = 2


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def predict_rule_baseline_row(
    overhead_deg: float,
    roll_deg: float,
    lean_deg: float,
    motion_rms: float,
    gyro_rms: float,
    cfg: DetectorConfig = DetectorConfig(),
) -> str:
    """Klasifikasi baseline statis berbasis aturan DetectorConfig pada satu baris fitur."""
    orientation = PORTRAIT
    if cfg.use_roll:
        if cfg.flat_gate_deg <= overhead_deg <= (180.0 - cfg.flat_gate_deg):
            if lean_deg >= cfg.roll_enter_deg:
                orientation = LANDSCAPE

    th = cfg.thresholds_for(orientation)
    if (
        overhead_deg >= cfg.face_down_deg
        and motion_rms <= th.still_motion_max
        and gyro_rms <= th.still_gyro_max
    ):
        return "DITARUH_TELUNGKUP"

    if overhead_deg >= th.enter_deg:
        return "REBAHAN_AKTIF"

    return "POSISI_DUDUK_SEHAT"


def load_and_validate_dataset(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File dataset '{path}' tidak ditemukan. Rekam dataset terlebih dahulu dengan:\n"
            "  python -m scripts.collect_dataset --label POSISI_DUDUK_SEHAT --samples 40"
        )

    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = defaultdict(int)
    class_sessions: dict[str, set[str]] = defaultdict(set)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_idx, raw in enumerate(reader, start=2):
            label = (raw.get("label") or "").strip()
            session_id = (raw.get("session_id") or "").strip()
            if label not in VALID_LABELS:
                continue
            if not session_id:
                raise ValueError(f"Baris ke-{line_idx} tidak memiliki session_id yang valid.")

            try:
                parsed_row: dict[str, Any] = {
                    "session_id": session_id,
                    "timestamp_ms": int(float(raw.get("timestamp_ms") or 0)),
                    "label": label,
                }
                for col in FEATURE_COLUMNS:
                    val = float(raw[col])
                    if not np.isfinite(val):
                        raise ValueError(f"Nilai non-finite pada kolom {col}")
                    parsed_row[col] = val
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Data tidak valid pada baris ke-{line_idx}: {exc}") from exc

            rows.append(parsed_row)
            class_counts[label] += 1
            class_sessions[label].add(session_id)

    problems: list[str] = []
    for lbl in VALID_LABELS:
        n_samples = class_counts[lbl]
        n_sessions = len(class_sessions[lbl])
        if n_samples < MIN_SAMPLES_PER_CLASS or n_sessions < MIN_SESSIONS_PER_CLASS:
            problems.append(
                f"- Kelas '{lbl}': {n_samples} sampel (min {MIN_SAMPLES_PER_CLASS}), "
                f"{n_sessions} session_id unik (min {MIN_SESSIONS_PER_CLASS}). "
                f"Rekam sesi tambahan dengan: python -m scripts.collect_dataset --label {lbl} --samples 40"
            )

    if problems:
        raise ValueError(
            "Dataset belum memenuhi syarat kualitas minimum bebas data leakage:\n"
            + "\n".join(problems)
        )

    return rows


def group_split_by_session(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    """Membagi dataset berdasarkan session_id utuh tanpa overlap antara train dan test."""
    sessions_by_label: dict[str, list[str]] = {}
    session_labels: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        session_labels[r["session_id"]].add(r["label"])

    for lbl in VALID_LABELS:
        sids = sorted({r["session_id"] for r in rows if r["label"] == lbl})
        sessions_by_label[lbl] = sids

    test_session_ids: set[str] = set()
    train_session_ids: set[str] = set()

    # Prioritaskan memilih sesi eksklusif satu kelas sebagai test untuk tiap kelas
    for lbl in VALID_LABELS:
        sids = sessions_by_label[lbl]
        exclusive_sids = [s for s in sids if len(session_labels[s]) == 1]
        candidates = exclusive_sids if exclusive_sids else sids
        n_test = max(1, len(sids) // 3)
        chosen_test = set(candidates[-n_test:])
        # Pastikan minimal 1 sesi tersisa untuk train di kelas ini
        remaining = [s for s in sids if s not in chosen_test and s not in test_session_ids]
        if not remaining:
            chosen_test = {candidates[-1]}
        test_session_ids.update(chosen_test)

    all_sessions = {r["session_id"] for r in rows}
    train_session_ids = all_sessions - test_session_ids

    if not train_session_ids.isdisjoint(test_session_ids):
        raise AssertionError("Terjadi overlap session_id antara train dan test!")

    train_rows = [r for r in rows if r["session_id"] in train_session_ids]
    test_rows = [r for r in rows if r["session_id"] in test_session_ids]

    train_classes = {r["label"] for r in train_rows}
    test_classes = {r["label"] for r in test_rows}
    missing_train = set(VALID_LABELS) - train_classes
    missing_test = set(VALID_LABELS) - test_classes
    if missing_train or missing_test:
        raise ValueError(
            "Gagal membagi sesi secara disjoint dengan ketiga kelas hadir di train & test "
            f"(missing_train={sorted(missing_train)}, missing_test={sorted(missing_test)}). "
            "Pastikan setiap perekaman kelas menggunakan session_id terpisah (minimal 2 sesi per kelas)."
        )

    return train_rows, test_rows, train_session_ids, test_session_ids


def train_and_evaluate(
    csv_path: str | Path = "data/posture_dataset.csv",
    model_out: str | Path = "models/posture_rf.joblib",
    metrics_out: str | Path = "models/metrics.json",
    cfg: DetectorConfig | None = None,
) -> dict[str, Any]:
    detector_cfg = cfg if cfg is not None else DetectorConfig()
    rows = load_and_validate_dataset(csv_path)
    train_rows, test_rows, train_sids, test_sids = group_split_by_session(rows)

    overlap_sids = train_sids & test_sids
    assert len(overlap_sids) == 0, f"Overlap sesi terdeteksi: {overlap_sids}"

    x_train = np.array([[r[c] for c in FEATURE_COLUMNS] for r in train_rows], dtype=float)
    y_train = np.array([r["label"] for r in train_rows], dtype=str)
    x_test = np.array([[r[c] for c in FEATURE_COLUMNS] for r in test_rows], dtype=float)
    y_test = np.array([r["label"] for r in test_rows], dtype=str)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_pred_rf = clf.predict(x_test)
    y_pred_rule = np.array(
        [
            predict_rule_baseline_row(
                r["overhead_deg"],
                r["roll_deg"],
                r["lean_deg"],
                r["motion_rms"],
                r["gyro_rms"],
                cfg=detector_cfg,
            )
            for r in test_rows
        ],
        dtype=str,
    )

    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_f1 = float(f1_score(y_test, y_pred_rf, labels=VALID_LABELS, average="macro", zero_division=0))
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=VALID_LABELS).tolist()

    rule_acc = float(accuracy_score(y_test, y_pred_rule))
    rule_f1 = float(f1_score(y_test, y_pred_rule, labels=VALID_LABELS, average="macro", zero_division=0))
    rule_cm = confusion_matrix(y_test, y_pred_rule, labels=VALID_LABELS).tolist()

    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    }

    model_path = Path(model_out)
    metrics_path = Path(metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "labels": list(clf.classes_),
        "valid_labels": VALID_LABELS,
    }
    joblib.dump(artifact, model_path)

    metrics_payload: dict[str, Any] = {
        "total_samples": len(rows),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_sessions": sorted(train_sids),
        "test_sessions": sorted(test_sids),
        "session_overlap_count": len(overlap_sids),
        "feature_columns": FEATURE_COLUMNS,
        "labels": VALID_LABELS,
        "random_forest": {
            "accuracy": round(rf_acc, 4),
            "macro_f1": round(rf_f1, 4),
            "confusion_matrix": rf_cm,
            "feature_importances": importances,
        },
        "rule_based_baseline": {
            "accuracy": round(rule_acc, 4),
            "macro_f1": round(rule_f1, 4),
            "confusion_matrix": rule_cm,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    _safe_print("=== REBAHGUARD ML TRAINING & BENCHMARK REPORT ===")
    _safe_print(
        f"Total Sampel  : {len(rows)} (Train: {len(train_rows)} dari {len(train_sids)} sesi | "
        f"Test: {len(test_rows)} dari {len(test_sids)} sesi)"
    )
    _safe_print(f"Session Overlap (Train ∩ Test): {len(overlap_sids)} sesi")
    _safe_print("\n--- Perbandingan pada Test Set Sesi yang Sama ---")
    _safe_print(f"{'Metode':<25} | {'Accuracy':<10} | {'Macro F1-Score':<14}")
    _safe_print("-" * 55)
    _safe_print(f"{'Random Forest (ML)':<25} | {rf_acc:>9.2%} | {rf_f1:>14.4f}")
    _safe_print(f"{'Rule-Based Baseline':<25} | {rule_acc:>9.2%} | {rule_f1:>14.4f}")

    _safe_print("\n--- Confusion Matrix 3x3: Random Forest (Baris=Aktual, Kolom=Prediksi) ---")
    _safe_print(f"Labels: {VALID_LABELS}")
    for lbl, row_cm in zip(VALID_LABELS, rf_cm):
        _safe_print(f"  {lbl:<20}: {row_cm}")

    _safe_print("\n--- Confusion Matrix 3x3: Rule-Based Baseline ---")
    for lbl, row_cm in zip(VALID_LABELS, rule_cm):
        _safe_print(f"  {lbl:<20}: {row_cm}")

    _safe_print("\n--- Feature Importance (Random Forest) ---")
    for col, val in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        _safe_print(f"  {col:<15}: {val:.4f}")

    _safe_print(f"\nModel disimpan ke  : {model_path}")
    _safe_print(f"Metrik disimpan ke : {metrics_path}")

    return metrics_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Latih model Random Forest RebahGuard bebas data leakage & bandingkan dengan Rule-Based."
    )
    parser.add_argument(
        "--input",
        default="data/posture_dataset.csv",
        help="Path file dataset CSV (default: data/posture_dataset.csv)",
    )
    parser.add_argument(
        "--model-out",
        default="models/posture_rf.joblib",
        help="Path penyimpanan model .joblib (default: models/posture_rf.joblib)",
    )
    parser.add_argument(
        "--metrics-out",
        default="models/metrics.json",
        help="Path penyimpanan metrik JSON (default: models/metrics.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_and_evaluate(
        csv_path=args.input,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    main()
