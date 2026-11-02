import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib

from nodes.posture_service.ml_model import PostureMLClassifier
from nodes.posture_service.rules import RebahDetector
from scripts.collect_dataset import CSV_COLUMNS, DatasetCollector, summarize_dataset
from scripts.train_model import (
    FEATURE_COLUMNS,
    VALID_LABELS,
    predict_rule_baseline_row,
    train_and_evaluate,
)
from shared.config import DetectorConfig
from shared.messages import Telemetry
from shared.processing import safe_lean_deg

NOW_MS = 1_790_000_000_000


def _make_edge1_telemetry(
    seq: int,
    overhead_deg: float = 125.0,
    roll_deg: float = 10.0,
    motion_rms: float = 0.15,
    gyro_rms: float = 0.08,
    sent_at_ms: int = NOW_MS,
) -> Telemetry:
    return Telemetry(
        device_id="edge1",
        boot_id="boot-hybrid",
        sequence=seq,
        sent_at_ms=sent_at_ms,
        overhead_deg=overhead_deg,
        roll_deg=roll_deg,
        motion_rms=motion_rms,
        gyro_rms=gyro_rms,
    )


class FakeMLClassifier:
    """Classifier palsu deterministik untuk menguji integrasi Hybrid ML + FSM."""

    def __init__(self, responses: list[tuple[str, float] | None]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def predict(
        self,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
    ) -> tuple[str, float] | None:
        if not self.responses:
            return None
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]


class CollectDatasetTests(unittest.TestCase):
    def test_collector_filters_edge2_stale_and_replay_and_computes_safe_lean_deg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "posture_dataset.csv"
            collector = DatasetCollector(
                label="POSISI_DUDUK_SEHAT",
                output_path=csv_path,
                target_samples=5,
                session_id="sess_20261102_120000_abcd",
            )

            # 1. Tolak device_id != "edge1"
            t_edge2 = Telemetry(
                device_id="edge2",
                boot_id="b1",
                sequence=1,
                sent_at_ms=NOW_MS,
                overhead_deg=50.0,
                roll_deg=10.0,
                motion_rms=0.1,
                gyro_rms=0.05,
            )
            ok, status, _ = collector.accept_payload(t_edge2.to_json(), now_ms=NOW_MS)
            self.assertFalse(ok)
            self.assertEqual(status, "REJECTED_DEVICE")

            # 2. Tolak paket stale (> 30_000 ms)
            t_stale = Telemetry(
                device_id="edge1",
                boot_id="b1",
                sequence=1,
                sent_at_ms=NOW_MS - 35_000,
                overhead_deg=50.0,
                roll_deg=10.0,
                motion_rms=0.1,
                gyro_rms=0.05,
            )
            ok, status, _ = collector.accept_payload(t_stale.to_json(), now_ms=NOW_MS)
            self.assertFalse(ok)
            self.assertEqual(status, "REJECTED_STALE")

            # 3. Terima paket valid pertama dari edge1
            t_valid1 = Telemetry(
                device_id="edge1",
                boot_id="b1",
                sequence=1,
                sent_at_ms=NOW_MS,
                overhead_deg=60.0,
                roll_deg=12.0,
                motion_rms=0.12,
                gyro_rms=0.04,
            )
            ok, status, row1 = collector.accept_payload(t_valid1.to_json(), now_ms=NOW_MS)
            self.assertTrue(ok)
            self.assertEqual(status, "ACCEPTED")
            self.assertIsNotNone(row1)
            self.assertEqual(row1["lean_deg"], safe_lean_deg(60.0, 12.0))

            # 4. Tolak paket replay (sequence sama atau mundur pada boot_id yang sama)
            ok, status, _ = collector.accept_payload(t_valid1.to_json(), now_ms=NOW_MS + 500)
            self.assertFalse(ok)
            self.assertEqual(status, "REJECTED_REPLAY")

            # 5. Terima paket datar (overhead=0.0, roll=0.0) -> lean_deg == 0.0 via safe_lean_deg
            t_flat = Telemetry(
                device_id="edge1",
                boot_id="b1",
                sequence=2,
                sent_at_ms=NOW_MS + 1000,
                overhead_deg=0.0,
                roll_deg=0.0,
                motion_rms=0.0,
                gyro_rms=0.0,
            )
            ok, status, row2 = collector.accept_payload(t_flat.to_json(), now_ms=NOW_MS + 1000)
            self.assertTrue(ok)
            self.assertEqual(row2["lean_deg"], 0.0)

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, CSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["session_id"], "sess_20261102_120000_abcd")
            self.assertEqual(rows[0]["label"], "POSISI_DUDUK_SEHAT")

            summary = summarize_dataset(csv_path)
            self.assertEqual(summary["POSISI_DUDUK_SEHAT"]["samples"], 2)
            self.assertEqual(summary["POSISI_DUDUK_SEHAT"]["sessions"], 1)


class TrainModelTests(unittest.TestCase):
    def _write_synthetic_csv(
        self,
        csv_path: Path,
        samples_per_session: int = 18,
        sessions_per_class: int = 2,
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            ts = NOW_MS
            for lbl in VALID_LABELS:
                for s_idx in range(sessions_per_class):
                    sid = f"sess_{lbl.lower()}_{s_idx}"
                    for i in range(samples_per_session):
                        ts += 1000
                        if lbl == "POSISI_DUDUK_SEHAT":
                            overhead = 50.0 + (i % 5)
                            roll = 8.0 + (i % 3)
                            motion = 0.12
                            gyro = 0.06
                        elif lbl == "DITARUH_TELUNGKUP":
                            overhead = 177.0 + (i % 3) * 0.5
                            roll = 1.0
                            motion = 0.005
                            gyro = 0.002
                        else:  # REBAHAN_AKTIF
                            overhead = 125.0 + (i % 6)
                            roll = 10.0 + (i % 4)
                            motion = 0.18
                            gyro = 0.09

                        writer.writerow(
                            {
                                "session_id": sid,
                                "timestamp_ms": ts,
                                "overhead_deg": overhead,
                                "roll_deg": roll,
                                "lean_deg": safe_lean_deg(overhead, roll),
                                "motion_rms": motion,
                                "gyro_rms": gyro,
                                "label": lbl,
                            }
                        )

    def test_raises_value_error_when_samples_below_30_or_sessions_below_2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Kasus 1: Hanya 1 sesi per kelas (walaupun sampel >= 30) -> wajib ValueError
            csv_1_sess = tmp / "one_session.csv"
            self._write_synthetic_csv(csv_1_sess, samples_per_session=35, sessions_per_class=1)
            with self.assertRaises(ValueError):
                train_and_evaluate(
                    csv_path=csv_1_sess,
                    model_out=tmp / "m1.joblib",
                    metrics_out=tmp / "m1.json",
                )

            # Kasus 2: 2 sesi per kelas tapi total sampel < 30 (misal 10 * 2 = 20) -> wajib ValueError
            csv_few_samples = tmp / "few_samples.csv"
            self._write_synthetic_csv(csv_few_samples, samples_per_session=10, sessions_per_class=2)
            with self.assertRaises(ValueError):
                train_and_evaluate(
                    csv_path=csv_few_samples,
                    model_out=tmp / "m2.joblib",
                    metrics_out=tmp / "m2.json",
                )

    def test_group_split_zero_overlap_and_creates_joblib_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_valid = tmp / "valid_dataset.csv"
            model_out = tmp / "models" / "posture_rf.joblib"
            metrics_out = tmp / "models" / "metrics.json"

            # 3 sesi per kelas x 15 sampel = 45 sampel per kelas (>= 30 sampel & >= 2 sesi)
            self._write_synthetic_csv(csv_valid, samples_per_session=15, sessions_per_class=3)

            metrics = train_and_evaluate(
                csv_path=csv_valid,
                model_out=model_out,
                metrics_out=metrics_out,
            )

            self.assertTrue(model_out.exists())
            self.assertTrue(metrics_out.exists())
            self.assertEqual(metrics["session_overlap_count"], 0)

            train_sids = set(metrics["train_sessions"])
            test_sids = set(metrics["test_sessions"])
            self.assertTrue(train_sids.isdisjoint(test_sids))
            self.assertGreaterEqual(len(test_sids), 3)

            # Pastikan model .joblib dapat dimuat dan memiliki feature_columns yang tepat
            artifact = joblib.load(model_out)
            self.assertEqual(artifact["feature_columns"], FEATURE_COLUMNS)
            self.assertIn("model", artifact)

            # Pastikan metrik RF dan Rule-Based Baseline tercatat lengkap di metrics.json
            saved_metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            self.assertIn("random_forest", saved_metrics)
            self.assertIn("rule_based_baseline", saved_metrics)
            self.assertGreaterEqual(saved_metrics["random_forest"]["accuracy"], 0.9)
            self.assertGreaterEqual(saved_metrics["rule_based_baseline"]["accuracy"], 0.9)
            self.assertEqual(len(saved_metrics["random_forest"]["confusion_matrix"]), 3)
            self.assertEqual(len(saved_metrics["rule_based_baseline"]["confusion_matrix"]), 3)

    def test_predict_rule_baseline_row_portrait_and_landscape(self):
        # Telungkup di meja
        self.assertEqual(
            predict_rule_baseline_row(178.0, 0.0, 0.0, 0.005, 0.002),
            "DITARUH_TELUNGKUP",
        )
        # Duduk main game landscape (overhead=78.48 < landscape_enter=80.0, lean=78.0 >= 60.0)
        self.assertEqual(
            predict_rule_baseline_row(78.48, 78.48, 78.0, 0.12, 0.05),
            "POSISI_DUDUK_SEHAT",
        )
        # Rebahan landscape (overhead=85.0 >= landscape_enter=80.0, lean=75.0 >= 60.0)
        self.assertEqual(
            predict_rule_baseline_row(85.0, 75.0, 75.0, 0.12, 0.05),
            "REBAHAN_AKTIF",
        )


class HybridMLDetectorTests(unittest.TestCase):
    def test_hard_veto_face_down_table_always_beats_ml_prediction(self):
        # Mock ML memprediksi REBAHAN_AKTIF dengan confidence 0.99
        fake_ml = FakeMLClassifier([("REBAHAN_AKTIF", 0.99), ("REBAHAN_AKTIF", 0.99)])
        detector = RebahDetector(ml_classifier=fake_ml)

        # Window 0: rebahan aktif via ML
        d0 = detector.accept(_make_edge1_telemetry(0, overhead_deg=125.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        self.assertEqual(d0.status, "REBAHAN_AKTIF")
        self.assertEqual(detector.detection_mode, "ml")
        self.assertEqual(detector.ml_prediction, "REBAHAN_AKTIF")
        self.assertAlmostEqual(detector.ml_confidence or 0.0, 0.99, places=4)

        # Window 1: HP diletakkan telungkup di meja (overhead=178.0 >= 175.0, motion=0.0, gyro=0.0)
        # Hard-veto WAJIB menang atas prediksi ML REBAHAN_AKTIF (0.99)!
        d_table = detector.accept(
            _make_edge1_telemetry(
                1,
                overhead_deg=178.0,
                roll_deg=0.0,
                motion_rms=0.0,
                gyro_rms=0.0,
                sent_at_ms=NOW_MS + 2000,
            ),
            now_ms=NOW_MS + 2000,
        )
        self.assertEqual(d_table.status, "DITARUH_TELUNGKUP")
        self.assertEqual(detector.lying_elapsed_sec, 0.0)
        self.assertEqual(detector.detection_mode, "rule")
        self.assertIsNone(detector.ml_prediction)
        self.assertIsNone(detector.ml_confidence)
        self.assertEqual(len(detector._ml_buffer), 0)

    def test_temporal_smoothing_single_glitch_does_not_reset_lying_timer(self):
        # ml_smoothing_window=3:
        # w0: REBAHAN_AKTIF -> buffer=[R] -> R (elapsed=0.0s)
        # w1: REBAHAN_AKTIF -> buffer=[R, R] -> R (elapsed=2.0s)
        # w2: POSISI_DUDUK_SEHAT (1 window glitch) -> buffer=[R, R, D] -> majority R (elapsed=4.0s, TIDAK reset!)
        # w3: REBAHAN_AKTIF -> buffer=[R, D, R] -> majority R (elapsed=6.0s)
        fake_ml = FakeMLClassifier(
            [
                ("REBAHAN_AKTIF", 0.92),
                ("REBAHAN_AKTIF", 0.91),
                ("POSISI_DUDUK_SEHAT", 0.88),
                ("REBAHAN_AKTIF", 0.94),
            ]
        )
        cfg = DetectorConfig(ml_smoothing_window=3, ml_confidence_min=0.55)
        detector = RebahDetector(config=cfg, ml_classifier=fake_ml)

        detector.accept(_make_edge1_telemetry(0, overhead_deg=100.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        self.assertEqual(detector.status, "REBAHAN_AKTIF")
        self.assertEqual(detector.lying_elapsed_sec, 0.0)

        detector.accept(_make_edge1_telemetry(1, overhead_deg=100.0, sent_at_ms=NOW_MS + 2000), now_ms=NOW_MS + 2000)
        self.assertEqual(detector.status, "REBAHAN_AKTIF")
        self.assertAlmostEqual(detector.lying_elapsed_sec, 2.0, places=3)

        # Window ke-2 adalah glitch POSISI_DUDUK_SEHAT, namun karena majority vote 2 vs 1 tetap REBAHAN_AKTIF!
        d_glitch = detector.accept(
            _make_edge1_telemetry(2, overhead_deg=70.0, sent_at_ms=NOW_MS + 4000),
            now_ms=NOW_MS + 4000,
        )
        self.assertEqual(d_glitch.status, "REBAHAN_AKTIF")
        self.assertEqual(detector.ml_prediction, "REBAHAN_AKTIF")
        self.assertAlmostEqual(detector.lying_elapsed_sec, 4.0, places=3)

        # Window ke-3 kembali REBAHAN_AKTIF -> timer terus bertambah ke 6.0s
        d_next = detector.accept(
            _make_edge1_telemetry(3, overhead_deg=105.0, sent_at_ms=NOW_MS + 6000),
            now_ms=NOW_MS + 6000,
        )
        self.assertEqual(d_next.status, "REBAHAN_AKTIF")
        self.assertAlmostEqual(detector.lying_elapsed_sec, 6.0, places=3)

    def test_posture_ml_classifier_throttles_getmtime_syscalls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_valid = tmp / "dataset.csv"
            model_path = tmp / "posture_rf.joblib"
            metrics_path = tmp / "metrics.json"
            TrainModelTests()._write_synthetic_csv(csv_valid, samples_per_session=15, sessions_per_class=3)
            train_and_evaluate(csv_path=csv_valid, model_out=model_path, metrics_out=metrics_path)

            clf_wrapper = PostureMLClassifier(model_path=model_path, reload_interval_sec=5.0)

            with mock.patch("nodes.posture_service.ml_model.os.path.getmtime", wraps=Path.stat) as _:
                pass

            real_getmtime = __import__("os").path.getmtime
            with mock.patch("nodes.posture_service.ml_model.os.path.getmtime", side_effect=real_getmtime) as m_mtime, \
                 mock.patch("nodes.posture_service.ml_model.time.monotonic") as m_mono:
                # Panggilan ke-1 pada t=100.0 -> mengecek getmtime (call_count = 1)
                m_mono.return_value = 100.0
                res1 = clf_wrapper.predict(52.0, 8.0, 0.12, 0.06)
                self.assertIsNotNone(res1)
                self.assertEqual(res1[0], "POSISI_DUDUK_SEHAT")
                self.assertEqual(m_mtime.call_count, 1)

                # Panggilan ke-2 & ke-3 pada t=101.0 dan t=104.9 (< 5.0 detik) -> TIDAK memanggil getmtime!
                m_mono.return_value = 101.0
                clf_wrapper.predict(126.0, 10.0, 0.18, 0.09)
                m_mono.return_value = 104.9
                res2 = clf_wrapper.predict(126.0, 10.0, 0.18, 0.09)
                self.assertIsNotNone(res2)
                self.assertEqual(res2[0], "REBAHAN_AKTIF")
                self.assertEqual(m_mtime.call_count, 1)

                # Panggilan ke-4 pada t=105.5 (>= 5.0 detik) -> mengecek getmtime lagi (call_count = 2)
                m_mono.return_value = 105.5
                clf_wrapper.predict(178.0, 1.0, 0.005, 0.002)
                self.assertEqual(m_mtime.call_count, 2)


if __name__ == "__main__":
    unittest.main()
