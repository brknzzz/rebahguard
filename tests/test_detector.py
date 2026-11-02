import json
import unittest

from nodes.posture_service.rules import RebahDetector
from shared.config import DetectorConfig
from shared.messages import AlertMessage, ControlCommand, StatusMessage, Telemetry
from shared.mqtt_connection import topic
from shared.processing import SensorSample, extract_features

NOW_MS = 1_790_000_000_000


def make_telemetry(
    device: str,
    seq: int,
    overhead_deg: float = 45.0,
    roll_deg: float = 10.0,
    motion_rms: float = 0.15,
    gyro_rms: float = 0.08,
    sent_at_ms: int = NOW_MS,
    boot_id: str = "boot-1",
) -> Telemetry:
    return Telemetry(
        device_id=device,
        boot_id=boot_id,
        sequence=seq,
        sent_at_ms=sent_at_ms,
        overhead_deg=overhead_deg,
        roll_deg=roll_deg,
        motion_rms=motion_rms,
        gyro_rms=gyro_rms,
    )


class RebahDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = RebahDetector()

    def test_mqtt_namespace_uses_rebahguard(self):
        t = topic("status")
        self.assertTrue(t.startswith("rebahguard/"))
        self.assertTrue(t.endswith("/status"))

    def test_landscape_gaming_sitting_does_not_trigger_rebahan(self):
        # Posisi duduk main game landscape (ax=9.81, ay=0.0, az=2.0) dengan default use_roll=True
        # overhead_deg ~ 78.48 deg (< landscape_overhead_enter_deg=80.0) -> tetap POSISI_DUDUK_SEHAT
        samples = [
            SensorSample(i * 0.2, (9.81, 0.0, 2.0), (0.05, 0.04, 0.02))
            for i in range(5)
        ]
        feat = extract_features(samples)
        self.assertTrue(self.detector.config.use_roll)
        self.assertLess(feat.overhead_deg, self.detector.config.landscape_overhead_enter_deg)

        for i in range(10):
            ts = NOW_MS + i * 2000
            t = make_telemetry(
                "edge1",
                seq=i,
                overhead_deg=feat.overhead_deg,
                roll_deg=feat.roll_deg,
                motion_rms=feat.motion_rms,
                gyro_rms=feat.gyro_rms,
                sent_at_ms=ts,
            )
            decision = self.detector.accept(t, now_ms=ts)
            self.assertEqual(decision.status, "POSISI_DUDUK_SEHAT")
            self.assertNotIn(decision.status, ("REBAHAN_AKTIF", "ALERT_MATA"))
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)
        self.assertEqual(self.detector.orientation, "landscape")
        self.assertEqual(self.detector.build_status(NOW_MS + 18_000).orientation, "landscape")

    def test_static_calibration_bias_face_down_table(self):
        # Akselerometer dengan bias kalibrasi statis (az = -9.55 konstan) -> motion_rms == 0.0 -> DITARUH_TELUNGKUP
        samples = [
            SensorSample(i * 0.2, (0.0, 0.0, -9.55), (0.0, 0.0, 0.0))
            for i in range(5)
        ]
        feat = extract_features(samples)
        self.assertEqual(feat.motion_rms, 0.0)
        self.assertEqual(feat.gyro_rms, 0.0)
        self.assertGreaterEqual(feat.overhead_deg, 175.0)

        t = make_telemetry(
            "edge1",
            seq=0,
            overhead_deg=feat.overhead_deg,
            roll_deg=feat.roll_deg,
            motion_rms=feat.motion_rms,
            gyro_rms=feat.gyro_rms,
            sent_at_ms=NOW_MS,
        )
        decision = self.detector.accept(t, now_ms=NOW_MS)
        self.assertEqual(decision.status, "DITARUH_TELUNGKUP")
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)

    def test_hysteresis_gray_zone_preserves_previous_state(self):
        # 1. Mulai dari duduk sehat (60 deg), lalu masuk zona abu-abu portrait (102 deg, di antara 95..110) -> tetap duduk sehat
        d0 = self.detector.accept(make_telemetry("edge1", 0, overhead_deg=60.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        self.assertEqual(d0.status, "POSISI_DUDUK_SEHAT")

        d1 = self.detector.accept(
            make_telemetry("edge1", 1, overhead_deg=102.0, sent_at_ms=NOW_MS + 1000),
            now_ms=NOW_MS + 1000,
        )
        self.assertEqual(d1.status, "POSISI_DUDUK_SEHAT")

        # 2. Masuk ke zona rebahan (>= 110 deg, misal 120 deg) -> REBAHAN_AKTIF
        d2 = self.detector.accept(
            make_telemetry("edge1", 2, overhead_deg=120.0, sent_at_ms=NOW_MS + 2000),
            now_ms=NOW_MS + 2000,
        )
        self.assertEqual(d2.status, "REBAHAN_AKTIF")

        # 3. Turun ke zona abu-abu (102 deg, antara 95..110) -> tetap REBAHAN_AKTIF (histeresis)
        d3 = self.detector.accept(
            make_telemetry("edge1", 3, overhead_deg=102.0, sent_at_ms=NOW_MS + 4000),
            now_ms=NOW_MS + 4000,
        )
        self.assertEqual(d3.status, "REBAHAN_AKTIF")
        self.assertGreater(self.detector.lying_elapsed_sec, 0.0)

        # 4. Turun ke <= 95 deg (90 deg) -> keluar ke POSISI_DUDUK_SEHAT
        d4 = self.detector.accept(
            make_telemetry("edge1", 4, overhead_deg=90.0, sent_at_ms=NOW_MS + 6000),
            now_ms=NOW_MS + 6000,
        )
        self.assertEqual(d4.status, "POSISI_DUDUK_SEHAT")
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)

    def test_orientation_transitions_and_flat_gate_retention(self):
        # Awalnya portrait
        self.detector.accept(
            make_telemetry("edge1", 0, overhead_deg=50.0, roll_deg=5.0, sent_at_ms=NOW_MS),
            now_ms=NOW_MS,
        )
        self.assertEqual(self.detector.build_status(NOW_MS).orientation, "portrait")

        # Putar ke landscape (overhead=75, roll=73 -> lean >= 60), butuh konfirmasi orientation_confirm_sec (0.5s)
        self.detector.accept(
            make_telemetry("edge1", 1, overhead_deg=75.0, roll_deg=73.0, sent_at_ms=NOW_MS + 200),
            now_ms=NOW_MS + 200,
        )
        self.assertEqual(self.detector.orientation, "portrait")  # belum 0.5 detik

        self.detector.accept(
            make_telemetry("edge1", 2, overhead_deg=75.0, roll_deg=73.0, sent_at_ms=NOW_MS + 800),
            now_ms=NOW_MS + 800,
        )
        self.assertEqual(self.detector.build_status(NOW_MS + 800).orientation, "landscape")

        # Saat HP hampir datar (overhead_deg < flat_gate_deg=20.0 atau > 160.0), tahan orientasi sebelumnya (landscape)
        self.detector.accept(
            make_telemetry("edge1", 3, overhead_deg=10.0, roll_deg=0.0, sent_at_ms=NOW_MS + 1800),
            now_ms=NOW_MS + 1800,
        )
        self.assertEqual(self.detector.build_status(NOW_MS + 1800).orientation, "landscape")

        self.detector.accept(
            make_telemetry("edge1", 4, overhead_deg=170.0, roll_deg=0.0, sent_at_ms=NOW_MS + 2800),
            now_ms=NOW_MS + 2800,
        )
        self.assertEqual(self.detector.build_status(NOW_MS + 2800).orientation, "landscape")

        # Kembali ke portrait (overhead=55, roll=5 -> lean < roll_exit_deg=45) setelah >= 0.5s
        self.detector.accept(
            make_telemetry("edge1", 5, overhead_deg=55.0, roll_deg=5.0, sent_at_ms=NOW_MS + 3500),
            now_ms=NOW_MS + 3500,
        )
        self.detector.accept(
            make_telemetry("edge1", 6, overhead_deg=55.0, roll_deg=5.0, sent_at_ms=NOW_MS + 4100),
            now_ms=NOW_MS + 4100,
        )
        self.assertEqual(self.detector.build_status(NOW_MS + 4100).orientation, "portrait")

    def test_face_down_and_sitting_both_clear_nagging_and_emit_clear_event(self):
        # Masuk ke REBAHAN_AKTIF lalu picu TEGURAN_ORTU
        self.detector.accept(make_telemetry("edge1", 0, overhead_deg=130.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        scold_dec = self.detector.apply_command("SCOLD", now_ms=NOW_MS + 500)
        self.assertEqual(scold_dec.status, "TEGURAN_ORTU")
        self.assertTrue(self.detector.nagging_active)
        self.assertIsNotNone(scold_dec.child_alert)
        self.assertEqual(scold_dec.child_alert.event, "TEGURAN_ORTU")

        # Case A: DITARUH_TELUNGKUP mematikan nagging_active dan mengirim CLEAR ke child
        d_table = self.detector.accept(
            make_telemetry(
                "edge1",
                1,
                overhead_deg=175.0,
                motion_rms=0.0,
                gyro_rms=0.0,
                sent_at_ms=NOW_MS + 1000,
            ),
            now_ms=NOW_MS + 1000,
        )
        self.assertEqual(d_table.status, "DITARUH_TELUNGKUP")
        self.assertFalse(self.detector.nagging_active)
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)
        self.assertIsNotNone(d_table.child_alert)
        self.assertEqual(d_table.child_alert.target, "child")
        self.assertEqual(d_table.child_alert.event, "CLEAR")

        # Masuk lagi ke REBAHAN_AKTIF & TEGURAN_ORTU
        self.detector.accept(
            make_telemetry("edge1", 2, overhead_deg=130.0, sent_at_ms=NOW_MS + 2000),
            now_ms=NOW_MS + 2000,
        )
        self.detector.apply_command("SCOLD", now_ms=NOW_MS + 2500)
        self.assertTrue(self.detector.nagging_active)

        # Case B: POSISI_DUDUK_SEHAT mematikan nagging_active dan mengirim CLEAR ke child
        d_sit = self.detector.accept(
            make_telemetry("edge1", 3, overhead_deg=50.0, sent_at_ms=NOW_MS + 3000),
            now_ms=NOW_MS + 3000,
        )
        self.assertEqual(d_sit.status, "POSISI_DUDUK_SEHAT")
        self.assertFalse(self.detector.nagging_active)
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)
        self.assertIsNotNone(d_sit.child_alert)
        self.assertEqual(d_sit.child_alert.target, "child")
        self.assertEqual(d_sit.child_alert.event, "CLEAR")

    def test_max_window_dt_clamping_and_alert_cooldown(self):
        # Packet 0 pada t=0
        self.detector.accept(make_telemetry("edge1", 0, overhead_deg=130.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        self.assertEqual(self.detector.lying_elapsed_sec, 0.0)

        # Packet 1 telat 9 detik (t=9000), tapi harus di-clamp ke max_window_dt_sec = 3.0 detik
        self.detector.accept(
            make_telemetry("edge1", 1, overhead_deg=130.0, sent_at_ms=NOW_MS + 9000),
            now_ms=NOW_MS + 9000,
        )
        self.assertAlmostEqual(self.detector.lying_elapsed_sec, 3.0, places=3)

        # Tambah beberapa window hingga mencapai lying_limit_sec (10.0s): +3s (6s), +3s (9s), +2s (11s -> ALERT_MATA pertama)
        self.detector.accept(
            make_telemetry("edge1", 2, overhead_deg=130.0, sent_at_ms=NOW_MS + 12_000),
            now_ms=NOW_MS + 12_000,
        )
        self.detector.accept(
            make_telemetry("edge1", 3, overhead_deg=130.0, sent_at_ms=NOW_MS + 15_000),
            now_ms=NOW_MS + 15_000,
        )
        first_alert = self.detector.accept(
            make_telemetry("edge1", 4, overhead_deg=130.0, sent_at_ms=NOW_MS + 17_000),
            now_ms=NOW_MS + 17_000,
        )
        self.assertEqual(first_alert.status, "ALERT_MATA")
        self.assertIsNotNone(first_alert.parent_alert)
        self.assertIsNotNone(first_alert.child_alert)
        self.assertEqual(first_alert.parent_alert.event, "ALERT_MATA")
        self.assertEqual(first_alert.child_alert.event, "ALERT_MATA")

        # Window berikutnya 2 detik kemudian (belum lewat alert_cooldown_sec=8.0s) -> TIDAK spam alert
        second_alert = self.detector.accept(
            make_telemetry("edge1", 5, overhead_deg=130.0, sent_at_ms=NOW_MS + 19_000),
            now_ms=NOW_MS + 19_000,
        )
        self.assertEqual(second_alert.status, "ALERT_MATA")
        self.assertIsNone(second_alert.parent_alert)
        self.assertIsNone(second_alert.child_alert)

        # Setelah >= 8.0s sejak alert pertama (17_000 + 8_000 = 25_000) -> terbitkan alert lagi
        third_alert = self.detector.accept(
            make_telemetry("edge1", 6, overhead_deg=130.0, sent_at_ms=NOW_MS + 25_000),
            now_ms=NOW_MS + 25_000,
        )
        self.assertEqual(third_alert.status, "ALERT_MATA")
        self.assertIsNotNone(third_alert.parent_alert)
        self.assertIsNotNone(third_alert.child_alert)

    def test_shake_to_scold_requires_two_consecutive_windows_and_cooldown(self):
        # Edge 1 dalam posisi rebahan
        self.detector.accept(make_telemetry("edge1", 0, overhead_deg=130.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)

        # Edge 2 goyang 1 window -> belum memicu TEGURAN_ORTU (butuh shake_confirm_windows = 2)
        d1 = self.detector.accept(
            make_telemetry("edge2", 0, motion_rms=2.5, gyro_rms=0.2, sent_at_ms=NOW_MS + 500),
            now_ms=NOW_MS + 500,
        )
        self.assertEqual(d1.status, "REBAHAN_AKTIF")
        self.assertFalse(self.detector.nagging_active)
        self.assertIsNone(d1.child_alert)

        # Jika diselingi window tenang, counter reset ke 0
        self.detector.accept(
            make_telemetry("edge2", 1, motion_rms=0.1, gyro_rms=0.1, sent_at_ms=NOW_MS + 1000),
            now_ms=NOW_MS + 1000,
        )
        d_single = self.detector.accept(
            make_telemetry("edge2", 2, motion_rms=2.5, gyro_rms=0.1, sent_at_ms=NOW_MS + 1500),
            now_ms=NOW_MS + 1500,
        )
        self.assertFalse(self.detector.nagging_active)

        # Window ke-2 berturut-turut dengan goyangan -> memicu TEGURAN_ORTU!
        d_confirm = self.detector.accept(
            make_telemetry("edge2", 3, motion_rms=0.2, gyro_rms=2.0, sent_at_ms=NOW_MS + 2000),
            now_ms=NOW_MS + 2000,
        )
        self.assertEqual(d_confirm.status, "TEGURAN_ORTU")
        self.assertTrue(self.detector.nagging_active)
        self.assertIsNotNone(d_confirm.child_alert)
        self.assertEqual(d_confirm.child_alert.event, "TEGURAN_ORTU")

    def test_edge_offline_timeout(self):
        # Sebelum Edge 1 pernah mengirim -> EDGE_OFFLINE
        initial_timeout = self.detector.check_timeouts(NOW_MS)
        self.assertEqual(initial_timeout.status, "EDGE_OFFLINE")

        # Edge 1 kirim pada NOW_MS -> online
        self.detector.accept(make_telemetry("edge1", 0, overhead_deg=50.0, sent_at_ms=NOW_MS), now_ms=NOW_MS)
        self.assertEqual(self.detector.check_timeouts(NOW_MS + 4000).status, "POSISI_DUDUK_SEHAT")

        # Lewat edge_timeout_sec (8.0s) -> EDGE_OFFLINE
        after_timeout = self.detector.check_timeouts(NOW_MS + 8500)
        self.assertEqual(after_timeout.status, "EDGE_OFFLINE")

    def test_live_update_config_and_validation(self):
        cmd = ControlCommand.from_json(
            json.dumps({"command": "UPDATE_CONFIG", "params": {"lying_limit_sec": 5.0, "use_roll": True}})
        )
        self.detector.apply_command(cmd, now_ms=NOW_MS)
        self.assertEqual(self.detector.config.lying_limit_sec, 5.0)
        self.assertTrue(self.detector.config.use_roll)

        # Validasi gagal jika overhead_exit_deg >= overhead_enter_deg atau nilai negatif
        with self.assertRaises(ValueError):
            DetectorConfig(overhead_enter_deg=100.0, overhead_exit_deg=110.0)
        with self.assertRaises(ValueError):
            self.detector.update_config({"lying_limit_sec": -3.0})

    def test_replay_and_stale_and_schema_validation_rejected(self):
        reading = make_telemetry("edge1", 3, sent_at_ms=NOW_MS)
        self.detector.accept(reading, now_ms=NOW_MS)
        replay = self.detector.accept(reading, now_ms=NOW_MS)
        self.assertEqual(replay.status, "REJECTED_REPLAY")

        stale = self.detector.accept(make_telemetry("edge1", 4, sent_at_ms=NOW_MS - 31_000), now_ms=NOW_MS)
        self.assertEqual(stale.status, "REJECTED_STALE")

        # Invalid JSON / NaN / wrong types
        with self.assertRaises(ValueError):
            Telemetry.from_json(
                json.dumps(
                    {
                        "device_id": "edge1",
                        "boot_id": "abc",
                        "sequence": 1,
                        "sent_at_ms": NOW_MS,
                        "overhead_deg": float("nan"),
                        "roll_deg": 0.0,
                        "motion_rms": 0.0,
                        "gyro_rms": 0.0,
                    }
                )
            )
        with self.assertRaises(ValueError):
            AlertMessage.from_json(
                json.dumps({"target": "unknown", "event": "ALERT_MATA", "reason": "x", "lying_sec": 1.0, "at_ms": NOW_MS})
            )
        with self.assertRaises(ValueError):
            StatusMessage.from_json(json.dumps({"status": ""}))


if __name__ == "__main__":
    unittest.main()
