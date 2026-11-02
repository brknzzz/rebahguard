import json
import pathlib
import re
import unittest
from unittest import mock

from scripts.offline_demo import main as run_offline_demo
from scripts.send_control import build_control_from_args
from shared import alarm
from shared.messages import AlertMessage, ControlCommand, StatusMessage, Telemetry
from shared.processing import extract_features
from shared.sensors import SIMULATOR_SCENARIOS, Simulator, parse_termux_sensor_stream


class MessageSchemaTests(unittest.TestCase):
    def test_status_message_roundtrip_and_fields(self):
        msg = StatusMessage(
            status="ALERT_MATA",
            reason="Rebahan terlalu lama",
            armed=True,
            nagging_active=False,
            lying_elapsed_sec=12.5,
            lying_limit_sec=10.0,
            edge1_online=True,
            edge2_online=True,
            at_ms=123456,
            device_id="edge1",
            sequence=4,
            orientation="portrait",
            detection_mode="ml",
            ml_prediction="REBAHAN_AKTIF",
            ml_confidence=0.925,
        )
        raw = msg.to_json()
        parsed = StatusMessage.from_json(raw)
        self.assertEqual(parsed, msg)
        body = json.loads(raw)
        self.assertEqual(
            body,
            {
                "status": "ALERT_MATA",
                "reason": "Rebahan terlalu lama",
                "armed": True,
                "nagging_active": False,
                "lying_elapsed_sec": 12.5,
                "lying_limit_sec": 10.0,
                "edge1_online": True,
                "edge2_online": True,
                "at_ms": 123456,
                "device_id": "edge1",
                "sequence": 4,
                "orientation": "portrait",
                "detection_mode": "ml",
                "ml_prediction": "REBAHAN_AKTIF",
                "ml_confidence": 0.925,
            },
        )

    def test_status_message_backward_compatible_and_ml_validation(self):
        legacy_payload = json.dumps(
            {
                "status": "POSISI_DUDUK_SEHAT",
                "reason": "Posisi sehat",
                "armed": True,
                "nagging_active": False,
                "lying_elapsed_sec": 0.0,
                "lying_limit_sec": 10.0,
                "edge1_online": True,
                "edge2_online": False,
                "at_ms": 123456,
                "device_id": "edge1",
                "sequence": 1,
            }
        )
        parsed = StatusMessage.from_json(legacy_payload)
        self.assertEqual(parsed.orientation, "portrait")
        self.assertEqual(parsed.detection_mode, "rule")
        self.assertIsNone(parsed.ml_prediction)
        self.assertIsNone(parsed.ml_confidence)

        # Tolak orientasi tidak valid
        with self.assertRaises(ValueError):
            StatusMessage(
                status="POSISI_DUDUK_SEHAT",
                reason="Invalid orientation",
                armed=True,
                nagging_active=False,
                lying_elapsed_sec=0.0,
                lying_limit_sec=10.0,
                edge1_online=True,
                edge2_online=False,
                at_ms=123456,
                orientation="diagonal",
            )

        # Tolak ml_confidence > 1.0, < 0.0, atau NaN
        for bad_conf in (1.05, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                StatusMessage(
                    status="REBAHAN_AKTIF",
                    reason="Bad confidence",
                    armed=True,
                    nagging_active=False,
                    lying_elapsed_sec=2.0,
                    lying_limit_sec=10.0,
                    edge1_online=True,
                    edge2_online=False,
                    at_ms=123456,
                    detection_mode="ml",
                    ml_prediction="REBAHAN_AKTIF",
                    ml_confidence=bad_conf,
                )

        # Tolak detection_mode atau ml_prediction tidak valid
        with self.assertRaises(ValueError):
            StatusMessage(
                status="REBAHAN_AKTIF",
                reason="Bad mode",
                armed=True,
                nagging_active=False,
                lying_elapsed_sec=2.0,
                lying_limit_sec=10.0,
                edge1_online=True,
                edge2_online=False,
                at_ms=123456,
                detection_mode="unknown",
            )
        with self.assertRaises(ValueError):
            StatusMessage(
                status="REBAHAN_AKTIF",
                reason="Bad label",
                armed=True,
                nagging_active=False,
                lying_elapsed_sec=2.0,
                lying_limit_sec=10.0,
                edge1_online=True,
                edge2_online=False,
                at_ms=123456,
                detection_mode="ml",
                ml_prediction="TIDUR_MIRING",
            )

    def test_alert_message_roundtrip(self):
        alert = AlertMessage(
            target="child",
            event="TEGURAN_ORTU",
            reason="Ortu goyang HP",
            lying_sec=8.2,
            at_ms=999,
        )
        self.assertEqual(AlertMessage.from_json(alert.to_json()), alert)

    def test_control_command_json_and_string_fallback(self):
        for s in ("ARM", "DISARM", "SCOLD", "SNOOZE", "RESET"):
            cmd = ControlCommand.from_json(s)
            self.assertEqual(cmd.command, s)
            self.assertEqual(cmd.params, {})

        cfg_cmd = ControlCommand.from_json('{"command": "UPDATE_CONFIG", "params": {"lying_limit_sec": 15.0}}')
        self.assertEqual(cfg_cmd.command, "UPDATE_CONFIG")
        self.assertEqual(cfg_cmd.params["lying_limit_sec"], 15.0)

    def test_send_control_cli_args_builder(self):
        self.assertEqual(build_control_from_args(["SCOLD"]).command, "SCOLD")
        limit_cmd = build_control_from_args(["--set-limit", "5"])
        self.assertEqual(limit_cmd.command, "UPDATE_CONFIG")
        self.assertEqual(limit_cmd.params, {"lying_limit_sec": 5.0})

    def test_telemetry_roundtrip(self):
        t = Telemetry.make(
            device_id="edge1",
            boot_id="boot-abc",
            sequence=1,
            overhead_deg=125.0,
            roll_deg=15.0,
            motion_rms=0.12,
            gyro_rms=0.05,
            sent_at_ms=5000,
        )
        self.assertEqual(Telemetry.from_json(t.to_json()), t)


class AlarmTests(unittest.TestCase):
    def test_laptop_only_prints(self):
        with mock.patch("shared.alarm.shutil.which", return_value=None), \
             mock.patch("shared.alarm.subprocess.run") as run:
            alarm.handle_parent_alert({"event": "ALERT_MATA", "lying_sec": 10.0, "reason": "test"})
            alarm.handle_child_alert({"event": "TEGURAN_ORTU", "lying_sec": 10.0, "reason": "test"})
            alarm.handle_child_alert({"event": "CLEAR", "lying_sec": 0.0, "reason": "test"})
        run.assert_not_called()

    def test_parent_alert_vibrates_1200_and_notifies(self):
        with mock.patch("shared.alarm.shutil.which", return_value="/bin/x"), \
             mock.patch("shared.alarm.subprocess.run") as run:
            alarm.handle_parent_alert({"event": "ALERT_MATA", "lying_sec": 11.0, "reason": "Rebahan"})
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(calls[0], ["termux-vibrate", "-d", "1200"])
        self.assertIn("rebahguard-parent", calls[1])

    def test_child_alerts_clear_alert_mata_and_teguran_ortu(self):
        with mock.patch("shared.alarm.shutil.which", return_value="/bin/x"), \
             mock.patch("shared.alarm.subprocess.run") as run:
            alarm.handle_child_alert({"event": "CLEAR", "lying_sec": 0.0, "reason": "Duduk sehat"})
            self.assertEqual(run.call_args_list[-1].args[0], ["termux-notification-remove", "rebahguard-child"])

            run.reset_mock()
            alarm.handle_child_alert({"event": "ALERT_MATA", "lying_sec": 10.5, "reason": "Batas terlampaui"})
            self.assertEqual(run.call_args_list[0].args[0], ["termux-vibrate", "-d", "600"])

            run.reset_mock()
            alarm.handle_child_alert({"event": "TEGURAN_ORTU", "lying_sec": 14.0, "reason": "Ortu goyang HP"})
            self.assertEqual(run.call_args_list[0].args[0], ["termux-vibrate", "-f", "-d", "1500"])
            self.assertIn("RebahGuard: Segera duduk tegak!", run.call_args_list[1].args[0])


class SensorAndSimulatorTests(unittest.TestCase):
    def test_parse_termux_multi_json_stream(self):
        stream = (
            '{"acc": {"values": [0.0, 0.0, -9.6]}, "gyr": {"values": [0.01, 0.02, 0.0]}}\n'
            '{"acc": {"values": [0.0, 0.0, -9.6]}, "gyr": {"values": [0.01, 0.02, 0.0]}}\n'
            '{"acc": {"values": [0.0, 0.0, -9.6]}, "gyr": {"values": [0.01, 0.02, 0.0]}}'
        )
        samples = parse_termux_sensor_stream(stream, "acc", "gyr", base_time=10.0, step_sec=0.1)
        self.assertEqual(len(samples), 3)
        feat = extract_features(samples)
        self.assertEqual(feat.motion_rms, 0.0)
        self.assertAlmostEqual(feat.overhead_deg, 180.0, places=2)

    def test_all_simulator_scenarios_produce_expected_features(self):
        for scen in SIMULATOR_SCENARIOS:
            sim = Simulator(scenario=scen, seed=7)
            window = [sim.read() for _ in range(6)]
            feat = extract_features(window)
            if scen == "duduk_sehat":
                self.assertLess(feat.overhead_deg, 90.0)
            elif scen == "landscape_game":
                self.assertLess(feat.overhead_deg, 90.0)
                self.assertGreater(feat.roll_deg, 65.0)
            elif scen == "telungkup_meja":
                self.assertGreaterEqual(feat.overhead_deg, 145.0)
                self.assertEqual(feat.motion_rms, 0.0)
                self.assertEqual(feat.gyro_rms, 0.0)
            elif scen == "rebahan_main":
                self.assertGreaterEqual(feat.overhead_deg, 120.0)
                self.assertGreater(feat.motion_rms, 0.05)
            elif scen == "ortu_shake":
                self.assertGreaterEqual(feat.motion_rms, 2.0)

    def test_offline_demo_runs_cleanly(self):
        run_offline_demo()

    def test_zero_legacy_names_in_repository(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        legacy_terms = ("pocket" + "guard", "theft" + "detector", "pocket_" + "guard", "hp_" + "dijaga", "hp_" + "pemilik")
        pat = re.compile("|".join(legacy_terms), re.IGNORECASE)
        hits = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pat.search(text):
                hits.append(str(path.relative_to(root)))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
