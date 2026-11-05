"""NODE 3: RebahGuard posture service subscribes to MQTT, applies FSM rules, publishes status/config/alerts.

Topics:
  subscribe  rebahguard/<PROJECT_ID>/edge/+/telemetry  (QoS 1)
  subscribe  rebahguard/<PROJECT_ID>/control           (QoS 1)
  publish    rebahguard/<PROJECT_ID>/status            (QoS 1, retain=True)
  publish    rebahguard/<PROJECT_ID>/config            (QoS 1, retain=True)
  publish    rebahguard/<PROJECT_ID>/alerts/parent     (QoS 1, retain=False)
  publish    rebahguard/<PROJECT_ID>/alerts/child      (QoS 1, retain=False)
"""

import json
import threading
import time

from nodes.posture_service.ml_model import PostureMLClassifier
from nodes.posture_service.rules import Decision, RebahDetector
from shared.messages import ControlCommand, StatusMessage, Telemetry
from shared.mqtt_connection import create_client, topic


def build_offline_lwt_json(lying_limit_sec: float = 10.0, now_ms: int | None = None) -> str:
    current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return StatusMessage(
        status="OFFLINE",
        reason="Posture service terputus dari broker",
        armed=False,
        nagging_active=False,
        lying_elapsed_sec=0.0,
        lying_limit_sec=lying_limit_sec,
        edge1_online=False,
        edge2_online=False,
        at_ms=current_ms,
        orientation="portrait",
        detection_mode="rule",
        ml_probabilities=None,
        ml_probabilities_source=None,
    ).to_json()


class PostureServiceBridge:
    """Thread-safe coordinator binding RebahDetector to an MQTT client."""

    def __init__(self, client, detector: RebahDetector | None = None) -> None:
        self.client = client
        self.detector = detector if detector is not None else RebahDetector()
        self.lock = threading.Lock()

        self.status_topic = topic("status")
        self.config_topic = topic("config")
        self.telemetry_topic = topic("edge/+/telemetry")
        self.control_topic = topic("control")
        self.parent_alert_topic = topic("alerts/parent")
        self.child_alert_topic = topic("alerts/child")

    def publish_status(self, reading: Telemetry | None = None, now_ms: int | None = None) -> None:
        msg = self.detector.build_status(now_ms=now_ms, reading=reading)
        self.client.publish(self.status_topic, msg.to_json(), qos=1, retain=True)

    def publish_config(self) -> None:
        payload = json.dumps(self.detector.config.to_dict(), allow_nan=False)
        self.client.publish(self.config_topic, payload, qos=1, retain=True)

    def _publish_alerts(self, decision: Decision) -> None:
        if decision.parent_alert is not None:
            self.client.publish(
                self.parent_alert_topic,
                decision.parent_alert.to_json(),
                qos=1,
                retain=False,
            )
        if decision.child_alert is not None:
            self.client.publish(
                self.child_alert_topic,
                decision.child_alert.to_json(),
                qos=1,
                retain=False,
            )

    def on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if getattr(reason_code, "is_failure", False):
            print(f"[POSTURE] Koneksi MQTT ditolak: {reason_code}")
            return
        client.subscribe([(self.telemetry_topic, 1), (self.control_topic, 1)])
        print(f"[POSTURE] Subscribed: {self.telemetry_topic} & {self.control_topic}")
        with self.lock:
            self.publish_config()
            self.publish_status()

    def handle_control(self, payload: bytes | str, now_ms: int | None = None) -> Decision | None:
        try:
            cmd = ControlCommand.from_json(payload)
        except ValueError as exc:
            print(f"[POSTURE] Control ditolak: {exc}")
            return None

        with self.lock:
            decision = self.detector.apply_command(cmd, now_ms=now_ms)
            print(f"[POSTURE] Control '{cmd.command}' -> {decision.status}: {decision.reason}")
            if cmd.command == "UPDATE_CONFIG":
                self.publish_config()
            self.publish_status(now_ms=now_ms)
            self._publish_alerts(decision)
            return decision

    def handle_telemetry(self, msg_topic: str, payload: bytes | str, now_ms: int | None = None) -> Decision | None:
        try:
            reading = Telemetry.from_json(payload)
            expected_topic = topic(f"edge/{reading.device_id}/telemetry")
            if msg_topic != expected_topic:
                print("[POSTURE] REJECTED: topic dan device_id tidak cocok")
                return None
        except ValueError as exc:
            print(f"[POSTURE] REJECTED malformed telemetry: {exc}")
            return None

        with self.lock:
            decision = self.detector.accept(reading, now_ms=now_ms)
            print(
                f"[POSTURE] {reading.device_id} #{reading.sequence}: "
                f"{decision.status} — {decision.reason}"
            )
            self.publish_status(reading=reading, now_ms=now_ms)
            self._publish_alerts(decision)
            return decision

    def on_message(self, client, userdata, message) -> None:
        if message.topic == self.control_topic:
            self.handle_control(message.payload)
            return
        self.handle_telemetry(message.topic, message.payload)

    def check_timeouts_and_publish(self, now_ms: int | None = None) -> Decision:
        with self.lock:
            prev_status = self.detector.status
            decision = self.detector.check_timeouts(now_ms=now_ms)
            if decision.status != prev_status:
                print(f"[POSTURE] Timeout transition: {prev_status} -> {decision.status}")
                self.publish_status(now_ms=now_ms)
            return decision


def main() -> None:
    ml_classifier = PostureMLClassifier()
    detector = RebahDetector(ml_classifier=ml_classifier)
    status_topic = topic("status")
    offline_lwt = build_offline_lwt_json(detector.config.lying_limit_sec)
    client = create_client("posture", will=(status_topic, offline_lwt))
    bridge = PostureServiceBridge(client, detector)

    client.on_connect = bridge.on_connect
    client.on_message = bridge.on_message

    print("[POSTURE] Menjalankan service RebahGuard (Hybrid ML + FSM). Tekan Ctrl+C untuk berhenti.")
    client.loop_start()
    try:
        while True:
            time.sleep(1.0)
            bridge.check_timeouts_and_publish()
    except KeyboardInterrupt:
        print("[POSTURE] Berhenti...")
    finally:
        try:
            info = client.publish(
                status_topic,
                build_offline_lwt_json(detector.config.lying_limit_sec),
                qos=1,
                retain=True,
            )
            info.wait_for_publish(timeout=3)
        except (RuntimeError, ValueError):
            pass
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
