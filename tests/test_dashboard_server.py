"""Tests for local dashboard HTTP server and PostureServiceBridge MQTT publish/retain policy."""

import json
import os
from unittest import TestCase
from unittest.mock import MagicMock, patch
from urllib.request import urlopen

from nodes.posture_service.main import PostureServiceBridge, build_offline_lwt_json
from shared.dashboard_server import public_browser_config, start_dashboard_server
from shared.messages import StatusMessage, Telemetry
from shared.mqtt_connection import topic

NOW_MS = 1_790_000_000_000


class DashboardServerTests(TestCase):
    def test_local_http_server_and_no_credentials(self):
        with patch.dict(
            os.environ,
            {
                "PROJECT_ID": "kelas_test",
                "MQTT_HOST": "broker.hivemq.com",
                "MQTT_PASSWORD": "DO_NOT_LEAK",
                "MQTT_USERNAME": "",
            },
        ):
            server, thread, url = start_dashboard_server(port=0)
            try:
                with urlopen(url + "/") as result:
                    self.assertEqual(result.status, 200)
                    body = result.read()
                    self.assertIn(b"RebahGuard Dashboard", body)
                    self.assertIn(b"Mode Kalibrasi", body)
                    self.assertIn(b'id="ai-monitor-panel"', body)
                    self.assertIn(b'id="h-ai-monitor"', body)
                    self.assertIn(b'id="hud-prob-list"', body)
                    self.assertIn(b"ALLOWED_POSTURE_LABELS", body)
                    self.assertIn(b"prefers-reduced-motion", body)
                    self.assertNotIn(b"innerHTML", body)
                with urlopen(url + "/config.json") as result:
                    self.assertEqual(result.status, 200)
                    config = json.loads(result.read())
                self.assertEqual(config["project_id"], "kelas_test")
                self.assertEqual(config["websocket_url"], "wss://broker.hivemq.com:8884/mqtt")
                self.assertTrue(config["automatic_connection"])
                self.assertNotIn("DO_NOT_LEAK", json.dumps(config))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_private_broker_does_not_expose_password(self):
        with patch.dict(
            os.environ,
            {
                "PROJECT_ID": "private_test",
                "MQTT_HOST": "private.example.com",
                "MQTT_WS_URL": "wss://private.example.com/mqtt",
                "MQTT_USERNAME": "secret-user",
                "MQTT_PASSWORD": "secret-pass",
            },
        ):
            config = public_browser_config()
            self.assertFalse(config["automatic_connection"])
            self.assertNotIn("secret", json.dumps(config))


class PostureServiceBridgeTests(TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.bridge = PostureServiceBridge(self.mock_client)

    def test_on_connect_subscribes_and_publishes_retained_config_and_status(self):
        reason = MagicMock()
        reason.is_failure = False
        self.bridge.on_connect(self.mock_client, None, None, reason, None)

        self.mock_client.subscribe.assert_called_once_with(
            [(topic("edge/+/telemetry"), 1), (topic("control"), 1)]
        )
        published_topics = {
            call.args[0]: (call.kwargs.get("qos"), call.kwargs.get("retain"))
            for call in self.mock_client.publish.call_args_list
        }
        self.assertEqual(published_topics[topic("config")], (1, True))
        self.assertEqual(published_topics[topic("status")], (1, True))

    def test_alerts_published_with_retain_false_and_config_update_retained(self):
        # Trigger SCOLD via control -> must publish status (retain=True) and alerts/child (retain=False)
        self.mock_client.reset_mock()
        self.bridge.handle_control('{"command": "SCOLD"}', now_ms=NOW_MS)

        calls = {
            call.args[0]: (call.kwargs.get("qos"), call.kwargs.get("retain"))
            for call in self.mock_client.publish.call_args_list
        }
        self.assertEqual(calls[topic("status")], (1, True))
        self.assertEqual(calls[topic("alerts/child")], (1, False))

        # UPDATE_CONFIG -> must publish config (retain=True) and status (retain=True)
        self.mock_client.reset_mock()
        self.bridge.handle_control(
            '{"command": "UPDATE_CONFIG", "params": {"lying_limit_sec": 4.0}}',
            now_ms=NOW_MS + 500,
        )
        calls_cfg = {
            call.args[0]: (call.kwargs.get("qos"), call.kwargs.get("retain"))
            for call in self.mock_client.publish.call_args_list
        }
        self.assertEqual(calls_cfg[topic("config")], (1, True))
        self.assertEqual(calls_cfg[topic("status")], (1, True))

    def test_timeout_loop_publishes_edge_offline(self):
        t0 = Telemetry("edge1", "boot-1", 0, NOW_MS, 45.0, 5.0, 0.1, 0.05)
        self.bridge.handle_telemetry(topic("edge/edge1/telemetry"), t0.to_json(), now_ms=NOW_MS)
        self.mock_client.reset_mock()

        # Setelah 9 detik (> edge_timeout_sec = 8.0s), check_timeouts_and_publish mem-publish EDGE_OFFLINE
        dec = self.bridge.check_timeouts_and_publish(now_ms=NOW_MS + 9000)
        self.assertEqual(dec.status, "EDGE_OFFLINE")
        self.mock_client.publish.assert_called_once()
        pub_call = self.mock_client.publish.call_args
        self.assertEqual(pub_call.args[0], topic("status"))
        self.assertEqual(pub_call.kwargs, {"qos": 1, "retain": True})
        parsed_status = StatusMessage.from_json(pub_call.args[1])
        self.assertEqual(parsed_status.status, "EDGE_OFFLINE")

    def test_offline_lwt_json_valid_status_message(self):
        lwt = build_offline_lwt_json(10.0, now_ms=NOW_MS)
        msg = StatusMessage.from_json(lwt)
        self.assertEqual(msg.status, "OFFLINE")
        self.assertFalse(msg.armed)
