"""ONE shared MQTT connection implementation, used by ALL nodes."""

import os
import uuid

from dotenv import load_dotenv
import paho.mqtt.client as mqtt

load_dotenv()

# Default: HiveMQ public broker. No account needed; anyone can read public topics.
DEFAULT_HOST = "broker.hivemq.com"
DEFAULT_PROJECT_ID = "iot_kelompok_14"  # Change this in .env to avoid topic collisions with other groups


def project_id() -> str:
    value = os.getenv("PROJECT_ID", DEFAULT_PROJECT_ID)
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError("PROJECT_ID may only contain letters, numbers, - and _")
    return value


def topic(suffix: str) -> str:
    return f"rebahguard/{project_id()}/{suffix}"


def create_client(role: str, will: tuple[str, str] | None = None) -> mqtt.Client:
    """Connect to the broker from .env. `will` = (topic, payload) the broker
    publishes (retained) if this client disconnects unexpectedly."""
    host = os.getenv("MQTT_HOST") or DEFAULT_HOST
    if project_id() == DEFAULT_PROJECT_ID:
        print(f"[MQTT] Warning: PROJECT_ID masih '{DEFAULT_PROJECT_ID}'. Ganti di .env agar topic tidak bercampur dengan kelompok lain.")

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{project_id()}-{role}-{uuid.uuid4().hex[:8]}",
        protocol=mqtt.MQTTv311,
    )
    username = os.getenv("MQTT_USERNAME", "")
    password = os.getenv("MQTT_PASSWORD", "")
    if username:
        client.username_pw_set(username, password)

    use_tls = os.getenv("MQTT_TLS", "true").lower() == "true"
    if use_tls:
        client.tls_set()
    else:
        if os.getenv("ALLOW_INSECURE_LOCAL") != "true":
            raise RuntimeError("TLS disabled. Only allow this for local development via ALLOW_INSECURE_LOCAL=true")

    if will is not None:
        client.will_set(will[0], will[1], qos=1, retain=True)

    port = int(os.getenv("MQTT_PORT", "8883" if use_tls else "1883"))
    client.connect(host, port, keepalive=30)
    return client
