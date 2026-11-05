"""MQTT message schema: strict contracts between edge nodes, posture service, and dashboard."""

from dataclasses import asdict, dataclass, field
import json
import math
import time
from typing import Any

from shared.config import DetectorConfig, LANDSCAPE, PORTRAIT

VALID_DEVICES = ("edge1", "edge2")
VALID_ORIENTATIONS = (PORTRAIT, LANDSCAPE)
VALID_DETECTION_MODES = ("rule", "ml")
VALID_POSTURE_LABELS = ("POSISI_DUDUK_SEHAT", "DITARUH_TELUNGKUP", "REBAHAN_AKTIF")
VALID_PROBA_SOURCES = ("ml", "heuristic", "veto")
VALID_ALERT_TARGETS = ("parent", "child")
VALID_ALERT_EVENTS = ("ALERT_MATA", "TEGURAN_ORTU", "CLEAR")
VALID_COMMANDS = ("ARM", "DISARM", "SCOLD", "SNOOZE", "RESET", "UPDATE_CONFIG")
FALLBACK_STRING_COMMANDS = ("ARM", "DISARM", "SCOLD", "SNOOZE", "RESET")


def _require_non_empty_str(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Invalid {name}: must be a non-empty string")
    return value


def _require_non_neg_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid {name}: must be a non-negative integer")
    return value


def _require_finite_float(value: Any, name: str, min_val: float = 0.0, max_val: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}: must be a finite number")
    fval = float(value)
    if fval < min_val:
        raise ValueError(f"Invalid {name}: must be >= {min_val}")
    if max_val is not None and fval > max_val:
        raise ValueError(f"Invalid {name}: must be <= {max_val}")
    return fval


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Invalid {name}: must be a boolean")
    return value


@dataclass(frozen=True)
class Telemetry:
    device_id: str
    boot_id: str
    sequence: int
    sent_at_ms: int
    overhead_deg: float
    roll_deg: float
    motion_rms: float
    gyro_rms: float

    def __post_init__(self) -> None:
        _require_non_empty_str(self.device_id, "device_id")
        if self.device_id not in VALID_DEVICES:
            raise ValueError(f"Unknown edge device: {self.device_id}")
        _require_non_empty_str(self.boot_id, "boot_id")
        _require_non_neg_int(self.sequence, "sequence")
        _require_non_neg_int(self.sent_at_ms, "sent_at_ms")
        object.__setattr__(self, "overhead_deg", _require_finite_float(self.overhead_deg, "overhead_deg", 0.0, 180.0))
        object.__setattr__(self, "roll_deg", _require_finite_float(self.roll_deg, "roll_deg", 0.0, 180.0))
        object.__setattr__(self, "motion_rms", _require_finite_float(self.motion_rms, "motion_rms", 0.0))
        object.__setattr__(self, "gyro_rms", _require_finite_float(self.gyro_rms, "gyro_rms", 0.0))

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)

    @staticmethod
    def from_json(payload: str | bytes) -> "Telemetry":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JSON in Telemetry: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Telemetry must be a JSON object")
        return Telemetry(
            device_id=value.get("device_id"),
            boot_id=value.get("boot_id"),
            sequence=value.get("sequence"),
            sent_at_ms=value.get("sent_at_ms"),
            overhead_deg=value.get("overhead_deg"),
            roll_deg=value.get("roll_deg"),
            motion_rms=value.get("motion_rms"),
            gyro_rms=value.get("gyro_rms"),
        )

    @staticmethod
    def make(
        device_id: str,
        boot_id: str,
        sequence: int,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
        sent_at_ms: int | None = None,
    ) -> "Telemetry":
        now_ms = sent_at_ms if sent_at_ms is not None else int(time.time() * 1000)
        return Telemetry(
            device_id=device_id,
            boot_id=boot_id,
            sequence=sequence,
            sent_at_ms=now_ms,
            overhead_deg=overhead_deg,
            roll_deg=roll_deg,
            motion_rms=motion_rms,
            gyro_rms=gyro_rms,
        )


@dataclass(frozen=True)
class AlertMessage:
    target: str
    event: str
    reason: str
    lying_sec: float
    at_ms: int

    def __post_init__(self) -> None:
        _require_non_empty_str(self.target, "target")
        if self.target not in VALID_ALERT_TARGETS:
            raise ValueError(f"Invalid target: {self.target}")
        _require_non_empty_str(self.event, "event")
        if self.event not in VALID_ALERT_EVENTS:
            raise ValueError(f"Invalid event: {self.event}")
        _require_non_empty_str(self.reason, "reason")
        object.__setattr__(self, "lying_sec", _require_finite_float(self.lying_sec, "lying_sec", 0.0))
        _require_non_neg_int(self.at_ms, "at_ms")

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)

    @staticmethod
    def from_json(payload: str | bytes) -> "AlertMessage":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JSON in AlertMessage: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("AlertMessage must be a JSON object")
        return AlertMessage(
            target=value.get("target"),
            event=value.get("event"),
            reason=value.get("reason"),
            lying_sec=value.get("lying_sec"),
            at_ms=value.get("at_ms"),
        )


@dataclass(frozen=True)
class StatusMessage:
    status: str
    reason: str
    armed: bool
    nagging_active: bool
    lying_elapsed_sec: float
    lying_limit_sec: float
    edge1_online: bool
    edge2_online: bool
    at_ms: int
    device_id: str | None = None
    sequence: int | None = None
    orientation: str = PORTRAIT
    detection_mode: str = "rule"
    ml_prediction: str | None = None
    ml_confidence: float | None = None
    ml_probabilities: dict[str, float] | None = None
    ml_probabilities_source: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.status, "status")
        _require_non_empty_str(self.reason, "reason")
        _require_bool(self.armed, "armed")
        _require_bool(self.nagging_active, "nagging_active")
        object.__setattr__(
            self, "lying_elapsed_sec", _require_finite_float(self.lying_elapsed_sec, "lying_elapsed_sec", 0.0)
        )
        object.__setattr__(
            self, "lying_limit_sec", _require_finite_float(self.lying_limit_sec, "lying_limit_sec", 1e-6)
        )
        _require_bool(self.edge1_online, "edge1_online")
        _require_bool(self.edge2_online, "edge2_online")
        _require_non_neg_int(self.at_ms, "at_ms")
        if self.device_id is not None:
            _require_non_empty_str(self.device_id, "device_id")
            if self.device_id not in VALID_DEVICES:
                raise ValueError(f"Invalid device_id: {self.device_id}")
        if self.sequence is not None:
            _require_non_neg_int(self.sequence, "sequence")
        _require_non_empty_str(self.orientation, "orientation")
        if self.orientation not in VALID_ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {self.orientation}")
        _require_non_empty_str(self.detection_mode, "detection_mode")
        if self.detection_mode not in VALID_DETECTION_MODES:
            raise ValueError(f"Invalid detection_mode: {self.detection_mode}")
        if self.ml_prediction is not None:
            _require_non_empty_str(self.ml_prediction, "ml_prediction")
            if self.ml_prediction not in VALID_POSTURE_LABELS:
                raise ValueError(f"Invalid ml_prediction: {self.ml_prediction}")
        if self.ml_confidence is not None:
            object.__setattr__(
                self,
                "ml_confidence",
                _require_finite_float(self.ml_confidence, "ml_confidence", 0.0, 1.0),
            )

        if (self.ml_probabilities is None) != (self.ml_probabilities_source is None):
            raise ValueError(
                "ml_probabilities and ml_probabilities_source must both be None or both be provided"
            )

        if self.ml_probabilities is not None:
            if not isinstance(self.ml_probabilities, dict):
                raise ValueError("ml_probabilities must be a dict")
            if set(self.ml_probabilities.keys()) != set(VALID_POSTURE_LABELS):
                raise ValueError(
                    f"ml_probabilities keys must be exactly {VALID_POSTURE_LABELS}"
                )
            cleaned_probs: dict[str, float] = {}
            for k, v in self.ml_probabilities.items():
                cleaned_probs[k] = _require_finite_float(v, f"ml_probabilities[{k}]", 0.0, 1.0)
            object.__setattr__(self, "ml_probabilities", cleaned_probs)

        if self.ml_probabilities_source is not None:
            _require_non_empty_str(self.ml_probabilities_source, "ml_probabilities_source")
            if self.ml_probabilities_source not in VALID_PROBA_SOURCES:
                raise ValueError(f"Invalid ml_probabilities_source: {self.ml_probabilities_source}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)

    @staticmethod
    def from_json(payload: str | bytes) -> "StatusMessage":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JSON in StatusMessage: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("StatusMessage must be a JSON object")
        return StatusMessage(
            status=value.get("status"),
            reason=value.get("reason"),
            armed=value.get("armed"),
            nagging_active=value.get("nagging_active"),
            lying_elapsed_sec=value.get("lying_elapsed_sec"),
            lying_limit_sec=value.get("lying_limit_sec"),
            edge1_online=value.get("edge1_online"),
            edge2_online=value.get("edge2_online"),
            at_ms=value.get("at_ms"),
            device_id=value.get("device_id"),
            sequence=value.get("sequence"),
            orientation=value.get("orientation") or "portrait",
            detection_mode=value.get("detection_mode") or "rule",
            ml_prediction=value.get("ml_prediction"),
            ml_confidence=value.get("ml_confidence"),
            ml_probabilities=value.get("ml_probabilities"),
            ml_probabilities_source=value.get("ml_probabilities_source"),
        )


@dataclass(frozen=True)
class ControlCommand:
    command: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.command, "command")
        cmd_upper = self.command.strip().upper()
        if cmd_upper not in VALID_COMMANDS:
            raise ValueError(f"Unknown control command: {self.command}")
        object.__setattr__(self, "command", cmd_upper)
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        if cmd_upper == "UPDATE_CONFIG":
            if not self.params:
                raise ValueError("UPDATE_CONFIG requires non-empty params dictionary")
            DetectorConfig().merged_with(self.params)

    def to_json(self) -> str:
        return json.dumps({"command": self.command, "params": self.params}, allow_nan=False)

    @staticmethod
    def from_json(payload: str | bytes) -> "ControlCommand":
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Control payload must be valid UTF-8") from exc
        elif isinstance(payload, str):
            text = payload
        else:
            raise ValueError("Control payload must be str or bytes")

        stripped = text.strip()
        if not stripped:
            raise ValueError("Empty control payload")

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            cmd = stripped.upper()
            if cmd in FALLBACK_STRING_COMMANDS:
                return ControlCommand(command=cmd, params={})
            raise ValueError(f"Invalid control command payload: {stripped[:40]}")

        if isinstance(parsed, str):
            cmd = parsed.strip().upper()
            if cmd in FALLBACK_STRING_COMMANDS:
                return ControlCommand(command=cmd, params={})
            raise ValueError(f"Unknown string command: {parsed}")

        if not isinstance(parsed, dict):
            raise ValueError("Control JSON must be an object")

        raw_cmd = parsed.get("command")
        if type(raw_cmd) is not str or not raw_cmd.strip():
            raise ValueError("Control JSON missing valid 'command' string")

        raw_params = parsed.get("params")
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError("Control JSON 'params' must be an object")

        return ControlCommand(command=raw_cmd.strip().upper(), params=raw_params)
