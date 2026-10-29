"""NODE 3: RebahGuard FSM & multisensor fusion rules (pure Python, no MQTT dependency)."""

from dataclasses import dataclass
import math
import time
from typing import Any

from shared.config import DetectorConfig, LANDSCAPE, PORTRAIT
from shared.messages import AlertMessage, ControlCommand, StatusMessage, Telemetry


def lean_from_features(overhead_deg: float, roll_deg: float) -> float | None:
    """Rekonstruksi in-plane lean (0=portrait, 90=landscape) dari telemetry.

    roll_deg = atan2(|ax|, sqrt(ay^2+az^2)), overhead_deg = acos(az/g).
    |ax| = g*sin(r), |az| = g*|cos(t)|  ->  |ay| = g*sqrt(cos^2 r - cos^2 t).
    Return None kalau HP hampir datar (lean tidak valid).
    """
    r = math.radians(roll_deg)
    t = math.radians(overhead_deg)
    ay_sq = math.cos(r) ** 2 - math.cos(t) ** 2
    ax = math.sin(r)
    ay = math.sqrt(max(0.0, ay_sq))
    if ax < 1e-6 and ay < 1e-6:
        return None
    return math.degrees(math.atan2(ax, ay))


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    parent_alert: AlertMessage | None = None
    child_alert: AlertMessage | None = None


class RebahDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config: DetectorConfig = config if config is not None else DetectorConfig()
        self.armed: bool = True
        self.status: str = "WAITING"
        self.reason: str = "Menunggu telemetry dari Edge 1"

        self.in_lying_zone: bool = False
        self.orientation: str = PORTRAIT
        self._pending_orientation: str | None = None
        self._pending_since_ms: int = 0
        self.active_limit_sec: float = self.config.lying_limit_sec
        self.lying_elapsed_sec: float = 0.0
        self.nagging_active: bool = False
        self.alert_active: bool = False

        self.snooze_until_ms: int = 0
        self.prev_edge1_sent_ms: int | None = None
        self.last_edge1_ms: int | None = None
        self.last_edge2_ms: int | None = None

        self.last_alert_ms: int | None = None
        self.last_nag_ms: int | None = None
        self.last_shake_ms: int | None = None
        self.shake_streak: int = 0

        self.latest: dict[str, Telemetry] = {}
        self.last_seq: dict[tuple[str, str], int] = {}

    def _clear_active_alerts(self, reason: str, current_ms: int) -> AlertMessage | None:
        was_active = self.nagging_active or self.alert_active or self.status in ("ALERT_MATA", "TEGURAN_ORTU")
        self.nagging_active = False
        self.alert_active = False
        self.last_alert_ms = None
        self.last_nag_ms = None
        if was_active:
            return AlertMessage(
                target="child",
                event="CLEAR",
                reason=reason,
                lying_sec=0.0,
                at_ms=current_ms,
            )
        return None

    def set_armed(self, armed: bool, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.armed = bool(armed)
        if not self.armed:
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            self.shake_streak = 0
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        self.status = "POSISI_DUDUK_SEHAT" if "edge1" in self.latest else "WAITING"
        self.reason = "Monitoring diaktifkan (ARMED)"
        return Decision(self.status, self.reason)

    def snooze(self, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.snooze_until_ms = current_ms + int(self.config.snooze_duration_sec * 1000)
        self.lying_elapsed_sec = 0.0
        self.in_lying_zone = False
        self.shake_streak = 0
        clear_msg = self._clear_active_alerts("Snooze diaktifkan sementara", current_ms)
        self.status = "SNOOZED"
        self.reason = f"Snooze aktif selama {self.config.snooze_duration_sec:.0f} detik"
        return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

    def reset(self, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.lying_elapsed_sec = 0.0
        self.in_lying_zone = False
        self.snooze_until_ms = 0
        self.shake_streak = 0
        clear_msg = self._clear_active_alerts("State di-reset ke posisi sehat", current_ms)
        self.status = "POSISI_DUDUK_SEHAT"
        self.reason = "State di-reset oleh pengguna"
        return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

    def trigger_scold(self, reason: str = "Teguran manual dari orang tua", now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        if not self.armed:
            return Decision("DISARMED", "Tidak dapat mengirim teguran saat DISARMED")
        self.nagging_active = True
        self.last_nag_ms = current_ms
        self.status = "TEGURAN_ORTU"
        self.reason = reason
        child_alert = AlertMessage(
            target="child",
            event="TEGURAN_ORTU",
            reason=reason,
            lying_sec=round(self.lying_elapsed_sec, 4),
            at_ms=current_ms,
        )
        return Decision(self.status, self.reason, parent_alert=None, child_alert=child_alert)

    def update_config(self, partial_dict: dict[str, Any]) -> DetectorConfig:
        self.config = self.config.merged_with(partial_dict)
        return self.config

    def apply_command(self, cmd: ControlCommand | str | bytes, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        parsed = cmd if isinstance(cmd, ControlCommand) else ControlCommand.from_json(cmd)
        if parsed.command == "ARM":
            return self.set_armed(True, now_ms=current_ms)
        if parsed.command == "DISARM":
            return self.set_armed(False, now_ms=current_ms)
        if parsed.command == "SNOOZE":
            return self.snooze(now_ms=current_ms)
        if parsed.command == "RESET":
            return self.reset(now_ms=current_ms)
        if parsed.command == "SCOLD":
            return self.trigger_scold("Teguran langsung dari orang tua", now_ms=current_ms)
        if parsed.command == "UPDATE_CONFIG":
            self.update_config(parsed.params)
            self.reason = "Konfigurasi diperbarui secara live"
            return Decision(self.status, self.reason)
        raise ValueError(f"Unsupported command: {parsed.command}")

    def is_edge1_online(self, now_ms: int | None = None) -> bool:
        if self.last_edge1_ms is None:
            return False
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        return (current_ms - self.last_edge1_ms) <= self.config.edge_timeout_sec * 1000

    def is_edge2_online(self, now_ms: int | None = None) -> bool:
        if self.last_edge2_ms is None:
            return False
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        return (current_ms - self.last_edge2_ms) <= self.config.edge_timeout_sec * 1000

    def check_timeouts(self, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        if not self.armed:
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason)

        if self.last_edge1_ms is None or (current_ms - self.last_edge1_ms) > self.config.edge_timeout_sec * 1000:
            self.status = "EDGE_OFFLINE"
            self.reason = "Telemetry Edge 1 terputus atau belum diterima"
            return Decision(self.status, self.reason)

        return Decision(self.status, self.reason)

    def _update_orientation(self, data: Telemetry, current_ms: int) -> str:
        cfg = self.config
        if not cfg.use_roll:
            self.orientation = PORTRAIT
            return self.orientation
        tilt = data.overhead_deg
        if tilt < cfg.flat_gate_deg or tilt > 180.0 - cfg.flat_gate_deg:
            return self.orientation  # hampir datar: lean tidak valid, tahan
        lean = lean_from_features(tilt, data.roll_deg)
        if lean is None:
            return self.orientation
        if self.orientation == LANDSCAPE:
            raw = PORTRAIT if lean < cfg.roll_exit_deg else LANDSCAPE
        else:
            raw = LANDSCAPE if lean >= cfg.roll_enter_deg else PORTRAIT
        if raw == self.orientation:
            self._pending_orientation = None
            return self.orientation
        if self._pending_orientation != raw:
            self._pending_orientation = raw
            self._pending_since_ms = current_ms
        if (current_ms - self._pending_since_ms) >= cfg.orientation_confirm_sec * 1000:
            self.orientation = raw
            self._pending_orientation = None
        return self.orientation

    def _accept_edge2(self, data: Telemetry, current_ms: int) -> Decision:
        self.last_edge2_ms = current_ms

        if not self.armed:
            self.shake_streak = 0
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        if current_ms < self.snooze_until_ms:
            self.shake_streak = 0
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            clear_msg = self._clear_active_alerts("Snooze sedang aktif", current_ms)
            self.status = "SNOOZED"
            self.reason = "Monitoring ditunda sementara (SNOOZED)"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        is_shaking = (
            data.motion_rms >= self.config.shake_motion_min
            or data.gyro_rms >= self.config.shake_gyro_min
        )
        if is_shaking:
            self.shake_streak += 1
        else:
            self.shake_streak = 0

        if self.shake_streak >= self.config.shake_confirm_windows:
            cooldown_elapsed = (
                self.last_shake_ms is None
                or (current_ms - self.last_shake_ms) >= self.config.shake_cooldown_sec * 1000
            )
            edge1_rebahan = self.in_lying_zone and self.status in ("REBAHAN_AKTIF", "ALERT_MATA", "TEGURAN_ORTU")
            if cooldown_elapsed and edge1_rebahan:
                self.last_shake_ms = current_ms
                self.shake_streak = 0
                self.nagging_active = True
                self.last_nag_ms = current_ms
                self.status = "TEGURAN_ORTU"
                self.reason = "Orang tua menggoyangkan HP (Shake-to-Scold) saat anak rebahan"
                child_alert = AlertMessage(
                    target="child",
                    event="TEGURAN_ORTU",
                    reason=self.reason,
                    lying_sec=round(self.lying_elapsed_sec, 4),
                    at_ms=current_ms,
                )
                return Decision(self.status, self.reason, parent_alert=None, child_alert=child_alert)

        return Decision(self.status, self.reason)

    def _accept_edge1(self, data: Telemetry, current_ms: int) -> Decision:
        prev_sent_ms = self.prev_edge1_sent_ms
        self.prev_edge1_sent_ms = data.sent_at_ms
        self.last_edge1_ms = current_ms

        if not self.armed:
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        if current_ms < self.snooze_until_ms:
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            clear_msg = self._clear_active_alerts("Snooze sedang aktif", current_ms)
            self.status = "SNOOZED"
            self.reason = "Monitoring ditunda sementara (SNOOZED)"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        # Klasifikasi postur Edge 1: orientasi (portrait/landscape) -> threshold per orientasi
        orientation = self._update_orientation(data, current_ms)
        th = self.config.thresholds_for(orientation)
        self.active_limit_sec = th.lying_limit_sec

        is_face_down_table = (
            data.overhead_deg >= self.config.face_down_deg
            and data.motion_rms <= th.still_motion_max
            and data.gyro_rms <= th.still_gyro_max
        )

        if is_face_down_table:
            self.in_lying_zone = False
            self.lying_elapsed_sec = 0.0
            clear_msg = self._clear_active_alerts("HP ditaruh telungkup di meja", current_ms)
            self.status = "DITARUH_TELUNGKUP"
            self.reason = "HP ditaruh telungkup diam di permukaan meja"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        # Histeresis sudut: roll TIDAK lagi memicu rebahan sendiri, hanya memilih threshold.
        if self.in_lying_zone:
            self.in_lying_zone = data.overhead_deg > th.exit_deg
        else:
            self.in_lying_zone = data.overhead_deg >= th.enter_deg

        if not self.in_lying_zone:
            self.lying_elapsed_sec = 0.0
            clear_msg = self._clear_active_alerts("Anak kembali ke posisi duduk sehat", current_ms)
            self.status = "POSISI_DUDUK_SEHAT"
            self.reason = "Posisi penggunaan HP sehat (duduk/berdiri)"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        # in_lying_zone == True: akumulasi durasi rebahan dengan batas max_window_dt_sec
        if prev_sent_ms is not None:
            raw_dt = (data.sent_at_ms - prev_sent_ms) / 1000.0
            dt = min(self.config.max_window_dt_sec, max(0.0, raw_dt))
            self.lying_elapsed_sec = round(self.lying_elapsed_sec + dt, 4)

        if self.nagging_active:
            self.status = "TEGURAN_ORTU"
            self.reason = f"Teguran orang tua aktif (rebahan {self.lying_elapsed_sec:.1f}s)"
            child_alert = None
            if (
                self.last_nag_ms is None
                or (current_ms - self.last_nag_ms) >= self.config.nag_interval_sec * 1000
            ):
                self.last_nag_ms = current_ms
                child_alert = AlertMessage(
                    target="child",
                    event="TEGURAN_ORTU",
                    reason=self.reason,
                    lying_sec=round(self.lying_elapsed_sec, 4),
                    at_ms=current_ms,
                )
            return Decision(self.status, self.reason, parent_alert=None, child_alert=child_alert)

        if self.lying_elapsed_sec >= self.active_limit_sec:
            first_entry = not self.alert_active
            cooldown_ready = (
                self.last_alert_ms is None
                or (current_ms - self.last_alert_ms) >= self.config.alert_cooldown_sec * 1000
            )
            self.alert_active = True
            self.status = "ALERT_MATA"
            self.reason = (
                f"Rebahan melebihi batas ({self.lying_elapsed_sec:.1f}s >= {self.active_limit_sec:.1f}s)"
            )
            parent_alert = None
            child_alert = None
            if first_entry or cooldown_ready:
                self.last_alert_ms = current_ms
                parent_alert = AlertMessage(
                    target="parent",
                    event="ALERT_MATA",
                    reason=self.reason,
                    lying_sec=round(self.lying_elapsed_sec, 4),
                    at_ms=current_ms,
                )
                child_alert = AlertMessage(
                    target="child",
                    event="ALERT_MATA",
                    reason=self.reason,
                    lying_sec=round(self.lying_elapsed_sec, 4),
                    at_ms=current_ms,
                )
            return Decision(self.status, self.reason, parent_alert=parent_alert, child_alert=child_alert)

        self.status = "REBAHAN_AKTIF"
        self.reason = (
            f"Terdeteksi rebahan ({self.lying_elapsed_sec:.1f}s / {self.active_limit_sec:.1f}s)"
        )
        return Decision(self.status, self.reason, parent_alert=None, child_alert=None)

    def accept(self, data: Telemetry, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        if abs(current_ms - data.sent_at_ms) > self.config.stale_message_ms:
            return Decision("REJECTED_STALE", "Timestamp di luar jendela waktu yang diizinkan")

        key = (data.device_id, data.boot_id)
        if data.sequence <= self.last_seq.get(key, -1):
            return Decision("REJECTED_REPLAY", "Sequence duplikat atau mundur untuk boot_id ini")
        self.last_seq[key] = data.sequence
        self.latest[data.device_id] = data

        if data.device_id == "edge2":
            return self._accept_edge2(data, current_ms)
        return self._accept_edge1(data, current_ms)

    def build_status(self, now_ms: int | None = None, reading: Telemetry | None = None) -> StatusMessage:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        return StatusMessage(
            status=self.status,
            reason=self.reason,
            armed=self.armed,
            nagging_active=self.nagging_active,
            lying_elapsed_sec=round(self.lying_elapsed_sec, 4),
            lying_limit_sec=self.active_limit_sec,
            edge1_online=self.is_edge1_online(current_ms),
            edge2_online=self.is_edge2_online(current_ms),
            at_ms=current_ms,
            device_id=reading.device_id if reading else None,
            sequence=reading.sequence if reading else None,
        )

