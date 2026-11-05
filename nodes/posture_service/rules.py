"""NODE 3: RebahGuard FSM & multisensor fusion rules (pure Python, no MQTT dependency)."""

from collections import Counter, deque
from dataclasses import dataclass
import math
import time
from typing import Any

from shared.config import DetectorConfig, LANDSCAPE, OrientationThresholds, PORTRAIT
from shared.messages import AlertMessage, ControlCommand, StatusMessage, Telemetry, VALID_POSTURE_LABELS
from shared.processing import lean_from_features, safe_lean_deg


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    parent_alert: AlertMessage | None = None
    child_alert: AlertMessage | None = None


class RebahDetector:
    def __init__(
        self,
        config: DetectorConfig | None = None,
        ml_classifier: Any | None = None,
    ) -> None:
        self.config: DetectorConfig = config if config is not None else DetectorConfig()
        self.ml_classifier: Any | None = ml_classifier
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

        self.detection_mode: str = "rule"
        self.ml_prediction: str | None = None
        self.ml_confidence: float | None = None
        self.ml_probabilities: dict[str, float] | None = None
        self.ml_probabilities_source: str | None = None
        self._ml_buffer: deque[str] = deque()
        self._prob_window: deque[dict[str, float]] = deque()
        self._effective_posture: str = "POSISI_DUDUK_SEHAT"

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

    def _reset_ml_state(self) -> None:
        self._ml_buffer.clear()
        self._prob_window.clear()
        self.detection_mode = "rule"
        self.ml_prediction = None
        self.ml_confidence = None
        self.ml_probabilities = None
        self.ml_probabilities_source = None

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
            self._reset_ml_state()
            self._effective_posture = "POSISI_DUDUK_SEHAT"
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        self._reset_ml_state()
        self.status = "POSISI_DUDUK_SEHAT" if "edge1" in self.latest else "WAITING"
        self.reason = "Monitoring diaktifkan (ARMED)"
        return Decision(self.status, self.reason)

    def snooze(self, now_ms: int | None = None) -> Decision:
        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.snooze_until_ms = current_ms + int(self.config.snooze_duration_sec * 1000)
        self.lying_elapsed_sec = 0.0
        self.in_lying_zone = False
        self.shake_streak = 0
        self._reset_ml_state()
        self._effective_posture = "POSISI_DUDUK_SEHAT"
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
        self._reset_ml_state()
        self._effective_posture = "POSISI_DUDUK_SEHAT"
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
        if not self.config.use_ml:
            self._ml_buffer.clear()
            self._prob_window.clear()
            self.detection_mode = "rule"
            self.ml_prediction = None
            self.ml_confidence = None
        else:
            while len(self._ml_buffer) > self.config.ml_smoothing_window:
                self._ml_buffer.popleft()
            while len(self._prob_window) > self.config.ml_smoothing_window:
                self._prob_window.popleft()
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
            self._reset_ml_state()
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason)

        if self.last_edge1_ms is None or (current_ms - self.last_edge1_ms) > self.config.edge_timeout_sec * 1000:
            self._reset_ml_state()
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
            return self.orientation  # hampir datar: lean tidak valid, tahan orientasi terakhir
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

    def _majority_vote_posture(self, raw_label: str) -> str:
        self._ml_buffer.append(raw_label)
        while len(self._ml_buffer) > self.config.ml_smoothing_window:
            self._ml_buffer.popleft()

        counts = Counter(self._ml_buffer)
        max_votes = max(counts.values())
        winners = [lbl for lbl, cnt in counts.items() if cnt == max_votes]
        if len(winners) == 1:
            return winners[0]
        # Jika seri (misal 3 kelas berbeda di window=3), pertahankan postur efektif sebelumnya
        if self.in_lying_zone:
            return "REBAHAN_AKTIF"
        return self._effective_posture

    def _smooth_ml_probabilities(self, probs: dict[str, float]) -> dict[str, float]:
        cleaned = {k: round(max(0.0, min(1.0, float(probs.get(k, 0.0)))), 4) for k in VALID_POSTURE_LABELS}
        self._prob_window.append(cleaned)
        while len(self._prob_window) > self.config.ml_smoothing_window:
            self._prob_window.popleft()
        n = len(self._prob_window)
        return {
            k: round(sum(p[k] for p in self._prob_window) / n, 4)
            for k in VALID_POSTURE_LABELS
        }

    def _compute_heuristic_probabilities(
        self,
        data: Telemetry,
        th: OrientationThresholds,
    ) -> dict[str, float]:
        tilt = float(data.overhead_deg)
        mid_deg = 0.5 * (th.exit_deg + th.enter_deg)
        span_deg = max(5.0, th.enter_deg - th.exit_deg)

        w_duduk = 1.0 / (1.0 + math.exp((tilt - mid_deg) / (span_deg * 0.5)))

        still_factor = (
            1.0
            if (data.motion_rms <= th.still_motion_max * 2.0 and data.gyro_rms <= th.still_gyro_max * 2.0)
            else 0.25
        )
        w_telungkup = still_factor / (1.0 + math.exp((self.config.face_down_deg - 5.0 - tilt) / 6.0))
        w_rebahan = (1.0 - w_duduk) * (1.0 - 0.7 * w_telungkup)

        total = max(1e-9, w_duduk + w_rebahan + w_telungkup)
        p_duduk = round(w_duduk / total, 4)
        p_telungkup = round(w_telungkup / total, 4)
        p_rebahan = round(max(0.0, 1.0 - p_duduk - p_telungkup), 4)
        return {
            "POSISI_DUDUK_SEHAT": p_duduk,
            "DITARUH_TELUNGKUP": p_telungkup,
            "REBAHAN_AKTIF": p_rebahan,
        }

    def _run_single_ml_inference(
        self,
        data: Telemetry,
    ) -> tuple[str, float, dict[str, float]] | None:
        if self.ml_classifier is None:
            return None

        if hasattr(self.ml_classifier, "predict_with_proba"):
            res = self.ml_classifier.predict_with_proba(
                data.overhead_deg,
                data.roll_deg,
                data.motion_rms,
                data.gyro_rms,
            )
            if (
                isinstance(res, tuple)
                and len(res) == 3
                and isinstance(res[0], str)
                and isinstance(res[2], dict)
            ):
                raw_label, conf, probs = res
                if raw_label in VALID_POSTURE_LABELS and set(probs.keys()) == set(VALID_POSTURE_LABELS):
                    return raw_label, float(conf), probs
                return None

        if hasattr(self.ml_classifier, "predict"):
            pred = self.ml_classifier.predict(
                data.overhead_deg,
                data.roll_deg,
                data.motion_rms,
                data.gyro_rms,
            )
            if pred is not None and len(pred) == 2:
                raw_label, conf = pred
                if raw_label in VALID_POSTURE_LABELS and math.isfinite(float(conf)):
                    conf_f = round(max(0.0, min(1.0, float(conf))), 4)
                    other = round((1.0 - conf_f) / 2.0, 4)
                    rem = round(max(0.0, 1.0 - conf_f - other), 4)
                    others = [lbl for lbl in VALID_POSTURE_LABELS if lbl != raw_label]
                    probs = {
                        raw_label: conf_f,
                        others[0]: other,
                        others[1]: rem,
                    }
                    return raw_label, conf_f, probs
        return None

    def _accept_edge2(self, data: Telemetry, current_ms: int) -> Decision:
        self.last_edge2_ms = current_ms

        if not self.armed:
            self.shake_streak = 0
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            self._reset_ml_state()
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        if current_ms < self.snooze_until_ms:
            self.shake_streak = 0
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            self._reset_ml_state()
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
            self._reset_ml_state()
            self._effective_posture = "POSISI_DUDUK_SEHAT"
            clear_msg = self._clear_active_alerts("Sistem dinonaktifkan (DISARMED)", current_ms)
            self.status = "DISARMED"
            self.reason = "Monitoring dinonaktifkan oleh orang tua"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        if current_ms < self.snooze_until_ms:
            self.lying_elapsed_sec = 0.0
            self.in_lying_zone = False
            self._reset_ml_state()
            self._effective_posture = "POSISI_DUDUK_SEHAT"
            clear_msg = self._clear_active_alerts("Snooze sedang aktif", current_ms)
            self.status = "SNOOZED"
            self.reason = "Monitoring ditunda sementara (SNOOZED)"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        # 1) Klasifikasi orientasi (portrait/landscape) -> threshold per orientasi
        orientation = self._update_orientation(data, current_ms)
        th = self.config.thresholds_for(orientation)
        self.active_limit_sec = th.lying_limit_sec

        # 2) Kasus A — HARD-VETO MEJA (Selalu Menang atas ML)
        is_face_down_table = (
            data.overhead_deg >= self.config.face_down_deg
            and data.motion_rms <= th.still_motion_max
            and data.gyro_rms <= th.still_gyro_max
        )

        if is_face_down_table:
            self._reset_ml_state()
            self.ml_probabilities = {
                "DITARUH_TELUNGKUP": 1.0,
                "POSISI_DUDUK_SEHAT": 0.0,
                "REBAHAN_AKTIF": 0.0,
            }
            self.ml_probabilities_source = "veto"
            self._effective_posture = "DITARUH_TELUNGKUP"
            self.in_lying_zone = False
            self.lying_elapsed_sec = 0.0
            clear_msg = self._clear_active_alerts("HP ditaruh telungkup di meja", current_ms)
            self.status = "DITARUH_TELUNGKUP"
            self.reason = "HP ditaruh telungkup diam di permukaan meja"
            return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

        # 3) Kasus B — Inferensi ML Tunggal + Temporal Smoothing (dengan Fallback ke Rule-Based)
        ml_used = False
        if self.config.use_ml and self.ml_classifier is not None:
            ml_result = self._run_single_ml_inference(data)
            if ml_result is not None:
                raw_label, conf, probs = ml_result
                if raw_label in VALID_POSTURE_LABELS and conf >= self.config.ml_confidence_min:
                    ml_used = True
                    smoothed_label = self._majority_vote_posture(raw_label)
                    avg_probs = self._smooth_ml_probabilities(probs)
                    self.detection_mode = "ml"
                    self.ml_prediction = smoothed_label
                    self.ml_confidence = round(float(conf), 4)
                    self.ml_probabilities = avg_probs
                    self.ml_probabilities_source = "ml"
                    self._effective_posture = smoothed_label

                    if smoothed_label == "DITARUH_TELUNGKUP":
                        self.in_lying_zone = False
                        self.lying_elapsed_sec = 0.0
                        clear_msg = self._clear_active_alerts("HP ditaruh di meja (ML)", current_ms)
                        self.status = "DITARUH_TELUNGKUP"
                        self.reason = f"HP ditaruh diam di permukaan meja (AI RF {conf:.0%})"
                        return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

                    if smoothed_label == "POSISI_DUDUK_SEHAT":
                        self.in_lying_zone = False
                        self.lying_elapsed_sec = 0.0
                        clear_msg = self._clear_active_alerts("Anak kembali ke posisi duduk sehat", current_ms)
                        self.status = "POSISI_DUDUK_SEHAT"
                        self.reason = f"Posisi penggunaan HP sehat (AI RF {conf:.0%})"
                        return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

                    # smoothed_label == "REBAHAN_AKTIF"
                    self.in_lying_zone = True

        # 4) Kasus C — Fallback Rule-Based jika ML nonaktif / model belum ada / confidence rendah
        if not ml_used:
            self._reset_ml_state()
            self.ml_probabilities = self._compute_heuristic_probabilities(data, th)
            self.ml_probabilities_source = "heuristic"
            if self.in_lying_zone:
                self.in_lying_zone = data.overhead_deg > th.exit_deg
            else:
                self.in_lying_zone = data.overhead_deg >= th.enter_deg

            if not self.in_lying_zone:
                self._effective_posture = "POSISI_DUDUK_SEHAT"
                self.lying_elapsed_sec = 0.0
                clear_msg = self._clear_active_alerts("Anak kembali ke posisi duduk sehat", current_ms)
                self.status = "POSISI_DUDUK_SEHAT"
                self.reason = "Posisi penggunaan HP sehat (duduk/berdiri)"
                return Decision(self.status, self.reason, parent_alert=None, child_alert=clear_msg)

            self._effective_posture = "REBAHAN_AKTIF"

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
            orientation=self.orientation,
            detection_mode=self.detection_mode,
            ml_prediction=self.ml_prediction,
            ml_confidence=self.ml_confidence,
            ml_probabilities=self.ml_probabilities,
            ml_probabilities_source=self.ml_probabilities_source,
        )
