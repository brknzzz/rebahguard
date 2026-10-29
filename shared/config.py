"""Centralized configuration for RebahGuard posture & interaction detection."""

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Optional

PORTRAIT = "portrait"
LANDSCAPE = "landscape"


@dataclass(frozen=True)
class OrientationThresholds:
    enter_deg: float
    exit_deg: float
    lying_limit_sec: float
    still_motion_max: float
    still_gyro_max: float


def screen_tilt_deg(gx: float, gy: float, gz: float) -> Optional[float]:
    """0 = layar ke atas, 90 = tegak, 180 = layar ke bawah. Sama untuk portrait & landscape."""
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    if g < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, gz / g))))


def in_plane_lean_deg(gx: float, gy: float) -> float:
    """0 = portrait (sumbu Y sejajar gravitasi), 90 = landscape. Tidak peduli tanda sumbu,
    jadi landscape kiri/kanan dan portrait terbalik ditangani sama."""
    return math.degrees(math.atan2(abs(gx), abs(gy)))


@dataclass(frozen=True)
class DetectorConfig:
    # --- Portrait ---
    overhead_enter_deg: float = 110.0
    overhead_exit_deg: float = 95.0
    face_down_deg: float = 175.0
    still_motion_max: float = 0.05
    still_gyro_max: float = 0.05
    lying_limit_sec: float = 10.0

    # --- Klasifikasi orientasi (portrait vs landscape) ---
    use_roll: bool = True              # False = semua dianggap portrait
    roll_enter_deg: float = 60.0       # lean >= ini -> landscape
    roll_exit_deg: float = 45.0        # lean < ini -> kembali portrait
    flat_gate_deg: float = 20.0        # tilt < 20 atau > 160: roll tidak valid, orientasi ditahan
    orientation_confirm_sec: float = 0.5

    # --- Landscape (game / nonton) ---
    landscape_overhead_enter_deg: float = 80.0
    landscape_overhead_exit_deg: float = 70.0
    landscape_lying_limit_sec: float = 15.0
    landscape_still_motion_max: float = 0.15   # jempol main game = lebih banyak gerak
    landscape_still_gyro_max: float = 0.12

    # --- Umum ---
    max_window_dt_sec: float = 3.0
    shake_motion_min: float = 2.0
    shake_gyro_min: float = 1.6
    shake_confirm_windows: int = 2
    shake_cooldown_sec: float = 5.0
    alert_cooldown_sec: float = 8.0
    nag_interval_sec: float = 2.5
    edge_timeout_sec: float = 8.0
    stale_message_ms: int = 30_000
    snooze_duration_sec: float = 30.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.use_roll) is not bool:
            raise ValueError("use_roll must be a boolean")

        int_fields = ("shake_confirm_windows", "stale_message_ms")
        for name in int_fields:
            val = getattr(self, name)
            if type(val) is not int or val <= 0:
                raise ValueError(f"{name} must be a positive integer")

        float_fields = (
            "overhead_enter_deg",
            "overhead_exit_deg",
            "face_down_deg",
            "still_motion_max",
            "still_gyro_max",
            "roll_enter_deg",
            "roll_exit_deg",
            "flat_gate_deg",
            "orientation_confirm_sec",
            "lying_limit_sec",
            "landscape_overhead_enter_deg",
            "landscape_overhead_exit_deg",
            "landscape_lying_limit_sec",
            "landscape_still_motion_max",
            "landscape_still_gyro_max",
            "max_window_dt_sec",
            "shake_motion_min",
            "shake_gyro_min",
            "shake_cooldown_sec",
            "alert_cooldown_sec",
            "nag_interval_sec",
            "edge_timeout_sec",
            "snooze_duration_sec",
        )
        for name in float_fields:
            val = getattr(self, name)
            if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, float(val))

        for deg_field in (
            "overhead_enter_deg",
            "overhead_exit_deg",
            "face_down_deg",
            "roll_enter_deg",
            "roll_exit_deg",
            "landscape_overhead_enter_deg",
            "landscape_overhead_exit_deg",
        ):
            if getattr(self, deg_field) > 180.0:
                raise ValueError(f"{deg_field} must be <= 180.0")

        if self.roll_enter_deg > 90.0 or self.roll_exit_deg > 90.0:
            raise ValueError("roll_enter_deg and roll_exit_deg must be <= 90.0 (in-plane lean range)")
        if self.flat_gate_deg >= 90.0:
            raise ValueError("flat_gate_deg must be < 90.0")

        if self.overhead_exit_deg >= self.overhead_enter_deg:
            raise ValueError("overhead_exit_deg must be strictly less than overhead_enter_deg")
        if self.landscape_overhead_exit_deg >= self.landscape_overhead_enter_deg:
            raise ValueError(
                "landscape_overhead_exit_deg must be strictly less than landscape_overhead_enter_deg"
            )
        if self.roll_exit_deg >= self.roll_enter_deg:
            raise ValueError("roll_exit_deg must be strictly less than roll_enter_deg")
        if max(self.overhead_enter_deg, self.landscape_overhead_enter_deg) >= self.face_down_deg:
            raise ValueError("face_down_deg must be greater than every overhead_enter_deg")

    # ---------- Logika orientasi ----------

    def thresholds_for(self, orientation: str) -> OrientationThresholds:
        if orientation == LANDSCAPE:
            return OrientationThresholds(
                enter_deg=self.landscape_overhead_enter_deg,
                exit_deg=self.landscape_overhead_exit_deg,
                lying_limit_sec=self.landscape_lying_limit_sec,
                still_motion_max=self.landscape_still_motion_max,
                still_gyro_max=self.landscape_still_gyro_max,
            )
        return OrientationThresholds(
            enter_deg=self.overhead_enter_deg,
            exit_deg=self.overhead_exit_deg,
            lying_limit_sec=self.lying_limit_sec,
            still_motion_max=self.still_motion_max,
            still_gyro_max=self.still_gyro_max,
        )

    def classify_orientation(self, gx: float, gy: float, gz: float, prev: str = PORTRAIT) -> str:
        """Raw orientation dengan hysteresis. Debounce (orientation_confirm_sec) dilakukan di detector."""
        prev = prev if prev in (PORTRAIT, LANDSCAPE) else PORTRAIT
        if not self.use_roll:
            return PORTRAIT
        tilt = screen_tilt_deg(gx, gy, gz)
        if tilt is None or tilt < self.flat_gate_deg or tilt > 180.0 - self.flat_gate_deg:
            return prev  # hampir datar: roll tidak valid, tahan orientasi terakhir
        lean = in_plane_lean_deg(gx, gy)
        if prev == LANDSCAPE:
            return PORTRAIT if lean < self.roll_exit_deg else LANDSCAPE
        return LANDSCAPE if lean >= self.roll_enter_deg else PORTRAIT

    def is_overhead(self, tilt: float, orientation: str, was_overhead: bool) -> bool:
        t = self.thresholds_for(orientation)
        if tilt >= self.face_down_deg:
            return False
        if was_overhead:
            return tilt > t.exit_deg
        return tilt >= t.enter_deg

    def is_still(self, motion: float, gyro: float, orientation: str) -> bool:
        t = self.thresholds_for(orientation)
        return motion <= t.still_motion_max and gyro <= t.still_gyro_max

    # ---------- Serialisasi ----------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorConfig":
        if not isinstance(data, dict):
            raise ValueError("Config data must be a dictionary")
        return cls().merged_with(data)

    def merged_with(self, partial_dict: dict[str, Any]) -> "DetectorConfig":
        if not isinstance(partial_dict, dict):
            raise ValueError("partial_dict must be a dictionary")
        valid_keys = {f.name for f in fields(self)}
        unknown = set(partial_dict.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")

        merged = self.to_dict()
        for key, value in partial_dict.items():
            if key in ("shake_confirm_windows", "stale_message_ms"):
                if type(value) is float and value.is_integer():
                    value = int(value)
            merged[key] = value
        return DetectorConfig(**merged)