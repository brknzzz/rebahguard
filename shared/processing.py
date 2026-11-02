"""DATA PROCESSING (used by BOTH edge devices and ML pipeline).

Processing A: Gravity Orientation Smoothing (overhead_deg & roll_deg).
Processing B: Mean-Centered Dynamic Motion RMS (calibration-bias-free motion_rms).
Processing C: Gyroscope Micro-Activity RMS (gyro_rms).
"""

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, radians, sin, sqrt


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lean_from_features(overhead_deg: float, roll_deg: float) -> float | None:
    """Rekonstruksi in-plane lean (0=portrait, 90=landscape) dari telemetry.

    roll_deg = atan2(|ax|, sqrt(ay^2+az^2)), overhead_deg = acos(az/g).
    |ax| = g*sin(r), |az| = g*|cos(t)|  ->  |ay| = g*sqrt(cos^2 r - cos^2 t).
    Return None kalau HP hampir datar (lean tidak valid).
    """
    r = radians(roll_deg)
    t = radians(overhead_deg)
    ay_sq = cos(r) ** 2 - cos(t) ** 2
    ax = sin(r)
    ay = sqrt(max(0.0, ay_sq))
    if ax < 1e-6 and ay < 1e-6:
        return None
    return degrees(atan2(ax, ay))


def safe_lean_deg(overhead_deg: float, roll_deg: float) -> float:
    """Shared helper for dataset recorder, training, and ML inference.

    Returns 0.0 when lean_from_features(...) is None (phone flat), rounded to 4 decimals.
    """
    lean = lean_from_features(overhead_deg, roll_deg)
    if lean is None:
        return 0.0
    return round(lean, 4)


@dataclass(frozen=True)
class SensorSample:
    seconds: float
    acceleration: tuple[float, float, float]
    gyroscope: tuple[float, float, float]


@dataclass(frozen=True)
class Features:
    overhead_deg: float
    roll_deg: float
    motion_rms: float
    gyro_rms: float
    raw_accel_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_gyro_last: tuple[float, float, float] = (0.0, 0.0, 0.0)


def extract_features(samples: list[SensorSample]) -> Features:
    if len(samples) < 2:
        raise ValueError("At least 2 sensor samples are required")

    n = len(samples)

    # Processing A: Gravity Orientation Smoothing
    ax_mean = sum(s.acceleration[0] for s in samples) / n
    ay_mean = sum(s.acceleration[1] for s in samples) / n
    az_mean = sum(s.acceleration[2] for s in samples) / n

    g_norm = max(1e-6, sqrt(ax_mean * ax_mean + ay_mean * ay_mean + az_mean * az_mean))
    overhead_deg = degrees(acos(_clamp(az_mean / g_norm, -1.0, 1.0)))
    roll_deg = degrees(atan2(abs(ax_mean), sqrt(ay_mean * ay_mean + az_mean * az_mean)))

    # Processing B: Mean-Centered Dynamic Motion RMS (bebas bias kalibrasi statis 9.81)
    magnitudes = [
        sqrt(s.acceleration[0] ** 2 + s.acceleration[1] ** 2 + s.acceleration[2] ** 2)
        for s in samples
    ]
    mean_mag = sum(magnitudes) / n
    motion_rms = sqrt(sum((m - mean_mag) ** 2 for m in magnitudes) / n)

    # Processing C: Gyroscope Micro-Activity RMS
    gyro_sq_sum = sum(
        s.gyroscope[0] ** 2 + s.gyroscope[1] ** 2 + s.gyroscope[2] ** 2
        for s in samples
    )
    gyro_rms = sqrt(gyro_sq_sum / n)

    last_gx, last_gy, last_gz = samples[-1].gyroscope

    return Features(
        overhead_deg=round(overhead_deg, 4),
        roll_deg=round(roll_deg, 4),
        motion_rms=round(motion_rms, 4),
        gyro_rms=round(gyro_rms, 4),
        raw_accel_mean=(round(ax_mean, 4), round(ay_mean, 4), round(az_mean, 4)),
        raw_gyro_last=(round(last_gx, 4), round(last_gy, 4), round(last_gz, 4)),
    )
