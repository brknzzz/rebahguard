"""Wrapper inferensi Random Forest RebahGuard dengan Throttled Auto-Reload berbasis time.monotonic()."""

import math
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from shared.messages import VALID_POSTURE_LABELS
from shared.processing import safe_lean_deg


class PostureMLClassifier:
    """Memuat model Random Forest (.joblib) dengan pengecekan mtime paling sering tiap reload_interval_sec."""

    def __init__(
        self,
        model_path: Path | str = "models/posture_rf.joblib",
        reload_interval_sec: float = 5.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.reload_interval_sec = max(0.0, float(reload_interval_sec))
        self._last_check_mono: float | None = None
        self._loaded_mtime: float | None = None
        self._model: Any | None = None

    def _maybe_reload(self) -> None:
        now_mono = time.monotonic()
        if (
            self._last_check_mono is not None
            and (now_mono - self._last_check_mono) < self.reload_interval_sec
        ):
            return

        self._last_check_mono = now_mono
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            self._model = None
            self._loaded_mtime = None
            return

        if self._loaded_mtime is not None and mtime == self._loaded_mtime and self._model is not None:
            return

        try:
            loaded = joblib.load(self.model_path)
            model = loaded.get("model") if isinstance(loaded, dict) else loaded
            if model is None or not hasattr(model, "predict_proba"):
                self._model = None
                self._loaded_mtime = None
                return
            self._model = model
            self._loaded_mtime = mtime
        except Exception:
            self._model = None
            self._loaded_mtime = None

    def predict_with_proba(
        self,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
    ) -> tuple[str, float, dict[str, float]] | None:
        """Inferensi tunggal: memanggil clf.predict_proba() tepat 1 kali per window."""
        self._maybe_reload()
        if self._model is None:
            return None

        lean_deg = safe_lean_deg(overhead_deg, roll_deg)
        features = np.array(
            [[float(overhead_deg), float(roll_deg), float(lean_deg), float(motion_rms), float(gyro_rms)]],
            dtype=float,
        )
        try:
            row_proba = self._model.predict_proba(features)[0]
            classes = [str(c) for c in self._model.classes_]
        except Exception:
            return None

        raw_map: dict[str, float] = {}
        for label, prob in zip(classes, row_proba):
            fprob = float(prob)
            if not math.isfinite(fprob):
                return None
            raw_map[str(label)] = round(max(0.0, min(1.0, fprob)), 4)

        if not set(raw_map.keys()).issubset(set(VALID_POSTURE_LABELS)):
            return None

        probs_dict: dict[str, float] = {
            label: raw_map.get(label, 0.0) for label in VALID_POSTURE_LABELS
        }
        best_label = max(probs_dict, key=probs_dict.get)
        best_conf = probs_dict[best_label]
        if best_label not in VALID_POSTURE_LABELS:
            return None
        return best_label, best_conf, probs_dict

    def predict(
        self,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
    ) -> tuple[str, float] | None:
        res = self.predict_with_proba(overhead_deg, roll_deg, motion_rms, gyro_rms)
        if res is None:
            return None
        best_label, best_conf, _ = res
        return best_label, best_conf

    def predict_proba(
        self,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
    ) -> dict[str, float] | None:
        res = self.predict_with_proba(overhead_deg, roll_deg, motion_rms, gyro_rms)
        if res is None:
            return None
        _, _, probs_dict = res
        return probs_dict
