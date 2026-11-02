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

    def predict(
        self,
        overhead_deg: float,
        roll_deg: float,
        motion_rms: float,
        gyro_rms: float,
    ) -> tuple[str, float] | None:
        self._maybe_reload()
        if self._model is None:
            return None

        lean_deg = safe_lean_deg(overhead_deg, roll_deg)
        features = np.array(
            [[float(overhead_deg), float(roll_deg), float(lean_deg), float(motion_rms), float(gyro_rms)]],
            dtype=float,
        )
        try:
            probs = self._model.predict_proba(features)[0]
            classes = list(self._model.classes_)
            best_idx = int(np.argmax(probs))
            predicted_label = str(classes[best_idx])
            max_proba = float(probs[best_idx])
        except Exception:
            return None

        if predicted_label not in VALID_POSTURE_LABELS:
            return None
        if not math.isfinite(max_proba):
            return None
        max_proba = max(0.0, min(1.0, max_proba))
        return predicted_label, max_proba

