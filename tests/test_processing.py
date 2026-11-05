import unittest

from shared.processing import (
    Features,
    SensorSample,
    extract_features,
    lean_from_features,
    safe_lean_deg,
)


class ProcessingTests(unittest.TestCase):
    def test_stationary_phone_with_calibration_bias_zero_motion_rms(self):
        # Akselerometer dengan bias kalibrasi statis (az = -9.55 konstan, bukan -9.81)
        samples = [SensorSample(i * 0.2, (0.0, 0.0, -9.55), (0.0, 0.0, 0.0)) for i in range(6)]
        result = extract_features(samples)
        self.assertIsInstance(result, Features)
        self.assertEqual(result.motion_rms, 0.0)
        self.assertEqual(result.gyro_rms, 0.0)
        self.assertAlmostEqual(result.overhead_deg, 180.0, places=2)
        self.assertAlmostEqual(result.roll_deg, 0.0, places=2)
        self.assertEqual(result.raw_accel_mean, (0.0, 0.0, -9.55))
        self.assertEqual(result.raw_gyro_last, (0.0, 0.0, 0.0))

    def test_landscape_sitting_angles(self):
        # Posisi duduk main game landscape (ax=9.81, ay=0.0, az=2.0)
        samples = [SensorSample(i * 0.2, (9.81, 0.0, 2.0), (0.02, 0.01, 0.0)) for i in range(5)]
        result = extract_features(samples)
        # az > 0 => overhead_deg < 90 derajat (sekitar 78.48 derajat), roll_deg sekitar 78.48 derajat
        self.assertLess(result.overhead_deg, 90.0)
        self.assertGreater(result.roll_deg, 70.0)

    def test_lean_from_features_and_safe_lean_deg(self):
        # Saat HP datar (overhead 0 deg atau 180 deg, roll 0 deg) -> lean_from_features None, safe_lean_deg 0.0
        self.assertIsNone(lean_from_features(0.0, 0.0))
        self.assertIsNone(lean_from_features(180.0, 0.0))
        self.assertEqual(safe_lean_deg(0.0, 0.0), 0.0)
        self.assertEqual(safe_lean_deg(180.0, 0.0), 0.0)

        # Posisi portrait tegak lurus (overhead=60, roll=5) -> lean kecil (< 15 deg)
        portrait_lean = safe_lean_deg(60.0, 5.0)
        self.assertGreater(portrait_lean, 0.0)
        self.assertLess(portrait_lean, 15.0)

        # Posisi landscape (overhead=78.48, roll=78.48) -> lean mendekati 90 deg (>= 60 deg)
        landscape_lean = safe_lean_deg(78.48, 78.48)
        self.assertGreaterEqual(landscape_lean, 60.0)

    def test_dynamic_movement_and_gyro_rms(self):
        samples = [
            SensorSample(0.0, (0.0, 0.0, -9.0), (0.3, 0.4, 0.0)),
            SensorSample(0.2, (0.0, 0.0, -11.0), (0.3, 0.4, 0.0)),
            SensorSample(0.4, (0.0, 0.0, -9.0), (0.3, 0.4, 0.0)),
            SensorSample(0.6, (0.0, 0.0, -11.0), (0.3, 0.4, 0.0)),
        ]
        result = extract_features(samples)
        # Magnitudo bolak-balik antara 9.0 dan 11.0 -> mean = 10.0, deviasi = 1.0 -> motion_rms = 1.0
        self.assertAlmostEqual(result.motion_rms, 1.0, places=4)
        # Gyro magnitude tiap sampel = sqrt(0.09 + 0.16) = 0.5
        self.assertAlmostEqual(result.gyro_rms, 0.5, places=4)

    def test_need_at_least_two_samples(self):
        with self.assertRaises(ValueError):
            extract_features([SensorSample(0.0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))])


if __name__ == "__main__":
    unittest.main()
