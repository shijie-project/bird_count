import unittest

from alarm.models import CountSample, MotionFilterConfig
from alarm.motion_filter import DirectionalMotionFilter


class DirectionalMotionFilterTest(unittest.TestCase):
    def test_excludes_fast_consistent_centroid_movement(self):
        filt = DirectionalMotionFilter(
            MotionFilterConfig(
                window_seconds=3,
                min_samples=6,
                min_speed_norm_per_sec=0.03,
                min_net_displacement_norm=0.08,
                min_direction_consistency=0.85,
            )
        )

        decision = None
        for i in range(16):
            decision = filt.evaluate(
                CountSample(
                    timestamp=i * 0.2,
                    camera_id="cam1",
                    count=180,
                    centroid_x=0.2 + i * 0.01,
                    centroid_y=0.5,
                ),
                threshold=150,
            )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.excluded)
        self.assertEqual(decision.reason, "directional_motion")
        self.assertGreater(decision.speed_norm_per_sec or 0.0, 0.03)
        self.assertGreater(decision.net_displacement_norm or 0.0, 0.08)
        self.assertAlmostEqual(decision.direction_degrees or 0.0, 0.0, delta=5.0)
        self.assertAlmostEqual(decision.velocity_y_norm_per_sec or 0.0, 0.0, delta=1e-6)

    def test_keeps_jittery_centroid(self):
        filt = DirectionalMotionFilter(
            MotionFilterConfig(
                window_seconds=3,
                min_samples=6,
                min_speed_norm_per_sec=0.03,
                min_net_displacement_norm=0.08,
                min_direction_consistency=0.85,
            )
        )

        decision = None
        for i in range(16):
            decision = filt.evaluate(
                CountSample(
                    timestamp=i * 0.2,
                    camera_id="cam1",
                    count=180,
                    centroid_x=0.5 + (0.015 if i % 2 else -0.015),
                    centroid_y=0.5,
                ),
                threshold=150,
            )

        self.assertIsNotNone(decision)
        self.assertFalse(decision.excluded)

    def test_keeps_stationary_dense_centroid(self):
        filt = DirectionalMotionFilter(
            MotionFilterConfig(
                window_seconds=3,
                min_samples=6,
                min_speed_norm_per_sec=0.03,
                min_net_displacement_norm=0.08,
                min_direction_consistency=0.85,
            )
        )

        decision = None
        for i in range(16):
            decision = filt.evaluate(
                CountSample(
                    timestamp=i * 0.2,
                    camera_id="cam1",
                    count=180,
                    centroid_x=0.5,
                    centroid_y=0.5,
                ),
                threshold=150,
            )

        self.assertIsNotNone(decision)
        self.assertFalse(decision.excluded)
        self.assertEqual(decision.reason, "not_directional_motion")


if __name__ == "__main__":
    unittest.main()
