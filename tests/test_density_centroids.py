"""Density centroid: the inferencer's definition, and the alarm acting on it.

`density_centroids` must agree with the delivery package's
`model_runtime/inference.py:density_centroid`, because the motion filter's
thresholds were tuned on that definition. The engine test then checks the
point of having it: a flock sweeping across the frame above threshold is not
a pile-up, while the same count standing still is.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from runtime.handlers._base import _NullAudit
from runtime.inferencer import BatchInferenceResult, InferenceResult
from runtime.inferencer._utils import density_centroids
from runtime.result_comsumer._alarm_engine import _AlarmEngine


def _delivery_centroid(density_map: np.ndarray):
    """Verbatim from the delivery package's model_runtime/inference.py."""
    mass = float(density_map.sum())
    if mass <= 1e-6:
        return None, None
    map_h, map_w = density_map.shape
    ys, xs = np.indices((map_h, map_w), dtype=np.float32)
    cx = float((density_map * xs).sum() / mass)
    cy = float((density_map * ys).sum() / mass)
    return (cx + 0.5) / max(1, map_w), (cy + 0.5) / max(1, map_h)


class DensityCentroidTest(unittest.TestCase):
    def test_single_cell_and_uniform_map(self):
        d = torch.zeros(2, 1, 4, 8)
        d[0, 0, 2, 6] = 3.0
        d[1] = 1.0
        out = density_centroids(d)
        np.testing.assert_allclose(out[0].numpy(), [6.5 / 8, 2.5 / 4], rtol=1e-6)
        np.testing.assert_allclose(out[1].numpy(), [0.5, 0.5], rtol=1e-6)

    def test_matches_delivery_definition(self):
        rng = np.random.default_rng(0)
        maps = rng.random((3, 90, 160)).astype(np.float32) * 0.05
        out = density_centroids(torch.from_numpy(maps)).numpy()
        for got, m in zip(out, maps):
            np.testing.assert_allclose(got, _delivery_centroid(m), rtol=1e-5)

    def test_empty_map_has_no_centroid(self):
        out = density_centroids(torch.zeros(1, 1, 4, 4))
        self.assertTrue(out.isnan().all())


_ALARM = {
    "output_dir": "runs",
    "rules": {"alert_trigger_seconds": 2.0, "recovery_seconds": 2.0, "max_level": 3},
    "motion_filter": {"enabled": True},
    "cameras": [{"camera_id": "axis1/B8A44FD52B7F", "threshold": 140}],
}


class MotionFilterWithCentroidTest(unittest.TestCase):
    def _run(self, xs):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alarm.json"
            path.write_text(json.dumps(_ALARM), encoding="utf-8")
            config = SimpleNamespace(
                num_streams=1,
                sid_to_mac={0: "B8A44FD52B7F"},
                sid_to_ip={0: "clip"},
                envs=SimpleNamespace(alarm_config_path=str(path)),
            )
            engine = _AlarmEngine(config)
        actions, reasons = [], set()
        for i, x in enumerate(xs):  # 5 FPS, count well over threshold throughout
            result = InferenceResult(0, 0, i, 1000.0 + i * 0.2, 200.0, 0.0, centroid_x=x, centroid_y=0.5)
            engine.evaluate_inplace(BatchInferenceResult([result]), set(), _NullAudit())
            actions += [a.action_type for a in result.alarm_actions]
            reasons.add(result.alarm_sample.motion_reason)
        return actions, reasons

    def test_sweeping_flock_does_not_alarm(self):
        actions, reasons = self._run([0.2 + 0.01 * i for i in range(40)])  # 0.05 of the frame per second
        self.assertEqual(actions, [])
        self.assertIn("directional_motion", reasons)

    def test_static_pile_still_alarms(self):
        actions, _ = self._run([0.5] * 40)
        self.assertEqual(actions, ["level1"])


if __name__ == "__main__":
    unittest.main()
