"""Region-mask loading and application (`runtime.inferencer._masks`).

Runs on CPU with synthetic masks. The assertions that matter are the numeric
ones: that a masked stream's count drops by exactly the masked-out density, and
that a density cell straddling the boundary is weighted by its overlap rather
than rounded in or out — that fractional weighting is the whole reason the
downsample is INTER_AREA.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from runtime.inferencer._masks import load_stream_masks


MAC_A = "B8A44FD51C3C"
MAC_B = "B8A44F5B4394"

H, W = 64, 96
CPU = torch.device("cpu")


def _write_mask(path: Path, keep_left_fraction: float) -> None:
    """White (count) on the left, black (ignore) on the right."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, : int(W * keep_left_fraction)] = 255
    cv2.imwrite(str(path), img)


class LoadStreamMasksTest(unittest.TestCase):
    def test_disabled_when_unset_or_missing(self):
        self.assertIsNone(load_stream_masks("", {0: MAC_A}, 1, (H, W), CPU))
        self.assertIsNone(load_stream_masks("does/not/exist", {0: MAC_A}, 1, (H, W), CPU))

    def test_bare_and_decorated_filenames_both_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_mask(root / f"{MAC_A}.png", 0.5)
            _write_mask(root / f"mask_axis2_{MAC_B}.png", 0.25)

            masks = load_stream_masks(root, {0: MAC_A, 1: MAC_B}, 2, (H, W), CPU)
            self.assertIsNotNone(masks)
            self.assertEqual(masks.covered, {0: MAC_A, 1: MAC_B})
            self.assertEqual(masks.uncovered, [])
            self.assertAlmostEqual(float(masks.frame_keep[0].mean()), 0.5, places=2)
            self.assertAlmostEqual(float(masks.frame_keep[1].mean()), 0.25, places=2)

    def test_unmatched_stream_gets_an_all_ones_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_mask(root / f"{MAC_A}.png", 0.5)

            masks = load_stream_masks(root, {0: MAC_A, 1: None}, 2, (H, W), CPU)
            self.assertEqual(masks.covered, {0: MAC_A})
            self.assertEqual(masks.uncovered, [1])
            # An unmasked stream must be a genuine no-op, not a partial mask.
            self.assertEqual(float(masks.frame_keep[1].min()), 1.0)

    def test_ambiguous_mask_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_mask(root / f"old_{MAC_A}.png", 0.5)
            _write_mask(root / f"new_{MAC_A}.png", 0.9)
            self.assertIsNone(load_stream_masks(root, {0: MAC_A}, 1, (H, W), CPU))

    def test_fully_black_mask_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_mask(root / f"{MAC_A}.png", 0.0)
            self.assertIsNone(load_stream_masks(root, {0: MAC_A}, 1, (H, W), CPU))

    def test_oversized_mask_is_resized_to_frame_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            img = np.zeros((H * 3, W * 3, 3), dtype=np.uint8)
            img[:, : (W * 3) // 2] = 255
            cv2.imwrite(str(root / f"{MAC_A}.png"), img)

            masks = load_stream_masks(root, {0: MAC_A}, 1, (H, W), CPU)
            self.assertEqual(tuple(masks.frame_keep.shape), (1, 1, H, W))
            self.assertAlmostEqual(float(masks.frame_keep[0].mean()), 0.5, places=2)


class MaskApplicationTest(unittest.TestCase):
    def _masks(self, root: Path, keep: float):
        _write_mask(root / f"{MAC_A}.png", keep)
        return load_stream_masks(root, {0: MAC_A}, 1, (H, W), CPU)

    def test_frame_mask_blacks_out_the_ignored_half(self):
        with tempfile.TemporaryDirectory() as d:
            masks = self._masks(Path(d), 0.5)
            frame = torch.full((1, 3, H, W), 200.0)
            frame.mul_(masks.frame_for([0]))
            self.assertEqual(float(frame[:, :, :, : W // 2].min()), 200.0)
            self.assertEqual(float(frame[:, :, :, W // 2 :].max()), 0.0)

    def test_count_drops_by_exactly_the_masked_out_density(self):
        with tempfile.TemporaryDirectory() as d:
            masks = self._masks(Path(d), 0.5)
            h, w = H // 8, W // 8
            density = torch.ones((1, 1, h, w), dtype=torch.float16)
            self.assertEqual(float(density.sum()), h * w)

            density.mul_(masks.density_for([0], (h, w)))
            self.assertAlmostEqual(float(density.sum()), h * w / 2, delta=0.5)

    def test_boundary_cells_are_area_weighted_not_rounded(self):
        with tempfile.TemporaryDirectory() as d:
            # Boundary at 3/8 of the width puts it mid-cell after an 8x
            # downsample, so a hard NEAREST mask would round the straddling
            # column to 0 or 1 and lose a strip of real birds.
            masks = self._masks(Path(d), 0.4375)
            h, w = H // 8, W // 8
            keep = masks.density_for([0], (h, w))[0, 0].float()
            fractional = keep[(keep > 0.01) & (keep < 0.99)]
            self.assertTrue(fractional.numel() > 0, "no partially-weighted cells; INTER_AREA not in effect")
            self.assertAlmostEqual(float(keep.mean()), 0.4375, places=2)

    def test_host_masks_are_released_after_the_density_build(self):
        with tempfile.TemporaryDirectory() as d:
            masks = self._masks(Path(d), 0.5)
            masks.density_for([0], (H // 8, W // 8))
            self.assertIsNone(masks._keep_u8, "host keep-masks should be freed once weights are built")
            # Cached: a second call at the same resolution must still work.
            self.assertIsNotNone(masks.density_for([0], (H // 8, W // 8)))


if __name__ == "__main__":
    unittest.main()
