import numpy as np
import pytest

from birdcount.training.eval.regions import assign_points, blob_regions, grid_regions


FRAME = (96, 64)  # density 8 rows x 12 cols


def _density() -> np.ndarray:
    d = np.zeros((8, 12), np.float32)
    d[1:3, 1:3] = 1.0  # blob A: 4 birds
    d[5, 8] = 0.5  # blob B: 0.5 birds
    d[5, 10] = 0.5  # blob C, 1 empty cell from B
    return d


def test_blobs_ordered_by_count_with_frame_geometry() -> None:
    labels, regions = blob_regions(_density(), FRAME, min_density=0.02)
    assert [r.count for r in regions] == [4.0, 0.5, 0.5]
    a = regions[0]
    assert a.centroid == pytest.approx((16.0, 16.0))  # cells 1..2 -> pixels 8..24
    assert a.bbox == pytest.approx((8.0, 8.0, 16.0, 16.0))
    assert (labels == 1).sum() == 4


def test_merge_bridges_gaps_and_min_count_drops() -> None:
    _, merged = blob_regions(_density(), FRAME, min_density=0.02, merge=1)
    assert sorted(r.count for r in merged) == [1.0, 4.0]
    _, kept = blob_regions(_density(), FRAME, min_density=0.02, min_count=1.0)
    assert [r.count for r in kept] == [4.0]


def test_points_scored_per_region_with_snap() -> None:
    labels, regions = blob_regions(_density(), FRAME, min_density=0.02)
    points = np.array([[10.0, 10.0], [20.0, 20.0], [20.0, 20.0], [30.0, 10.0], [70.0, 60.0]])
    outside = assign_points(labels, regions, points, 8)
    assert regions[0].gt == 3 and regions[0].error == pytest.approx(1.0)
    assert outside == 2
    labels, regions = blob_regions(_density(), FRAME, min_density=0.02)
    assert assign_points(labels, regions, points, 8, snap=1) == 1  # (30,10) is 1 cell from A


def test_grid_tiles_cover_the_frame_and_sum_to_total() -> None:
    d = _density()
    labels, tiles = grid_regions(d, FRAME, 2, 3)
    assert len(tiles) == 6 and labels.min() == 1
    assert sum(t.count for t in tiles) == pytest.approx(d.sum())
    assert (tiles[0].row, tiles[0].col, tiles[5].row, tiles[5].col) == (1, 1, 2, 3)
    assert tiles[0].count == 4.0


def test_grid_too_fine_is_rejected() -> None:
    with pytest.raises(ValueError):
        grid_regions(_density(), FRAME, 9, 3)


def test_region_dict_has_errors() -> None:
    labels, regions = blob_regions(_density(), FRAME, min_density=0.02)
    assign_points(labels, regions, np.empty((0, 2)), 8)
    d = regions[0].to_dict()
    assert d["gt"] == 0 and d["error"] == 4.0 and d["relative_error"] is None and "row" not in d
