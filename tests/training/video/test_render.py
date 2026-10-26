import numpy as np

from birdcount.training.video.render import Renderer
from birdcount.training.video.timeline import Sample


SIZE = (96, 64)
SAMPLE = Sample(0, 0, 65.0, 151.4, 7.25, (8, 8, 40, 40))


def frame_and_density():
    frame = np.full((64, 96, 3), 90, np.uint8)
    density = np.zeros((8, 12), np.float16)
    density[2:4, 2:4] = 0.2
    return frame, density


def test_caption_carries_the_threshold_next_to_its_number() -> None:
    r = Renderer(SIZE, thresholds={"count": 150, "peak": 0})
    assert r.caption(SAMPLE, ("count",)) == ("0:01:05  count 151 / thr 150 OVER", True)
    assert r.caption(SAMPLE, ("peak", "count"))[0] == "0:01:05  peak 7.2  count 151 / thr 150 OVER"
    assert Renderer(SIZE).caption(SAMPLE, ("count",)) == ("0:01:05  count 151", False)


def test_drawing_never_touches_the_decoded_frame() -> None:
    frame, density = frame_and_density()
    before = frame.copy()
    keep = np.ones((64, 96), np.uint8)
    keep[:, 48:] = 0
    r = Renderer(SIZE, keep=keep)
    for style in ("overlay", "boxes"):
        out = r.frame_image(style, frame, density, SAMPLE)
        assert out.shape == frame.shape and not np.array_equal(out, frame)
    for mode in ("overlay", "split", "density"):
        assert r.video_frame(mode, frame, density, SAMPLE).shape[::-1][1:] == r.video_size(mode)
    np.testing.assert_array_equal(frame, before)


def test_output_width_scales_everything_but_peak_pictures() -> None:
    frame, density = frame_and_density()
    r = Renderer(SIZE, width=48)
    assert r.frame_image("plain", frame, density, SAMPLE).shape == (32, 48, 3)
    assert r.video_size("split") == (48, 16)
    assert r.video_frame("split", frame, density, SAMPLE).shape == (16, 48, 3)
    assert r.peak_image(frame, density, SAMPLE).shape == (64, 96, 3)
    np.testing.assert_array_equal(Renderer(SIZE).frame_image("plain", frame, density, SAMPLE), frame)
