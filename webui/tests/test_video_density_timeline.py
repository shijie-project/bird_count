import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "ops"))

import video_density_timeline as timeline

from webui.runs import Run
from webui.schema import get_schema


class VideoDensityTimelineSchemaTests(unittest.TestCase):
    def test_schema_exposes_the_video_controls_on_its_own_page(self):
        schema = get_schema("video_density")
        self.assertEqual(schema["page"], "video")
        self.assertEqual([group["title"] for group in schema["groups"]], ["input", "model", "density", "output"])
        options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
        self.assertTrue(options["video"]["required"])
        self.assertEqual(options["sample_seconds"]["default"], 1.0)
        self.assertEqual(options["window"]["default"], 16)
        self.assertEqual(options["threshold_metric"]["choices"], ["peak", "count"])
        self.assertEqual(options["report"]["default"], "both")
        self.assertEqual(options["report"]["choices"], ["both", "count", "peak"])
        self.assertEqual(options["save_frames"]["default"], "none")
        self.assertEqual(options["save_frames"]["choices"], ["none", "overlay", "plain"])
        self.assertEqual(options["frame_width"]["default"], 0)
        self.assertEqual(options["caption_corner"]["default"], "top-left")
        self.assertEqual(options["caption_corner"]["choices"], ["top-left", "top-right"])
        self.assertEqual(options["max_samples"]["default"], 5000)
        self.assertIsNone(options["output_dir"]["default"])


class VideoDensityTimelineUnitTests(unittest.TestCase):
    def test_start_time_accepts_a_bare_clock_and_a_full_stamp(self):
        self.assertIsNone(timeline.parse_start_time(""))
        self.assertEqual(timeline.parse_start_time("16:20").strftime("%H:%M:%S"), "16:20:00")
        self.assertEqual(
            timeline.parse_start_time("2026-08-28 16:20:30"),
            datetime(2026, 8, 28, 16, 20, 30),
        )
        with self.assertRaises(SystemExit):
            timeline.parse_start_time("half past four")

    def test_clock_falls_back_to_elapsed_time_without_a_start(self):
        start = timeline.parse_start_time("23:59")
        self.assertEqual(timeline.clock_of(start, 90), "00:00:30")  # rolls past midnight
        self.assertEqual(timeline.clock_of(None, 3725), "1:02:05")

    def test_frame_size_parsing(self):
        self.assertIsNone(timeline.parse_frame_size(""))
        self.assertEqual(timeline.parse_frame_size("1080x720"), (1080, 720))
        with self.assertRaises(SystemExit):
            timeline.parse_frame_size("1080")
        with self.assertRaises(SystemExit):
            timeline.parse_frame_size("8x8")

    def test_window_peak_is_the_birds_inside_the_busiest_patch(self):
        maps = torch.zeros(1, 1, 10, 10)
        maps[0, 0, 4:6, 4:6] = 0.5  # four cells holding 0.5 birds each
        values, indices = timeline.window_peaks(maps, window=4)
        self.assertAlmostEqual(float(values[0]), 2.0, places=5)
        # The reported box covers the mass, in pixels of the analysed frame.
        x0, y0, x1, y1 = timeline.peak_box(int(indices[0]), (10, 10), 4, 80, 80)
        self.assertLessEqual(x0, 32)
        self.assertGreaterEqual(x1, 48)
        self.assertLessEqual(y0, 32)
        self.assertGreaterEqual(y1, 48)

    def test_window_larger_than_the_map_still_reports_the_total(self):
        maps = torch.full((1, 1, 3, 3), 0.25)
        values, _ = timeline.window_peaks(maps, window=99)
        self.assertAlmostEqual(float(values[0]), 2.25, places=5)

    def test_rolling_mean_keeps_the_series_length_and_smooths(self):
        smoothed = timeline.rolling_mean([0.0, 10.0, 0.0, 0.0, 0.0], 3)
        self.assertEqual(len(smoothed), 5)
        self.assertLess(smoothed[1], 10.0)
        self.assertAlmostEqual(float(np.mean(smoothed)), 2.0, places=1)
        self.assertEqual(timeline.rolling_mean([1.0, 2.0], 1), [1.0, 2.0])

    def test_report_selects_which_series_the_figure_shows(self):
        meta = {"window_px": 128}
        self.assertEqual([panel[1] for panel in timeline.series_panels({**meta, "report": "both"})], ["peak", "count"])
        self.assertEqual([panel[1] for panel in timeline.series_panels({**meta, "report": "count"})], ["count"])
        self.assertEqual([panel[1] for panel in timeline.series_panels({**meta, "report": "peak"})], ["peak"])
        # An older run's JSON has no "report" key; it reported both.
        self.assertEqual(len(timeline.series_panels(meta)), 2)

    def test_a_single_series_figure_renders(self):
        """The one-panel path returns a bare Axes, not an array — regression guard."""
        samples = [
            {"i": i, "t": float(i), "count": 250.0 + i, "peak": 9.0 + 0.1 * i, "frame": i * 30} for i in range(12)
        ]
        meta = {
            "name": "demo.mkv",
            "window_px": 128,
            "smooth": 5,
            "threshold": 260.0,
            "threshold_metric": "count",
            "report": "count",
            "interval": 1.0,
        }
        summary = timeline.summarize(samples, {**meta, "duration": 12.0}, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.png"
            self.assertTrue(timeline.render_chart(path, samples, meta, summary, None))
            self.assertGreater(path.stat().st_size, 10_000)

    def test_frame_names_sort_by_time_and_carry_the_second(self):
        self.assertEqual(timeline.frame_filename(0, 0.0), "frame_00000_000000s.jpg")
        self.assertEqual(timeline.frame_filename(13, 6.5), "frame_00013_000006s.jpg")
        self.assertEqual(timeline.frame_filename(7, 3661.2), "frame_00007_003661s.jpg")
        names = [timeline.frame_filename(i, i * 0.5) for i in range(12)]
        self.assertEqual(names, sorted(names))  # a file browser shows them in order

    def test_the_web_ui_builds_the_same_frame_name(self):
        """The chart opens a frame by rebuilding its path; the two must agree."""
        app_js = (ROOT / "webui" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("`${dir}/frame_${index}_${seconds}s.jpg`", app_js)
        self.assertIn("String(sample.i).padStart(5, '0')", app_js)
        self.assertIn("String(Math.floor(sample.t)).padStart(6, '0')", app_js)

    def test_saving_a_frame_can_downscale_without_touching_the_original(self):
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        density = np.zeros((60, 80), dtype=np.float32)
        density[30, 40] = 1.0
        overlay = timeline.blend_density(original, density, vmax=0.12)
        self.assertTrue(overlay.any())  # something was painted
        self.assertFalse(original.any())  # ... on a copy, not on the frame

        timeline.draw_caption(overlay, "04:00:07  count 275  peak 10.2")
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / "full.jpg"
            small = Path(directory) / "small.jpg"
            self.assertGreater(timeline.save_jpeg(full, overlay), 0)
            timeline.save_jpeg(small, overlay, width=320)
            self.assertEqual(cv2.imread(str(full)).shape[:2], (480, 640))
            self.assertEqual(cv2.imread(str(small)).shape[:2], (240, 320))

    def test_the_mask_gives_both_the_blanked_region_and_its_outline(self):
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            mask = np.zeros((200, 300, 3), dtype=np.uint8)
            mask[40:160, 60:240] = 255  # the area the camera actually watches
            cv2.imwrite(str(mask_path), mask)
            region, contours = timeline.load_static_mask(str(mask_path), 300, 200)

        self.assertTrue(region[10, 10])  # outside the kept area: blanked before inference
        self.assertFalse(region[100, 150])  # inside it: kept
        self.assertEqual(len(contours), 1)
        self.assertEqual(cv2.boundingRect(contours[0]), (60, 40, 180, 120))

    def test_a_mask_drawn_at_another_size_is_scaled_to_the_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            mask = np.zeros((200, 300, 3), dtype=np.uint8)
            mask[40:160, 60:240] = 255
            cv2.imwrite(str(mask_path), mask)
            region, contours = timeline.load_static_mask(str(mask_path), 600, 400)

        self.assertEqual(region.shape, (400, 600))
        x, y, w, h = cv2.boundingRect(contours[0])
        self.assertEqual((x, y, w, h), (120, 80, 360, 240))

    def test_the_mask_region_is_outlined_in_red_and_nothing_else_is_touched(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        contours = (np.array([[[60, 40]], [[239, 40]], [[239, 159]], [[60, 159]]], dtype=np.int32),)
        timeline.draw_mask_region(image, contours)
        self.assertEqual(tuple(image[40, 150]), timeline.MASK_REGION_COLOR)  # on the outline
        self.assertEqual(tuple(image[100, 150]), (0, 0, 0))  # inside, untouched
        timeline.draw_mask_region(image, ())  # no mask given: nothing to draw, no crash

    def test_the_caption_is_sized_for_the_image_that_is_saved(self):
        """Captioning before the downscale is what made the text unreadable."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            small = Path(directory) / "small.jpg"
            timeline.save_jpeg(small, frame, 480, "04:00:00  count 275  peak 10.2")
            saved = cv2.imread(str(small))

        self.assertEqual(saved.shape[1], 480)
        band = saved[: saved.shape[0] // 4]
        self.assertGreater(int((band > 200).sum()), 400)  # the caption is really there
        self.assertFalse(frame.any())  # ... and was not burned into the caller's frame

    def test_the_caption_can_sit_in_either_top_corner(self):
        def bright_side(corner):
            image = np.zeros((200, 400, 3), dtype=np.uint8)
            timeline.draw_caption(image, "04:00:00 count 275", corner)
            band = image[:60]
            left = int((band[:, :200] > 200).sum())
            right = int((band[:, 200:] > 200).sum())
            return left, right

        left, right = bright_side("top-left")
        self.assertGreater(left, right)
        left, right = bright_side("top-right")
        self.assertGreater(right, left)

    def test_peak_frames_follow_the_reported_series(self):
        samples = [
            {"t": 0.0, "peak": 9.0, "count": 100.0},
            {"t": 10.0, "peak": 1.0, "count": 300.0},
        ]
        self.assertEqual(timeline.pick_peak_samples(samples, 1, 5.0, "peak")[0]["t"], 0.0)
        self.assertEqual(timeline.pick_peak_samples(samples, 1, 5.0, "count")[0]["t"], 10.0)

    def test_peak_frames_are_spread_out_instead_of_one_event(self):
        samples = [{"t": float(i), "peak": peak} for i, peak in enumerate([1, 9, 8.5, 2, 7, 1, 1])]
        chosen = timeline.pick_peak_samples(samples, count=2, gap_seconds=3.0)
        self.assertEqual([sample["t"] for sample in chosen], [1.0, 4.0])


class VideoDensityTimelineParsingTests(unittest.TestCase):
    """The web UI's chart is built from the run's own log lines."""

    LOG = [
        "$ python -u webui/ops/video_density_timeline.py --video ../data/demo.mkv",
        "Video: ../data/demo.mkv",
        "  SAMPLE 0 | t 0.000 | count 288.96 | peak 9.164",
        "  SAMPLE 1 | t 0.500 | count 276.21 | peak 10.810",
        'TIMELINE META {"name": "demo.mkv", "interval": 0.5, "window": 16, "window_px": 128, '
        '"report": "peak", "threshold": 9.0, "threshold_metric": "peak", "smooth": 5, '
        '"start_time": "2000-01-01 16:20:00"}',
        'TIMELINE SUMMARY {"samples": 2, "peak_max": 10.81, "peak_max_clock": "16:20:00"}',
        'TIMELINE ARTIFACT {"kind": "chart", "path": "outputs/video_density/demo/timeline.png"}',
        'TIMELINE ARTIFACT {"kind": "frame", "path": "outputs/video_density/demo/peak_01.jpg", '
        '"t": 0.5, "clock": "16:20:00", "peak": 10.81, "count": 276.21}',
        "[webui] finished with exit code 0",
    ]

    def _parsed(self) -> dict:
        run = Run("video_density", [], {})
        for line in self.LOG:
            run._parse(line)
        return run.result["timeline"]

    def test_samples_meta_summary_and_artifacts_are_collected(self):
        parsed = self._parsed()
        self.assertEqual(
            parsed["samples"],
            [
                {"i": 0, "t": 0.0, "count": 288.96, "peak": 9.164},
                {"i": 1, "t": 0.5, "count": 276.21, "peak": 10.81},
            ],
        )
        self.assertEqual(parsed["meta"]["window_px"], 128)
        self.assertEqual(parsed["meta"]["threshold_metric"], "peak")
        self.assertEqual(parsed["meta"]["report"], "peak")
        self.assertEqual(parsed["summary"]["peak_max"], 10.81)
        self.assertEqual([artifact["kind"] for artifact in parsed["artifacts"]], ["chart", "frame"])

    def test_a_truncated_json_line_is_ignored_rather_than_failing_the_run(self):
        run = Run("video_density", [], {})
        run._parse('TIMELINE META {"name": "demo.mkv", "inter')
        self.assertEqual(run.result["timeline"]["meta"], {})

    def test_prose_lines_are_left_to_the_log(self):
        run = Run("video_density", [], {})
        run._parse("  Peak density     : 10.81 birds in one 128px patch at 16:20:00")
        self.assertEqual(run.result["timeline"]["samples"], [])
        self.assertEqual(run.result["timeline"]["summary"], {})

    def test_history_restores_the_chart_from_the_log_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_density-20260828-120000-1.log"
            path.write_text("\n".join(self.LOG), encoding="utf-8")
            run = Run.from_log(path)

        self.assertIsNotNone(run)
        self.assertEqual(run.kind, "video_density")
        self.assertEqual(run.status, "done")
        self.assertEqual(len(run.result["timeline"]["samples"]), 2)
        self.assertEqual(len(run.result["timeline"]["artifacts"]), 2)


if __name__ == "__main__":
    unittest.main()


class ThresholdFromAlarmConfigTests(unittest.TestCase):
    """The caption threshold is looked up by the MAC in the video filename."""

    CONFIG = {
        "cameras": [
            {"camera_id": "axis1/B8A44FD51C3C", "name": "c1", "threshold": 180},
            {"camera_id": "axis4/ACCC8E9B4972", "name": "c2", "threshold": 105},
        ]
    }

    def _config(self, tmp: Path) -> Path:
        path = tmp / "alarm.json"
        path.write_text(json.dumps(self.CONFIG), encoding="utf-8")
        return path

    def test_resolves_a_bare_mac_filename(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(Path(d))
            value, note = timeline.threshold_from_alarm_config("../data/videos/B8A44FD51C3C.mkv", cfg)
            self.assertEqual(value, 180.0)
            self.assertEqual(note, "axis1/B8A44FD51C3C")

    def test_resolves_the_delivery_package_path_layout(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(Path(d))
            value, _ = timeline.threshold_from_alarm_config("/x/axis4/axis-ACCC8E9B4972/clip.mkv", cfg)
            self.assertEqual(value, 105.0)

    def test_every_miss_explains_itself_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(Path(d))
            for video, fragment in (
                ("demo.mkv", "no MAC in the path"),
                ("AABBCCDDEEFF.mkv", "not in alarm.json"),
            ):
                value, note = timeline.threshold_from_alarm_config(video, cfg)
                self.assertEqual(value, 0.0, video)
                self.assertIn(fragment, note, video)

            value, note = timeline.threshold_from_alarm_config("B8A44FD51C3C.mkv", Path(d) / "nope.json")
            self.assertEqual(value, 0.0)
            self.assertIn("not found", note)


class CaptionMetricTests(unittest.TestCase):
    SAMPLE = {"count": 173.4, "peak": 4.25}

    def test_plain_series_when_no_threshold_applies(self):
        self.assertEqual(timeline.metric_text(self.SAMPLE, "count"), "count 173")
        self.assertEqual(timeline.metric_text(self.SAMPLE, "peak"), "peak 4.2")

    def test_threshold_rides_only_on_the_series_that_owns_it(self):
        self.assertEqual(timeline.metric_text(self.SAMPLE, "count", 150, "count"), "count 173 / thr 150 OVER")
        # Same call, other series: a count threshold must not appear on peak.
        self.assertEqual(timeline.metric_text(self.SAMPLE, "peak", 150, "count"), "peak 4.2")

    def test_under_threshold_shows_the_line_without_the_flag(self):
        self.assertEqual(timeline.metric_text(self.SAMPLE, "count", 200, "count"), "count 173 / thr 200")

    def test_at_the_threshold_counts_as_over(self):
        # The alarm state machine fires on N >= T, so the caption must agree.
        self.assertIn("OVER", timeline.metric_text({"count": 150.0}, "count", 150, "count"))
