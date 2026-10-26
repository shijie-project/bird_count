"""Bare-MAC identity resolution: `runtime.camera_identity` + the alarm bridge.

These two modules decide which threshold and which region mask attach to a
live camera, so the interesting cases here are the ones where getting it wrong
would be silent: an ambiguous filename, a typo'd topology entry, a MAC that
belongs to no configured camera.
"""

import unittest
from pathlib import Path

from alarm.camera_ids import mac_of, resolve_camera_id
from runtime.camera_identity import extract_mac, index_by_mac, normalize_mac, resolve_stream_mac


MAC = "B8A44FD51C3C"
KNOWN = ["axis1/B8A44FD51C3C", "axis2/B8A44F5B4394", "axis4/ACCC8E9B4972"]


class NormalizeMacTest(unittest.TestCase):
    def test_accepts_the_usual_spellings(self):
        for spelling in ("B8A44FD51C3C", "b8a44fd51c3c", "b8:a4:4f:d5:1c:3c", "B8-A4-4F-D5-1C-3C"):
            self.assertEqual(normalize_mac(spelling), MAC, spelling)

    def test_rejects_non_macs(self):
        for junk in ("", "138.25.209.125", "B8A44FD51C3", "B8A44FD51C3CC", "ZZA44FD51C3C"):
            self.assertEqual(normalize_mac(junk), "", junk)


class ExtractMacTest(unittest.TestCase):
    def test_pulls_the_mac_out_of_every_naming_scheme_in_use(self):
        cases = {
            "B8A44FD51C3C": "bare",
            "mask_axis1_B8A44FD51C3C": "the mask files on disk today",
            "axis1-B8A44FD51C3C_mask": "delivery-package spelling",
            "axis1/B8A44FD51C3C": "alarm config id",
            r"..\data\videos\B8A44FD51C3C.mkv": "video mode source",
            "/data/axis1/axis-B8A44FD51C3C/clip.mkv": "delivery video layout",
        }
        for text, why in cases.items():
            self.assertEqual(extract_mac(text), MAC, why)

    def test_none_when_absent_or_ambiguous(self):
        self.assertIsNone(extract_mac("138.25.209.125"))
        self.assertIsNone(extract_mac(""))
        # Two different MACs in one name: refuse rather than pick one, because
        # either choice silently mis-assigns a threshold.
        self.assertIsNone(extract_mac("B8A44FD51C3C_vs_ACCC8E9B4972"))


class ResolveStreamMacTest(unittest.TestCase):
    def test_explicit_topology_entry_wins(self):
        got = resolve_stream_mac(
            explicit="axis1/B8A44FD51C3C", source="/x/ACCC8E9B4972.mkv", identifier="ACCC8E9B4972"
        )
        self.assertEqual(got, MAC)

    def test_video_mode_falls_back_to_the_clip_name(self):
        got = resolve_stream_mac(explicit=None, source=r"..\data\videos\B8A44FD51C3C.mkv", identifier="B8A44FD51C3C")
        self.assertEqual(got, MAC)

    def test_camera_mode_without_camera_ids_yields_nothing(self):
        got = resolve_stream_mac(
            explicit=None,
            source="https://root:root@138.25.209.125/mjpg/1/video.mjpg",
            identifier="138.25.209.125",
        )
        self.assertIsNone(got)


class IndexByMacTest(unittest.TestCase):
    def test_indexes_mixed_naming(self):
        unique, ambiguous = index_by_mac([Path("B8A44FD51C3C.png"), Path("mask_axis2_B8A44F5B4394.png")])
        self.assertEqual(set(unique), {MAC, "B8A44F5B4394"})
        self.assertEqual(ambiguous, {})

    def test_exact_name_wins_over_a_decorated_duplicate(self):
        unique, ambiguous = index_by_mac([Path("mask_axis1_B8A44FD51C3C.png"), Path("B8A44FD51C3C.png")])
        self.assertEqual(unique[MAC].name, "B8A44FD51C3C.png")
        self.assertEqual(ambiguous, {})

    def test_two_decorated_duplicates_are_reported_not_guessed(self):
        unique, ambiguous = index_by_mac([Path("old_B8A44FD51C3C.png"), Path("new_B8A44FD51C3C.png")])
        self.assertEqual(unique, {})
        self.assertEqual(len(ambiguous[MAC]), 2)


class ResolveCameraIdTest(unittest.TestCase):
    def test_adds_the_axis_prefix_back(self):
        self.assertEqual(resolve_camera_id(mac=MAC, known=KNOWN), "axis1/B8A44FD51C3C")

    def test_case_insensitive(self):
        self.assertEqual(resolve_camera_id(mac="b8a44fd51c3c", known=KNOWN), "axis1/B8A44FD51C3C")

    def test_unknown_mac_is_no_coverage_not_an_error(self):
        self.assertIsNone(resolve_camera_id(mac="AABBCCDDEEFF", known=KNOWN))
        self.assertIsNone(resolve_camera_id(mac=None, known=KNOWN))

    def test_bare_mac_config_needs_no_translation(self):
        self.assertEqual(resolve_camera_id(mac=MAC, known=[MAC]), MAC)

    def test_same_mac_under_two_recorders_is_refused(self):
        # Shouldn't happen, but guessing would attach the wrong threshold.
        self.assertIsNone(resolve_camera_id(mac=MAC, known=["axis1/" + MAC, "axis3/" + MAC]))

    def test_mac_of(self):
        self.assertEqual(mac_of("axis1/B8A44FD51C3C"), MAC)
        self.assertEqual(mac_of(MAC), MAC)


if __name__ == "__main__":
    unittest.main()
