"""`tools/discover_camera_ids.py` logic, with the network stubbed out.

The VAPIX call itself can only be exercised on the farm network, so what is
tested here is everything around it: parsing a camera's answer, matching it to
thresholds and masks, generating the paste-ready block, and — the one that
matters most — `--verify` noticing that topology.yaml disagrees with reality.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tools import discover_camera_ids as tool


MAC_A, MAC_B, MAC_C = "B8A44FD51C3C", "B8A44FD51C87", "ACCC8E9B4972"

ZONES = [
    {"name": "Zone_01", "cameras": ["10.0.0.1", "10.0.0.2"], "camera_ids": [MAC_A, MAC_B]},
    {"name": "Zone_02", "cameras": ["10.0.0.3"]},
]

LIVE = {"10.0.0.1": MAC_A, "10.0.0.2": MAC_B, "10.0.0.3": MAC_C}


def _probe(zones, live):
    """Run the discovery pass against a fake network."""

    def fake_fetch(ip, auth, timeout):
        return (live[ip], "") if ip in live else (None, "ConnectTimeout")

    with mock.patch.object(tool, "_fetch_mac", side_effect=fake_fetch):
        with redirect_stdout(io.StringIO()):
            return tool._probe_all(zones, ("root", "root"), 1.0, 4)


class ParamParsingTest(unittest.TestCase):
    def test_reads_the_mac_out_of_a_param_cgi_body(self):
        body = "root.Network.eth0.MACAddress=B8:A4:4F:D5:1C:3C\n"
        match = tool._PARAM_LINE.search(body)
        self.assertIsNotNone(match)
        self.assertEqual(tool.normalize_mac(match.group(1)), MAC_A)

    def test_ignores_a_body_without_a_mac(self):
        self.assertIsNone(tool._PARAM_LINE.search("# Error: Error setting param\n"))


class ProbeTest(unittest.TestCase):
    def test_reports_reachable_and_unreachable_separately(self):
        probes = _probe(ZONES, {"10.0.0.1": MAC_A})
        self.assertEqual(len(probes), 3)
        found = {p.ip: p.mac for p in probes if p.ok}
        self.assertEqual(found, {"10.0.0.1": MAC_A})
        self.assertEqual([p.error for p in probes if not p.ok], ["ConnectTimeout"] * 2)

    def test_slots_stay_aligned_with_the_zone_camera_list(self):
        probes = _probe(ZONES, LIVE)
        self.assertEqual([(p.zone, p.slot, p.mac) for p in probes][:2], [("Zone_01", 0, MAC_A), ("Zone_01", 1, MAC_B)])


class BlockOutputTest(unittest.TestCase):
    def test_emits_a_paste_ready_block(self):
        probes = _probe(ZONES, LIVE)
        buf = io.StringIO()
        with redirect_stdout(buf):
            tool._print_block(ZONES, probes)
        out = buf.getvalue()
        self.assertIn("camera_ids:", out)
        self.assertIn(f'- "{MAC_A}"', out)
        self.assertIn(f'- "{MAC_C}"', out)

    def test_unreachable_camera_keeps_its_slot_and_says_why(self):
        probes = _probe(ZONES, {"10.0.0.1": MAC_A})
        buf = io.StringIO()
        with redirect_stdout(buf):
            tool._print_block(ZONES, probes)
        out = buf.getvalue()
        # An empty entry, not a dropped line: the list is positional, so a
        # missing entry would shift every camera after it onto a wrong MAC.
        entries = [ln for ln in out.splitlines() if ln.startswith("      - ")]
        self.assertEqual(len(entries), 3, f"expected one entry per camera, got {entries}")
        self.assertEqual(sum(ln.strip().startswith('- ""') for ln in entries), 2)
        self.assertIn("UNRESOLVED: ConnectTimeout", out)


class VerifyTest(unittest.TestCase):
    def _verify(self, zones, live):
        probes = _probe(zones, live)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = tool._verify(zones, probes)
        return code, buf.getvalue()

    def test_clean_when_topology_matches_reality(self):
        code, out = self._verify(ZONES, LIVE)
        self.assertEqual(code, 0)
        self.assertIn("2 correct, 0 WRONG", out)

    def test_flags_a_swapped_camera(self):
        # The classic silent failure: .1 and .2 were re-IPed against each other,
        # so both are running on the other's threshold and mask.
        code, out = self._verify(ZONES, {"10.0.0.1": MAC_B, "10.0.0.2": MAC_A, "10.0.0.3": MAC_C})
        self.assertEqual(code, 1, "a wrong mapping must be a non-zero exit")
        self.assertIn("2 WRONG", out)
        self.assertIn(f"topology says {MAC_A}, camera says {MAC_B}", out)

    def test_unmapped_camera_is_reported_but_not_an_error(self):
        code, out = self._verify(ZONES, LIVE)  # Zone_02 has no camera_ids
        self.assertEqual(code, 0)
        self.assertIn("1 unmapped", out)
        self.assertIn(f"10.0.0.3: camera says {MAC_C}", out)

    def test_unreachable_camera_is_not_treated_as_a_mismatch(self):
        code, out = self._verify(ZONES, {"10.0.0.1": MAC_A})
        self.assertEqual(code, 0, "an offline camera says nothing about the mapping")
        self.assertIn("2 unreachable", out)


class CrossCheckTest(unittest.TestCase):
    def test_thresholds_are_keyed_by_bare_mac(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "alarm.json"
            cfg.write_text(json.dumps({"cameras": [{"camera_id": f"axis1/{MAC_A}", "threshold": 180}]}))
            self.assertEqual(tool._known_thresholds(cfg), {MAC_A: 180})

    def test_missing_alarm_config_is_not_fatal(self):
        self.assertEqual(tool._known_thresholds(Path("nope.json")), {})

    def test_masks_are_found_under_either_naming(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / f"{MAC_A}.png").write_bytes(b"x")
            (root / f"mask_axis1_{MAC_B}.png").write_bytes(b"x")
            (root / "notes.txt").write_bytes(b"x")
            self.assertEqual(set(tool._known_masks(root)), {MAC_A, MAC_B})

    def test_coverage_flags_a_camera_with_no_threshold_or_mask(self):
        probes = _probe(ZONES, LIVE)
        buf = io.StringIO()
        with redirect_stdout(buf):
            tool._report_coverage(probes, {MAC_A: 180}, {MAC_A: Path(f"{MAC_A}.png")})
        out = buf.getvalue()
        self.assertEqual(out.count("<- incomplete"), 2)


if __name__ == "__main__":
    unittest.main()
