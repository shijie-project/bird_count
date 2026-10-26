"""Bare-MAC identity resolution decides which mask (and alarm threshold)
attaches to a live camera, so the interesting cases are the ones where a wrong
answer would be silent: an ambiguous filename, a typo, a MAC nobody owns."""

from pathlib import Path

import pytest

from birdcount.core.camera_identity import extract_mac, index_by_mac, normalize_mac, resolve_stream_mac


MAC = "B8A44FD51C3C"


@pytest.mark.parametrize("spelling", ["B8A44FD51C3C", "b8a44fd51c3c", "b8:a4:4f:d5:1c:3c", "B8-A4-4F-D5-1C-3C"])
def test_normalize_accepts_the_usual_spellings(spelling) -> None:
    assert normalize_mac(spelling) == MAC


@pytest.mark.parametrize("junk", ["", "138.25.209.125", "B8A44FD51C3", "B8A44FD51C3CC", "ZZA44FD51C3C"])
def test_normalize_rejects_non_macs(junk) -> None:
    assert normalize_mac(junk) == ""


@pytest.mark.parametrize(
    "text",
    [
        "B8A44FD51C3C",  # bare
        "mask_axis1_B8A44FD51C3C",  # mask files on disk
        "axis1-B8A44FD51C3C_mask",  # delivery-package spelling
        "axis1/B8A44FD51C3C",  # alarm config id
        r"..\data\videos\B8A44FD51C3C.mkv",  # video-mode source
        "/data/axis1/axis-B8A44FD51C3C/clip.mkv",  # delivery video layout
    ],
)
def test_extract_from_every_naming_scheme(text) -> None:
    assert extract_mac(text) == MAC


def test_extract_none_when_absent_or_ambiguous() -> None:
    assert extract_mac("138.25.209.125") is None
    assert extract_mac("") is None
    # Two different MACs: refuse rather than pick, either choice mis-assigns.
    assert extract_mac("B8A44FD51C3C_vs_ACCC8E9B4972") is None


def test_explicit_topology_entry_wins() -> None:
    got = resolve_stream_mac(explicit="axis1/B8A44FD51C3C", source="/x/ACCC8E9B4972.mkv", identifier="ACCC8E9B4972")
    assert got == MAC


def test_video_mode_falls_back_to_the_clip_name() -> None:
    got = resolve_stream_mac(explicit=None, source=r"..\data\videos\B8A44FD51C3C.mkv", identifier="B8A44FD51C3C")
    assert got == MAC


def test_camera_mode_without_camera_ids_yields_nothing() -> None:
    got = resolve_stream_mac(
        explicit=None, source="https://root:root@138.25.209.125/mjpg/1/video.mjpg", identifier="138.25.209.125"
    )
    assert got is None


def test_index_mixed_naming() -> None:
    unique, ambiguous = index_by_mac([Path("B8A44FD51C3C.png"), Path("mask_axis2_B8A44F5B4394.png")])
    assert set(unique) == {MAC, "B8A44F5B4394"}
    assert ambiguous == {}


def test_exact_name_wins_over_a_decorated_duplicate() -> None:
    unique, ambiguous = index_by_mac([Path("mask_axis1_B8A44FD51C3C.png"), Path("B8A44FD51C3C.png")])
    assert unique[MAC].name == "B8A44FD51C3C.png"
    assert ambiguous == {}


def test_two_decorated_duplicates_are_reported_not_guessed() -> None:
    unique, ambiguous = index_by_mac([Path("old_B8A44FD51C3C.png"), Path("new_B8A44FD51C3C.png")])
    assert unique == {}
    assert len(ambiguous[MAC]) == 2
