from birdcount.deploy import countlog


def test_the_floor_is_60_percent_of_the_threshold_rounded_up_to_a_bin() -> None:
    assert countlog.floor_of(130) == 80  # 78 -> 80
    assert countlog.floor_of(100) == 60  # on a bin already
    assert countlog.floor_of(95) == 60  # 57 -> 60
    assert countlog.floor_of(None) == 0  # no threshold: every count
