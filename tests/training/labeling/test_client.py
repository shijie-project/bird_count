import httpx
import pytest

from birdcount.training.labeling.client import LabelStudio, LabelStudioError, jwt_claims

from .conftest import REFRESH, FakeLabelStudio


def test_a_personal_token_is_exchanged_once_and_renewed_on_401(fake, ls) -> None:
    ls.project(1)
    ls.project(1)
    assert fake.refreshes == 1  # the access token is reused while valid
    fake.expire_access()  # the server forgets it (expired, revoked)
    assert ls.project(1)["title"] == "Birds"
    assert fake.refreshes == 2


def test_an_access_token_about_to_expire_is_renewed_first(fake, ls) -> None:
    fake.access_ttl = 10  # inside the renewal margin
    ls.project(1)
    ls.project(1)
    assert fake.refreshes == 2
    assert ("GET", "/api/projects/1/") in fake.calls


def test_a_legacy_token_is_sent_as_is() -> None:
    fake = FakeLabelStudio()
    with LabelStudio("http://ls.test", "legacy-token", transport=httpx.MockTransport(fake.handler)) as ls:
        assert ls.project(1)["id"] == 1
    assert fake.refreshes == 0


def test_errors_name_the_call_never_the_token(fake, ls) -> None:
    with pytest.raises(LabelStudioError, match=r"GET /api/projects/9/: HTTP 404") as e:
        ls.project(9)
    assert REFRESH not in str(e.value)
    bad = LabelStudio("http://ls.test", REFRESH[:-3] + "xyz", transport=httpx.MockTransport(fake.handler))
    with pytest.raises(LabelStudioError, match="refused") as e:
        bad.project(1)
    assert "xyz" not in str(e.value)
    with pytest.raises(LabelStudioError, match="LABEL_STUDIO_API_KEY"):
        LabelStudio("http://ls.test", "")


def test_tasks_are_read_page_by_page() -> None:
    fake = FakeLabelStudio(page_size_cap=7)  # a server capping page_size below ours
    for i in range(23):
        fake.add_task(f"/data/local-files/?d=images%5Call%5C{i:03d}.jpg")
    with LabelStudio("http://ls.test", REFRESH, transport=httpx.MockTransport(fake.handler)) as ls:
        assert [t["data"]["img"][-7:] for t in ls.tasks(1)] == [f"{i:03d}.jpg" for i in range(23)]


def test_jwt_claims() -> None:
    assert jwt_claims(REFRESH)["token_type"] == "refresh"
    assert jwt_claims("0123456789abcdef") is None
    assert jwt_claims("a.!!!.c") is None
