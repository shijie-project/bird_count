"""`.env.example` and the settings classes describe exactly the same keys.

`Settings.load()` treats `.env.example` as the list of valid keys, so a field
missing from it would be reported as a typo, and a stale entry would hide a
real one.
"""

from __future__ import annotations

import importlib
import pkgutil

import birdcount
from birdcount.core.settings import ENV_EXAMPLE_FILE, Settings, env_file_keys


def _all_settings_classes() -> list[type[Settings]]:
    # Settings live in modules named `settings`; import them all so every
    # subclass is registered before walking the hierarchy.
    for info in pkgutil.walk_packages(birdcount.__path__, prefix="birdcount."):
        if info.name.rsplit(".", 1)[-1] == "settings":
            importlib.import_module(info.name)
    found, stack = [], [Settings]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("birdcount."):
                found.append(sub)
            stack.append(sub)
    return found


def _field_keys() -> dict[str, str]:
    return {
        field.upper(): f"{cls.__module__}.{cls.__name__}"
        for cls in _all_settings_classes()
        for field in cls.model_fields
    }


def test_every_setting_is_documented() -> None:
    documented = env_file_keys(ENV_EXAMPLE_FILE)
    missing = {key: owner for key, owner in _field_keys().items() if key not in documented}
    assert not missing, f".env.example lacks: {missing}"


def test_every_documented_key_is_read() -> None:
    stale = env_file_keys(ENV_EXAMPLE_FILE) - set(_field_keys())
    assert not stale, f".env.example documents keys no setting reads: {sorted(stale)}"
