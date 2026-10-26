"""Static checks on the import graph — the package structure, enforced.

1. Dependency direction: `core` imports nothing from the apps; `deploy` and
   `training` import `core` but never each other.
2. Private modules (`_name`) are imported only from inside the package that
   owns them; otherwise a "private" module becomes de-facto public API that
   nobody can change safely.
3. The control panel is a leaf: nothing imports it. (v2's evaluation script
   imported the web UI package and with it FastAPI, so it could not run
   where FastAPI was missing.)
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from birdcount.core.paths import PROJECT_ROOT


PACKAGE = "birdcount"
PACKAGE_DIR = PROJECT_ROOT / PACKAGE

ALLOWED_LAYERS = {
    "core": {"core"},
    "deploy": {"core", "deploy"},
    "training": {"core", "training"},
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> Iterator[tuple[int, str]]:
    """(line, absolute dotted name) for every import in `path`.

    `from pkg import name` yields both `pkg` and `pkg.name`, since `name` may
    be a submodule — which is exactly how a private module gets imported.
    """
    module = _module_name(path)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")
                drop = node.level - 1 if is_package else node.level
                base = base[: len(base) - drop] if drop else base
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            yield node.lineno, target
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{target}.{alias.name}"


def _python_files(*roots: Path) -> list[Path]:
    return sorted(p for root in roots if root.exists() for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _fail_on(violations: list[str]) -> None:
    # `from a import b` reports both `a` and `a.b`; one line per offending
    # import statement is enough.
    seen, unique = set(), []
    for v in violations:
        where = v.split(": ", 1)[0]
        if where not in seen:
            seen.add(where)
            unique.append(v)
    assert not unique, "\n".join(unique)


def test_package_exists() -> None:
    assert (PACKAGE_DIR / "__init__.py").is_file()


def test_dependency_direction() -> None:
    violations = []
    for path in _python_files(PACKAGE_DIR):
        importer = _module_name(path).split(".")
        if len(importer) < 2 or importer[1] not in ALLOWED_LAYERS:
            continue
        layer = importer[1]
        for line, target in _imports(path):
            parts = target.split(".")
            if (
                parts[0] == PACKAGE
                and len(parts) > 1
                and parts[1] in ALLOWED_LAYERS
                and parts[1] not in ALLOWED_LAYERS[layer]
            ):
                violations.append(f"{_rel(path)}:{line}: {layer} must not import {target}")
    _fail_on(violations)


def test_private_modules_stay_private() -> None:
    violations = []
    for path in _python_files(PACKAGE_DIR):
        importer = _module_name(path)
        for line, target in _imports(path):
            parts = target.split(".")
            if parts[0] != PACKAGE:
                continue
            for i, part in enumerate(parts):
                if part.startswith("_") and not part.startswith("__"):
                    owner = ".".join(parts[:i])
                    if importer != owner and not importer.startswith(owner + "."):
                        violations.append(f"{_rel(path)}:{line}: {target} is private to {owner}")
                    break
    _fail_on(violations)


def test_the_panel_is_a_leaf() -> None:
    panel = f"{PACKAGE}.training.webui"
    violations = [
        f"{_rel(path)}:{line}: imports {target}"
        for path in _python_files(PACKAGE_DIR)
        if not _module_name(path).startswith(panel)
        for line, target in _imports(path)
        if target == panel or target.startswith(panel + ".")
    ]
    _fail_on(violations)
