"""Declared Python support must match what CI actually tests."""

from __future__ import annotations

import re
import tomllib

import yaml

from tests.conftest import ROOT


def _version(s: str) -> tuple[int, int]:
    major, minor = s.split(".")[:2]
    return int(major), int(minor)


def test_declared_python_floor_is_the_oldest_ci_tested_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    m = re.fullmatch(r">=\s*(\d+\.\d+)", project["project"]["requires-python"])
    assert m, "requires-python must be a simple lower bound"
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    tested = sorted(_version(v) for v in ci["jobs"]["test"]["strategy"]["matrix"]["python"])
    assert _version(m.group(1)) == tested[0], (m.group(1), tested)
    ruff = project["tool"]["ruff"]["target-version"]
    assert ruff == "py{}{}".format(*tested[0]), ruff
