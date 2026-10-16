"""harness-p0-p1 Ticket 03 — dependency pin contract (spec T11).

Pins the working dependency line so a clean environment installs what the
kernel actually runs on:

* ``openai>=2.45,<3`` — the old ``openai==1.50.0`` pin conflicts with
  ``langchain-openai>=1.4`` (which requires openai>=1.58 and supports the
  2.x line); the conflict evidence is recorded in
  ``third_party/uniharness-provenance.md`` and the verification log.
* ``langchain-openai>=1.4,<2`` — the 1.x factory surface the kernel uses.
* ``mistune>=3,<4`` and ``pypdf>=6.10,<7`` — explicit declarations for the
  Ticket 04 PDF rendering/verification path; ReportLab stays as-is.
* ``requirements-test.txt`` pins ``pytest==9.0.3`` /
  ``pytest-asyncio==1.3.0`` without upgrading anything else.

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_dependency_pins_contract.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SERVICE = Path(__file__).resolve().parent

_REQUIREMENTS = "requirements.txt"
_REQUIREMENTS_MINIMAL = "requirements-minimal.txt"
_REQUIREMENTS_TEST = "requirements-test.txt"


def _parse_requirements(path: Path) -> dict[str, str]:
    """Map distribution name → raw specifier (comments/blank lines skipped)."""
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", line)
        assert match, f"unparsable requirement line: {raw_line!r}"
        name, spec = match.group(1).lower(), match.group(2).strip()
        pins[name] = spec
    return pins


@pytest.mark.parametrize("filename", [_REQUIREMENTS, _REQUIREMENTS_MINIMAL])
def test_application_dependencies_declare_working_lines(filename: str):
    pins = _parse_requirements(AI_SERVICE / filename)

    assert pins.get("openai") == ">=2.45,<3", pins.get("openai")
    assert pins.get("langchain-openai") == ">=1.4,<2", pins.get("langchain-openai")
    # DeepEval 4.2.9 requires python-dotenv>=1.1.1; the app and eval
    # requirements must resolve together rather than overwrite each other.
    assert pins.get("python-dotenv") == ">=1.1.1,<2", pins.get("python-dotenv")
    assert pins.get("uvicorn") == "[standard]>=0.31.1,<1", pins.get("uvicorn")
    assert pins.get("httpx") == ">=0.27.1,<0.29", pins.get("httpx")
    # Ticket 04 rendering stack, explicitly declared.
    assert pins.get("mistune") == ">=3,<4", pins.get("mistune")
    assert pins.get("pypdf") == ">=6.10,<7", pins.get("pypdf")
    # ReportLab is retained, not replaced.
    assert pins.get("reportlab", "").startswith(">="), pins.get("reportlab")


def test_test_entry_point_pins_pytest_stack():
    path = AI_SERVICE / _REQUIREMENTS_TEST
    assert path.exists(), f"{_REQUIREMENTS_TEST} missing"
    pins = _parse_requirements(path)

    assert pins.get("pytest") == "==9.0.3", pins.get("pytest")
    assert pins.get("pytest-asyncio") == "==1.3.0", pins.get("pytest-asyncio")
    # The test entry composes the runtime requirements instead of forking them.
    assert "-r requirements.txt" in path.read_text(encoding="utf-8")


def test_declared_ranges_admit_the_installed_working_versions():
    """The ranges must actually accept what the suite runs on today."""
    from importlib.metadata import version
    from packaging.requirements import Requirement

    checks = [
        ("openai", ">=2.45,<3"),
        ("langchain-openai", ">=1.4,<2"),
        ("python-dotenv", ">=1.1.1,<2"),
        ("uvicorn", ">=0.31.1,<1"),
        ("httpx", ">=0.27.1,<0.29"),
        ("mistune", ">=3,<4"),
        ("pypdf", ">=6.10,<7"),
        ("pytest", "==9.0.3"),
        ("pytest-asyncio", "==1.3.0"),
    ]
    for name, spec in checks:
        installed = version(name)
        assert Requirement(f"{name}{spec}").specifier.contains(
            installed, prereleases=True
        ), f"{name} {installed} outside declared range {spec}"


def test_old_openai_pin_conflict_is_gone():
    """Regression: the removed ``openai==1.50.0`` pin is what conflicted
    with langchain-openai>=1.4 (needs openai>=1.58). It must not return."""
    for filename in (_REQUIREMENTS, _REQUIREMENTS_MINIMAL):
        pins = _parse_requirements(AI_SERVICE / filename)
        assert "==1.50.0" not in pins.get("openai", ""), filename
