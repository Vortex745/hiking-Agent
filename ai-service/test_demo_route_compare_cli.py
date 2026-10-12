"""Demo script CLI contract (issue 15); offline default stays drift-free.

Runs the script as a subprocess — the CLI is the seam. Full offline/live
chain runs stay manual: they need sandbox + MCP credentials.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "demo_route_compare.py"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    # The CLI reports errors in Chinese and the parent decodes UTF-8, so pin the
    # child's stream encoding instead of letting Windows default it to cp936.
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent),
           "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=str(Path(__file__).parent),
                          env=env, encoding="utf-8")


def test_default_data_mode_is_offline():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--data-mode" in result.stdout
    assert "{offline,live}" in result.stdout


def test_live_without_candidates_json_fails_explicitly(tmp_path: Path):
    result = run_cli("--data-mode", "live", "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2
    assert "--candidates-json" in result.stderr


def test_live_with_missing_candidates_file_fails_explicitly(tmp_path: Path):
    result = run_cli("--data-mode", "live", "--candidates-json", str(tmp_path / "nope.json"),
                     "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 1
    assert "不存在" in result.stderr


def test_live_with_malformed_candidates_file_fails_explicitly(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"city": "缺少name"}]), encoding="utf-8")
    result = run_cli("--data-mode", "live", "--candidates-json", str(bad),
                     "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 1
    assert "name" in result.stderr
