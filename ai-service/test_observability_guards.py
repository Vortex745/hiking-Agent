"""Static guards for the observability boundary (ticket 07).

These tests pin the *shape* of the integration so future changes cannot
quietly grow Langfuse features (prompt management, datasets/experiments,
human scoring) or leak the backend dialect into business code.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from observability import tracing

SERVICE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_ROOT.parent

BUSINESS_DIRS = (
    "agent_v2",
    "api",
    "mcp_servers",
    "observability",
    "rag",
    "route_compare",
    "scripts",
    "tools",
)
_EXCLUDED_PARTS = {"__pycache__", ".venv", "uniharness", "pytest-of-14253"}
_EXCLUDED_NAME_PATTERNS = (
    "test_*.py",
    "conftest.py",
    "*.pyc",
)
_EXCLUDED_DIR_PREFIXES = ("pip-", "pytest-", "v2-sandbox-")

ADAPTER_PATH = SERVICE_ROOT / "observability" / "export" / "langfuse.py"

# Capabilities the product explicitly does NOT use. Symbol names are
# Langfuse-SDK-specific on purpose: generic words like "prompt" or "score"
# have unrelated meanings inside the RAG pipeline.
FORBIDDEN_CAPABILITY_SYMBOLS = (
    "create_prompt",
    "get_prompt",
    "update_prompt",
    "delete_prompt",
    "create_dataset",
    "create_dataset_item",
    "dataset_run",
    "create_experiment",
    "run_experiment",
    "create_score",
    "score_current_trace",
    "score_observation",
    "score_trace",
    "LangfuseCallbackHandler",
)

_FORBIDDEN_SYMBOL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in FORBIDDEN_CAPABILITY_SYMBOLS) + r")\b"
)
_DIALECT_KEY_RE = re.compile(r'"langfuse\.[a-z]')
_LANGFUSE_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+langfuse(?:\.\w+)*\s+import|import\s+langfuse(?:\.\w+)*)",
    re.MULTILINE,
)

EXPECTED_FACADE_API = {
    "attributes",
    "current_session",
    "current_span",
    "current_trace_id",
    "extract_traceparent",
    "force_flush",
    "generation_attributes",
    "get_tracer",
    "mark_ok",
    "observation_attributes",
    "record_exception",
    "reset_session",
    "set_session",
    "shutdown_tracing",
    "start_span",
    "tracing_enabled",
}


def _business_files() -> list[Path]:
    files: list[Path] = []
    for directory in BUSINESS_DIRS:
        root = SERVICE_ROOT / directory
        if not root.is_dir():
            continue
        files.extend(root.rglob("*.py"))
    files.extend(SERVICE_ROOT.glob("*.py"))
    result = []
    for path in files:
        parts = set(path.relative_to(SERVICE_ROOT).parts)
        if parts & _EXCLUDED_PARTS:
            continue
        if path.name == "conftest.py" or path.name.startswith("test_"):
            continue
        if any(part.startswith(_EXCLUDED_DIR_PREFIXES) for part in parts):
            continue
        result.append(path)
    return result


def _find_symbols(source: str) -> list[str]:
    return sorted(set(_FORBIDDEN_SYMBOL_RE.findall(source)))


def test_forbidden_capability_symbols_do_not_appear_in_business_code():
    offenders = {
        str(path.relative_to(SERVICE_ROOT)): found
        for path in _business_files()
        if (found := _find_symbols(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, offenders


def test_capability_guard_detects_an_introduced_symbol():
    """Red baseline: the guard is not vacuously passing."""
    synthetic = "client.create_prompt(name='x')\nclient.create_dataset('y')\n"
    assert _find_symbols(synthetic) == ["create_dataset", "create_prompt"]


def test_langfuse_dialect_keys_live_only_in_the_adapter():
    offenders = []
    for path in _business_files():
        if path == ADAPTER_PATH:
            continue
        if _DIALECT_KEY_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SERVICE_ROOT)))
    assert offenders == [], offenders


def test_langfuse_package_is_imported_only_by_the_adapter():
    offenders = []
    for path in _business_files():
        if path == ADAPTER_PATH:
            continue
        if _LANGFUSE_IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SERVICE_ROOT)))
    assert offenders == [], offenders


def test_dialect_and_import_guards_detect_introduced_violations():
    """Red baseline for both static scans."""
    synthetic_dialect = 'attrs = {"langfuse.session.id": session}'
    assert _DIALECT_KEY_RE.search(synthetic_dialect)

    synthetic_import = "from langfuse import Langfuse\n"
    assert _LANGFUSE_IMPORT_RE.search(synthetic_import)


def test_trace_facade_api_surface_is_fixed():
    public = {
        name
        for name, value in inspect.getmembers(tracing, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == tracing.__name__
    }
    assert public == EXPECTED_FACADE_API


# ---- Evaluation isolation ---------------------------------------------------


def _eval_python_files() -> list[Path]:
    evals_root = REPO_ROOT / "evals"
    if not evals_root.is_dir():
        return []
    return [
        path
        for path in evals_root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_eval_code_does_not_reference_langfuse():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _eval_python_files()
        if _LANGFUSE_IMPORT_RE.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], offenders


def test_eval_request_ids_use_the_eval_prefix():
    """Evaluation traffic must be filterable via the `eval-` request-id rule."""
    import ast

    literal_re = re.compile(r'"request_id"\s*:\s*f?"([^"]+)"')
    prefixed_call_names = {"_run_single_chat", "_run_chat_turns"}

    offenders: list[str] = []
    for path in _eval_python_files():
        source = path.read_text(encoding="utf-8")
        for match in literal_re.finditer(source):
            value = match.group(1)
            if value.startswith("{"):
                continue  # composed from a validated prefix argument
            if not value.startswith("eval-"):
                offenders.append(f"{path.name}: request_id={value}")

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name not in prefixed_call_names:
                continue
            for keyword in node.keywords:
                if keyword.arg != "prefix" or not isinstance(keyword.value, ast.Constant):
                    continue
                value = keyword.value.value
                if isinstance(value, str) and not value.startswith("eval-"):
                    offenders.append(f"{path.name}: prefix={value}")

    assert offenders == [], offenders
