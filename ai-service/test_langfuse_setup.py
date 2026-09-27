import sys
import types

from observability import langfuse_setup


def _reset() -> None:
    langfuse_setup._callbacks = None


def _configure(settings, **values) -> None:
    for name, value in values.items():
        setattr(settings, name, value)


def test_callbacks_empty_when_keys_missing(monkeypatch) -> None:
    _reset()
    _configure(
        langfuse_setup.settings,
        langfuse_enabled=True,
        langfuse_public_key="",
        langfuse_secret_key="",
    )
    assert langfuse_setup.get_langchain_callbacks() == []
    _reset()


def test_callbacks_empty_when_disabled(monkeypatch) -> None:
    _reset()
    _configure(
        langfuse_setup.settings,
        langfuse_enabled=False,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    assert langfuse_setup.get_langchain_callbacks() == []
    _reset()


def test_callbacks_return_handler_when_configured(monkeypatch) -> None:
    _reset()
    _configure(
        langfuse_setup.settings,
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://cloud.langfuse.com",
    )

    class _FakeLangfuse:
        def __init__(self, **_kwargs) -> None:
            pass

    class _FakeHandler:
        pass

    fake_pkg = types.ModuleType("langfuse")
    fake_pkg.Langfuse = _FakeLangfuse
    fake_sub = types.ModuleType("langfuse.langchain")
    fake_sub.CallbackHandler = _FakeHandler
    monkeypatch.setitem(sys.modules, "langfuse", fake_pkg)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_sub)

    callbacks = langfuse_setup.get_langchain_callbacks()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], _FakeHandler)
    # 结果应被缓存，避免重复初始化 SDK 单例。
    assert langfuse_setup.get_langchain_callbacks() is callbacks
    _reset()


def test_callbacks_empty_when_sdk_init_fails(monkeypatch) -> None:
    _reset()
    _configure(
        langfuse_setup.settings,
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://cloud.langfuse.com",
    )

    def _boom(**_kwargs):
        raise RuntimeError("sdk init exploded")

    fake_pkg = types.ModuleType("langfuse")
    fake_pkg.Langfuse = _boom
    fake_sub = types.ModuleType("langfuse.langchain")
    fake_sub.CallbackHandler = object
    monkeypatch.setitem(sys.modules, "langfuse", fake_pkg)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_sub)

    assert langfuse_setup.get_langchain_callbacks() == []
    _reset()


def test_langchain_run_config_session_scope() -> None:
    from observability.langfuse_setup import (
        langchain_run_config,
        reset_langfuse_session,
        set_langfuse_session,
    )

    assert langchain_run_config() == {}
    token = set_langfuse_session("chat-1")
    try:
        assert langchain_run_config() == {
            "metadata": {"langfuse_session_id": "chat-1"}
        }
    finally:
        reset_langfuse_session(token)
    assert langchain_run_config() == {}
