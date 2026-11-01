"""Decision provider public contract; no live model calls."""
import asyncio
from types import SimpleNamespace
import pytest

from route_compare.providers import ProviderConfig, select_provider
from route_compare.models import DecisionRequest


class Model:
    async def ainvoke(self, messages):
        return SimpleNamespace(content='{"ranking":["a","b"],"confidence":0.9}')


def test_default_provider_uses_llm_and_marks_confidence_heuristic():
    provider = select_provider(ProviderConfig(), llm=Model())
    result = asyncio.run(provider.decide(DecisionRequest(candidate_ids=["a", "b"])))
    assert result.provider == "llm"
    assert result.status == "ok"
    assert result.ranking == ["a", "b"]
    assert result.confidence.kind == "heuristic"


@pytest.mark.parametrize("content", ['bad json', '{"ranking":["ghost"],"confidence":1}',
                                      '{"ranking":["a","a"],"confidence":1}',
                                      '{"ranking":["a","b"],"confidence":2}'])
def test_invalid_model_response_never_becomes_a_recommendation(content):
    class BadModel:
        async def ainvoke(self, messages):
            return SimpleNamespace(content=content)
    result = asyncio.run(select_provider(ProviderConfig(), llm=BadModel()).decide(
        DecisionRequest(candidate_ids=["a", "b"])))
    assert result.status == "invalid_response"
    assert result.ranking == []


@pytest.mark.parametrize("error", [TimeoutError(), PermissionError("private secret")])
def test_provider_failure_is_explicit_and_has_no_sensitive_detail(error):
    class BrokenModel:
        async def ainvoke(self, messages):
            raise error
    result = asyncio.run(select_provider(ProviderConfig(), llm=BrokenModel()).decide(
        DecisionRequest(candidate_ids=["a"])))
    assert result.status == "unavailable"
    assert result.ranking == []
    assert "private" not in result.model_dump_json()


def test_deterministic_is_never_an_implicit_live_fallback():
    result = asyncio.run(select_provider(ProviderConfig(provider="deterministic")).decide(
        DecisionRequest(candidate_ids=["a"])))
    assert result.reason == "offline_only"


def test_optional_external_provider_translates_typed_scores():
    class Client:
        async def evaluate(self, body):
            return {"model": "jev-test", "answers": {
                "a": {"type": "score", "score": 3.0, "confidence": 0.85},
                "b": {"type": "score", "score": 1.0, "confidence": 0.9},
            }}
    provider = select_provider(ProviderConfig(provider="external", external_key="test"), client=Client())
    result = asyncio.run(provider.decide(DecisionRequest(candidate_ids=["a", "b"])))
    assert result.status == "ok"
    assert result.ranking == ["a", "b"]
    assert result.confidence.kind == "calibrated"
    assert result.model == "jev-test"


def test_transient_rate_limit_is_retried_then_succeeds():
    class FlakyModel:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Error code: 429 - rate limited")
            return SimpleNamespace(content='{"ranking":["a","b"],"confidence":0.9}')
    model = FlakyModel()
    result = asyncio.run(select_provider(ProviderConfig(), llm=model).decide(
        DecisionRequest(candidate_ids=["a", "b"])))
    assert result.status == "ok"
    assert result.ranking == ["a", "b"]
    assert model.calls == 2


def test_persistent_rate_limit_stops_after_bounded_attempts():
    class LimitedModel:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            raise RuntimeError("Error code: 429 - rate limited")
    model = LimitedModel()
    result = asyncio.run(select_provider(ProviderConfig(), llm=model).decide(
        DecisionRequest(candidate_ids=["a"])))
    assert result.status == "unavailable"
    assert result.reason == "provider_unavailable"
    assert result.ranking == []
    assert model.calls == 3
    assert "429" not in result.model_dump_json()


def test_invalid_response_is_never_retried():
    class BadModel:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            return SimpleNamespace(content='bad json')
    model = BadModel()
    result = asyncio.run(select_provider(ProviderConfig(), llm=model).decide(
        DecisionRequest(candidate_ids=["a"])))
    assert result.status == "invalid_response"
    assert model.calls == 1


def test_auth_error_is_never_retried_and_leaks_nothing():
    class ForbiddenModel:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            raise PermissionError("invalid api key sk-secret")
    model = ForbiddenModel()
    result = asyncio.run(select_provider(ProviderConfig(), llm=model).decide(
        DecisionRequest(candidate_ids=["a"])))
    assert result.status == "unavailable"
    assert model.calls == 1
    assert "sk-secret" not in result.model_dump_json()


def test_malformed_external_response_is_unavailable_with_no_ranking():
    class Client:
        async def evaluate(self, body):
            return {"model": "jev-test", "answers": {}}
    provider = select_provider(ProviderConfig(provider="external", external_key="test"), client=Client())
    result = asyncio.run(provider.decide(DecisionRequest(candidate_ids=["a"])))
    assert result.status == "unavailable"
    assert result.reason == "invalid_response"
    assert result.ranking == []
    assert result.confidence.kind == "unavailable"
