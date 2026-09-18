"""Contract tests for the thesis LLM arm; no network and no secret-file access."""
from __future__ import annotations

import json

from src.analysis.llm_client import AzureOpenAIProvider, DeepSeekProvider
from src.research.llm_trader import ThesisLLMRunner, parse_decision, provider_from_environment


class FakeProvider:
    def __init__(self, content: str):
        self.content = content

    def generate(self, system_prompt, user_prompt, max_tokens, temperature):
        return {"content": self.content, "tokens_used": 7, "cost_known": False}


class SequenceProvider:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0

    def generate(self, system_prompt, user_prompt, max_tokens, temperature):
        self.calls += 1
        return {"content": next(self.contents), "tokens_used": 7, "cost_known": False}


def test_parse_decision_maps_to_five_level_weight_and_rejects_invalid():
    valid = parse_decision({"direccion": "long", "tamano": 0.5, "confianza": 0.8}, -1.0)
    assert valid.valid and valid.weight == 0.5
    invalid = parse_decision({"direccion": "buy", "tamano": 1, "confianza": 1}, -0.5)
    assert not invalid.valid and invalid.weight == -0.5


def test_runner_records_hashes_and_keeps_previous_weight_on_bad_json(tmp_path):
    ledger = tmp_path / "llm.jsonl"
    runner = ThesisLLMRunner(FakeProvider("not json"), "deepseek", "deepseek-chat", ledger)
    decision = runner.decide(session_date="2026-01-02", bar=0, system_prompt="s", user_prompt="u",
                             previous_weight=0.5)
    assert not decision.valid and decision.weight == 0.5
    row = json.loads(ledger.read_text(encoding="utf-8"))
    assert row["valid_json"] is False
    assert row["unavailable"] is False
    assert row["temperature"] == 0.1 and row["top_p"] == 0.9 and row["max_tokens"] == 256
    assert len(row["prompt_hash"]) == 64 and len(row["raw_response_sha256"]) == 64
    assert "api_key" not in row and "secret" not in row


def test_runner_records_valid_json_without_network(tmp_path):
    ledger = tmp_path / "llm.jsonl"
    payload = json.dumps({"direccion": "short", "tamano": 1, "confianza": 0.7})
    runner = ThesisLLMRunner(FakeProvider(payload), "azure_openai", "deployment-v1", ledger)
    decision = runner.decide(session_date="2026-01-02", bar=58, system_prompt="s", user_prompt="u",
                             previous_weight=0.0)
    assert decision.valid and decision.weight == -1.0


def test_runner_retries_once_after_invalid_json_and_records_final_attempt(tmp_path):
    payload = json.dumps({"direccion": "long", "tamano": 1, "confianza": 0.9})
    provider = SequenceProvider(["not json", payload])
    runner = ThesisLLMRunner(provider, "deepseek", "deepseek-chat", tmp_path / "llm.jsonl")
    decision = runner.decide(session_date="2026-01-02", bar=1,
                             system_prompt="s", user_prompt="u", previous_weight=0.0)
    assert decision.valid and decision.weight == 1.0
    assert provider.calls == 2
    row = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8"))
    assert row["attempt"] == 2 and row["valid_json"] is True


def test_runner_distinguishes_provider_failure_from_invalid_json(tmp_path):
    class DownProvider:
        def generate(self, *args):
            raise TimeoutError("simulated")

    ledger = tmp_path / "llm.jsonl"
    runner = ThesisLLMRunner(DownProvider(), "deepseek", "deepseek-chat", ledger)
    decision = runner.decide(session_date="2026-01-02", bar=0, system_prompt="s", user_prompt="u",
                             previous_weight=0.25)
    assert not decision.valid and decision.weight == 0.25
    row = json.loads(ledger.read_text(encoding="utf-8"))
    assert row["unavailable"] is True and row["valid_json"] is False


def test_runner_rejects_invalid_sampling_parameters(tmp_path):
    runner = ThesisLLMRunner(FakeProvider("{}"), "deepseek", "deepseek-chat", tmp_path / "llm.jsonl")
    try:
        runner.decide(session_date="2026-01-02", bar=0, system_prompt="s", user_prompt="u",
                      previous_weight=0.0, top_p=0.0)
    except ValueError as exc:
        assert "sampling" in str(exc)
    else:
        raise AssertionError("invalid sampling parameters must fail closed")


def test_runner_refuses_duplicate_before_provider_call(tmp_path):
    ledger = tmp_path / "llm.jsonl"
    payload = json.dumps({"direccion": "long", "tamano": 0.5, "confianza": 0.8})
    runner = ThesisLLMRunner(FakeProvider(payload), "deepseek", "deepseek-chat", ledger)
    runner.decide(session_date="2026-01-02", bar=0, system_prompt="s", user_prompt="u",
                  previous_weight=0.0)
    duplicate = ThesisLLMRunner(FakeProvider(payload), "deepseek", "deepseek-chat", ledger)
    try:
        duplicate.decide(session_date="2026-01-02", bar=0, system_prompt="s", user_prompt="u",
                         previous_weight=0.0)
    except ValueError as exc:
        assert "duplicate decision_id" in str(exc)
    else:
        raise AssertionError("duplicate decision should be refused")


def test_provider_selection_is_explicit(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-not-a-real-secret")
    provider = provider_from_environment("deepseek")
    assert isinstance(provider, DeepSeekProvider)
    assert provider.health_check() is True


def test_deepseek_without_key_fails_closed(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepSeekProvider()
    assert provider.health_check() is False


def test_azure_provider_uses_frozen_environment_deployment(monkeypatch):
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_DEPLOYMENT", "thesis-deployment")
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_API_VERSION", "2025-01-01")
    provider = AzureOpenAIProvider()
    assert provider.deployment == "thesis-deployment"
    assert provider.api_version == "2025-01-01"
    assert provider.health_check() is True
