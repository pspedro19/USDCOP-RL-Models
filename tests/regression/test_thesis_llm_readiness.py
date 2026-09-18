from __future__ import annotations

import sys
from pathlib import Path

from scripts.validation.check_thesis_llm_readiness import check


def test_deepseek_readiness_checks_presence_without_network(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = check("deepseek")
    assert result["ready"] is False
    assert result["network_called"] is False
    assert result["secrets_printed"] is False


def test_azure_readiness_requires_key_and_endpoint(monkeypatch):
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.delenv("USDCOP_AZURE_OPENAI_ENDPOINT", raising=False)
    result = check("azure_openai")
    assert result["ready"] is False
    assert "USDCOP_AZURE_OPENAI_ENDPOINT" in result["missing"]


def test_azure_readiness_requires_deployment(monkeypatch):
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("USDCOP_AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.delenv("USDCOP_AZURE_OPENAI_DEPLOYMENT", raising=False)
    result = check("azure_openai")
    assert result["ready"] is False
    assert "USDCOP_AZURE_OPENAI_DEPLOYMENT" in result["missing"]


def test_operator_dotenv_path_is_forwarded_without_printing_values(monkeypatch, tmp_path, capsys):
    import dotenv
    from scripts.validation import check_thesis_llm_readiness as module

    captured = {}

    def fake_load(path, override=False):
        captured["path"] = Path(path)
        captured["override"] = override
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load)
    monkeypatch.setattr(sys, "argv", ["readiness", "--load-dotenv",
                                       "--dotenv-path", str(tmp_path / "operator.env")])
    assert module.main() == 0
    assert captured == {"path": tmp_path / "operator.env", "override": False}
    assert "test-only" not in capsys.readouterr().out
