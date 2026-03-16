"""
LLM Client (SDD-07 §2)
========================
Strategy pattern: Azure OpenAI (primary) + Anthropic (fallback).
Handles retries, cost tracking, and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> dict:
        """Generate a response.

        Returns:
            {content: str, tokens_used: int, model: str, cost_usd: float}
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: str = "gpt-4o-mini",
        api_version: str = "2024-12-01-preview",
    ):
        self.api_key = api_key or os.environ.get("USDCOP_AZURE_OPENAI_API_KEY", "")
        self.endpoint = endpoint or os.environ.get("USDCOP_AZURE_OPENAI_ENDPOINT", "")
        self.deployment = deployment
        self.api_version = api_version

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> dict:
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI API key or endpoint not configured")

        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

        response = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = response.usage
        # GPT-4o-mini pricing: ~$0.00015/1K input + $0.0006/1K output
        cost = (usage.prompt_tokens * 0.00015 + usage.completion_tokens * 0.0006) / 1000

        return {
            "content": response.choices[0].message.content,
            "tokens_used": usage.total_tokens,
            "model": f"azure/{self.deployment}",
            "cost_usd": round(cost, 6),
        }

    def health_check(self) -> bool:
        return bool(self.api_key and self.endpoint)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (fallback)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250514",
    ):
        self.api_key = api_key or os.environ.get("USDCOP_ANTHROPIC_API_KEY", "")
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> dict:
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Sonnet pricing: ~$0.003/1K input + $0.015/1K output
        cost = (input_tokens * 0.003 + output_tokens * 0.015) / 1000

        return {
            "content": response.content[0].text,
            "tokens_used": input_tokens + output_tokens,
            "model": f"anthropic/{self.model}",
            "cost_usd": round(cost, 6),
        }

    def health_check(self) -> bool:
        return bool(self.api_key)


class LLMClient:
    """Orchestrates LLM calls with fallback, caching, and cost tracking.

    Primary: Azure OpenAI (GPT-4o)
    Fallback: Anthropic (Claude Sonnet)
    Switch to fallback after max_consecutive_failures.
    """

    def __init__(
        self,
        primary: Optional[LLMProvider] = None,
        fallback: Optional[LLMProvider] = None,
        max_failures: int = 3,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
    ):
        self.primary = primary or AzureOpenAIProvider()
        self.fallback = fallback or AnthropicProvider()
        self.max_failures = max_failures
        self._failure_count = 0
        self._total_cost = 0.0
        self._total_tokens = 0

        # File-based cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_hours = cache_ttl_hours
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        cache_key: Optional[str] = None,
    ) -> dict:
        """Generate response with fallback and caching.

        Returns:
            {content, tokens_used, model, cost_usd, cached}
        """
        # Check cache
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                cached["cached"] = True
                return cached

        # Try primary provider
        provider = self.primary if self._failure_count < self.max_failures else self.fallback

        try:
            result = provider.generate(system_prompt, user_prompt, max_tokens, temperature)
            self._failure_count = 0  # Reset on success
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            self._failure_count += 1

            if provider is not self.fallback:
                try:
                    result = self.fallback.generate(
                        system_prompt, user_prompt, max_tokens, temperature,
                    )
                except Exception as e2:
                    logger.error(f"Fallback provider also failed: {e2}")
                    raise
            else:
                raise

        # Track costs
        self._total_cost += result.get("cost_usd", 0)
        self._total_tokens += result.get("tokens_used", 0)
        result["cached"] = False

        # Save to cache
        if cache_key:
            self._set_cached(cache_key, result)

        return result

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[dict] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        cache_key: Optional[str] = None,
    ) -> dict:
        """Generate response with structured JSON output (response_format).

        For Azure OpenAI: uses response_format parameter for guaranteed JSON schema.
        For Anthropic: falls back to regex JSON extraction from generate().

        Returns:
            {content: dict (parsed JSON), raw_content: str, tokens_used, model, cost_usd, cached}
        """
        # Check cache
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                cached["cached"] = True
                return cached

        provider = self.primary if self._failure_count < self.max_failures else self.fallback

        try:
            if isinstance(provider, AzureOpenAIProvider) and response_format:
                result = self._generate_azure_structured(
                    provider, system_prompt, user_prompt,
                    response_format, max_tokens, temperature,
                )
            else:
                # Fallback: regular generate + JSON extraction
                result = provider.generate(system_prompt, user_prompt, max_tokens, temperature)
                raw = result.get("content", "")
                result["raw_content"] = raw
                result["content"] = self._extract_json(raw)

            self._failure_count = 0
        except Exception as e:
            logger.warning(f"Primary structured generation failed: {e}")
            self._failure_count += 1

            if provider is not self.fallback:
                try:
                    fallback_result = self.fallback.generate(
                        system_prompt, user_prompt, max_tokens, temperature,
                    )
                    raw = fallback_result.get("content", "")
                    result = fallback_result
                    result["raw_content"] = raw
                    result["content"] = self._extract_json(raw)
                except Exception as e2:
                    logger.error(f"Fallback structured generation also failed: {e2}")
                    raise
            else:
                raise

        self._total_cost += result.get("cost_usd", 0)
        self._total_tokens += result.get("tokens_used", 0)
        result["cached"] = False

        if cache_key:
            self._set_cached(cache_key, result)

        return result

    @staticmethod
    def _generate_azure_structured(
        provider: "AzureOpenAIProvider",
        system_prompt: str,
        user_prompt: str,
        response_format: dict,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Generate with Azure OpenAI response_format for guaranteed JSON schema."""
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = AzureOpenAI(
            api_key=provider.api_key,
            azure_endpoint=provider.endpoint,
            api_version=provider.api_version,
        )

        response = client.chat.completions.create(
            model=provider.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

        usage = response.usage
        cost = (usage.prompt_tokens * 0.00015 + usage.completion_tokens * 0.0006) / 1000
        raw_content = response.choices[0].message.content

        import json as _json
        try:
            parsed = _json.loads(raw_content)
        except _json.JSONDecodeError:
            parsed = {"raw": raw_content}

        return {
            "content": parsed,
            "raw_content": raw_content,
            "tokens_used": usage.total_tokens,
            "model": f"azure/{provider.deployment}",
            "cost_usd": round(cost, 6),
        }

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response text (for non-structured providers)."""
        import json as _json
        import re

        # Try direct parse
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(1))
            except _json.JSONDecodeError:
                pass

        # Try to find first { ... } block
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(0))
            except _json.JSONDecodeError:
                pass

        return {"raw": text}

    @property
    def total_cost(self) -> float:
        return round(self._total_cost, 4)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    # ------------------------------------------------------------------
    # File-based cache
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _get_cached(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("_cached_at", ""))
            age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
            if age_hours > self.cache_ttl_hours:
                path.unlink(missing_ok=True)
                return None
            data.pop("_cached_at", None)
            return data
        except Exception:
            return None

    def _set_cached(self, key: str, data: dict) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            cache_data = {**data, "_cached_at": datetime.utcnow().isoformat()}
            path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
