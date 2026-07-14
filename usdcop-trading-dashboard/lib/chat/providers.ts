/**
 * Chat providers (CTR-CHAT-001) — Azure OpenAI (primary) + Anthropic (fallback).
 *
 * Each provider reads its own env and reports `configured`. Defaults match the
 * working analysis pipeline SSOT (config/analysis/weekly_analysis_ssot.yaml):
 * deployment `gpt-4o-mini`, api-version `2024-12-01-preview` — NOT the generic
 * `gpt-4o`/`2024-10-21`, which is not deployed on this Azure resource and was the
 * cause of the chatbot silently failing.
 */

import type { ChatCompletionOptions, ChatProvider, ChatResult } from './types';

const DEFAULT_MAX_TOKENS = 2000;
const DEFAULT_TEMPERATURE = 0.7;

class AzureOpenAIProvider implements ChatProvider {
  readonly name = 'azure-openai';
  private readonly key = process.env.USDCOP_AZURE_OPENAI_API_KEY ?? '';
  private readonly endpoint = (process.env.USDCOP_AZURE_OPENAI_ENDPOINT ?? '').replace(/\/+$/, '');
  private readonly deployment = process.env.USDCOP_AZURE_OPENAI_DEPLOYMENT || 'gpt-4o-mini';
  private readonly apiVersion = process.env.USDCOP_AZURE_OPENAI_API_VERSION || '2024-12-01-preview';

  get configured(): boolean {
    return this.key.length > 10 && this.endpoint.startsWith('http');
  }

  async complete(system: string, user: string, opts: ChatCompletionOptions = {}): Promise<ChatResult> {
    const url = `${this.endpoint}/openai/deployments/${this.deployment}/chat/completions?api-version=${this.apiVersion}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'api-key': this.key },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
        max_tokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
        temperature: opts.temperature ?? DEFAULT_TEMPERATURE,
      }),
    });
    if (!res.ok) {
      throw new Error(`azure-openai ${res.status}: ${(await res.text()).slice(0, 200)}`);
    }
    const d = await res.json();
    return {
      reply: d.choices?.[0]?.message?.content ?? '',
      tokensUsed: d.usage?.total_tokens ?? 0,
      provider: this.name,
    };
  }
}

class AnthropicProvider implements ChatProvider {
  readonly name = 'anthropic';
  private readonly key = process.env.USDCOP_ANTHROPIC_API_KEY ?? '';
  private readonly model = process.env.USDCOP_ANTHROPIC_MODEL || 'claude-sonnet-4-5-20250514';

  get configured(): boolean {
    return this.key.length > 10;
  }

  async complete(system: string, user: string, opts: ChatCompletionOptions = {}): Promise<ChatResult> {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.key,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
        system,
        messages: [{ role: 'user', content: user }],
      }),
    });
    if (!res.ok) {
      throw new Error(`anthropic ${res.status}: ${(await res.text()).slice(0, 200)}`);
    }
    const d = await res.json();
    return {
      reply: d.content?.[0]?.text ?? '',
      tokensUsed: (d.usage?.input_tokens ?? 0) + (d.usage?.output_tokens ?? 0),
      provider: this.name,
    };
  }
}

/** Configured providers in priority order (primary first). Empty ⇒ no LLM available. */
export function getChatProviders(): ChatProvider[] {
  return [new AzureOpenAIProvider(), new AnthropicProvider()].filter((p) => p.configured);
}

/**
 * Try each configured provider in order; return the first successful completion.
 * Returns null when no provider is configured OR all fail (caller renders a
 * graceful placeholder). Never throws.
 */
export async function completeWithFallback(
  system: string,
  user: string,
  opts?: ChatCompletionOptions,
): Promise<ChatResult | null> {
  const providers = getChatProviders();
  for (const p of providers) {
    try {
      const r = await p.complete(system, user, opts);
      if (r.reply.trim()) return r;
    } catch (err) {
      console.error(`[chat] provider ${p.name} failed:`, (err as Error).message);
    }
  }
  return null;
}

/** True when at least one provider is configured (used to choose placeholder vs live). */
export function hasChatProvider(): boolean {
  return getChatProviders().length > 0;
}
