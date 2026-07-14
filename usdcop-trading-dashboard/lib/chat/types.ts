/**
 * Chat provider abstraction (CTR-CHAT-001).
 *
 * DIP: the /api/analysis/chat route depends on this interface, not on any
 * concrete LLM vendor. Add a provider = one class implementing ChatProvider +
 * one line in the factory (OCP). Providers self-report `configured` from env,
 * so an unset key degrades gracefully instead of throwing at import time.
 */

export interface ChatCompletionOptions {
  maxTokens?: number;
  temperature?: number;
}

export interface ChatResult {
  reply: string;
  tokensUsed: number;
  provider: string;
}

export interface ChatProvider {
  /** Stable id for logging/telemetry. */
  readonly name: string;
  /** True only when the env this provider needs is present. */
  readonly configured: boolean;
  /** One-shot completion. Throws on a transport/HTTP error so the caller can fall back. */
  complete(system: string, user: string, opts?: ChatCompletionOptions): Promise<ChatResult>;
}
