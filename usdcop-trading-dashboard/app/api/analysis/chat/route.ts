/**
 * POST /api/analysis/chat — grounded weekly-analysis chatbot (CTR-CHAT-001).
 *
 * Composition root (thin): resolve user → per-plan quota (monetization) → build
 * grounded context → complete via the provider abstraction (Azure→Anthropic
 * fallback). All real logic lives in lib/chat/* (DIP/SRP). Gated by middleware
 * (analysis:read); the quota is the monetization lever on top.
 *
 * Request:  { message, session_id, year, week, asset? }
 * Response: { reply, tokens_used, session_id, quota_remaining, provider }
 */

import { NextRequest, NextResponse } from 'next/server';

import { readAnalysisJson } from '@/lib/analysis-paths';
import { getCurrentUser } from '@/lib/auth/api-auth';
import { getEntitlements } from '@/lib/auth/entitlements';
import { getAnalysisAsset } from '@/lib/contracts/analysis-assets';
import { buildSystemPrompt, buildWeekContext } from '@/lib/chat/context';
import { completeWithFallback, hasChatProvider } from '@/lib/chat/providers';
import { chatQuotaFor, consumeQuota } from '@/lib/chat/quota';

const MAX_RESPONSE_TOKENS = 2000;

interface ChatRequest {
  message: string;
  session_id: string;
  year: number;
  week: number;
  asset?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequest = await request.json();
    if (!body.message?.trim()) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    // Identity + monetization: per-plan daily quota, enforced server-side.
    const user = await getCurrentUser();
    const userId = user?.id ?? 'anonymous';
    const entitlements = await getEntitlements(user?.id);
    const quota = chatQuotaFor(entitlements);
    const state = consumeQuota(userId, quota);
    if (!state.allowed) {
      return NextResponse.json(
        {
          error: `Límite diario del chat alcanzado (${quota.dailyLimit} en el plan ${quota.plan}). Mejora tu plan para más consultas.`,
          quota_remaining: 0,
          plan: quota.plan,
        },
        { status: 429 },
      );
    }

    // Grounded context from the selected week.
    const asset = getAnalysisAsset(body.asset);
    const weekStr = String(body.week).padStart(2, '0');
    const weekData = await readAnalysisJson<any>(asset.asset_id, `weekly_${body.year}_W${weekStr}.json`);
    const context = buildWeekContext(weekData, asset);
    const systemPrompt = buildSystemPrompt(asset, context);

    // No provider configured → helpful, context-filled placeholder (graceful).
    if (!hasChatProvider()) {
      return NextResponse.json({
        reply:
          `**Chat sin LLM configurado**\n\nPara habilitar respuestas del asistente, ` +
          `configura \`USDCOP_AZURE_OPENAI_API_KEY\` (+ endpoint/deployment) o ` +
          `\`USDCOP_ANTHROPIC_API_KEY\`.\n\n**Contexto de la semana ${body.year}-W${weekStr}:**\n${context}`,
        tokens_used: 0,
        session_id: body.session_id,
        quota_remaining: state.remaining,
        provider: 'none',
      });
    }

    const result = await completeWithFallback(systemPrompt, body.message.trim(), {
      maxTokens: MAX_RESPONSE_TOKENS,
    });

    return NextResponse.json({
      reply: result?.reply || 'Lo siento, no pude procesar tu mensaje en este momento. Intenta de nuevo más tarde.',
      tokens_used: result?.tokensUsed ?? 0,
      session_id: body.session_id,
      quota_remaining: state.remaining,
      provider: result?.provider ?? 'none',
    });
  } catch {
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
