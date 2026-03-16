/**
 * POST /api/analysis/chat
 * Sends a user message to the LLM with weekly context injection.
 *
 * Request body: { message: string, session_id: string, year: number, week: number }
 * Response: { reply: string, tokens_used: number, session_id: string }
 *
 * In dev mode (USDCOP_DEV_AUTH_BYPASS=true), no auth required.
 * Rate limit: 50 messages per session, 2000 max tokens per response.
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const MAX_MESSAGES_PER_SESSION = 50;
const MAX_RESPONSE_TOKENS = 2000;

// In-memory session message counters (resets on server restart)
const sessionCounters = new Map<string, number>();

interface ChatRequest {
  message: string;
  session_id: string;
  year: number;
  week: number;
}

interface ChatResponse {
  reply: string;
  tokens_used: number;
  session_id: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequest = await request.json();

    if (!body.message?.trim()) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    // Rate limiting
    const count = sessionCounters.get(body.session_id) ?? 0;
    if (count >= MAX_MESSAGES_PER_SESSION) {
      return NextResponse.json(
        { error: 'Session message limit reached (50). Start a new session.' },
        { status: 429 }
      );
    }
    sessionCounters.set(body.session_id, count + 1);

    // Load week context
    const weekStr = String(body.week).padStart(2, '0');
    const filename = `weekly_${body.year}_W${weekStr}.json`;
    const filepath = path.join(process.cwd(), 'public', 'data', 'analysis', filename);

    let contextSummary = 'No weekly context available.';
    try {
      const raw = await fs.readFile(filepath, 'utf-8');
      const weekData = JSON.parse(raw);
      // Build compact context from weekly data
      const parts: string[] = [];
      if (weekData.weekly_summary?.headline) {
        parts.push(`Headline: ${weekData.weekly_summary.headline}`);
      }
      if (weekData.weekly_summary?.sentiment) {
        parts.push(`Sentiment: ${weekData.weekly_summary.sentiment}`);
      }
      if (weekData.weekly_summary?.ohlcv) {
        const o = weekData.weekly_summary.ohlcv;
        parts.push(`USDCOP: Open ${o.open}, High ${o.high}, Low ${o.low}, Close ${o.close}, Change ${o.change_pct}%`);
      }
      if (weekData.signals?.h5?.direction) {
        parts.push(`H5 Signal: ${weekData.signals.h5.direction} (conf: ${weekData.signals.h5.confidence})`);
      }
      if (weekData.signals?.h1?.direction) {
        parts.push(`H1 Signal: ${weekData.signals.h1.direction}`);
      }
      if (weekData.news_context) {
        parts.push(`News: ${weekData.news_context.article_count} articles, avg sentiment ${weekData.news_context.avg_sentiment}`);
      }
      contextSummary = parts.join('\n');
    } catch {
      // Context file not found — proceed without
    }

    // Check for LLM API keys
    const azureKey = process.env.USDCOP_AZURE_OPENAI_API_KEY;
    const anthropicKey = process.env.USDCOP_ANTHROPIC_API_KEY;

    if (!azureKey && !anthropicKey) {
      // Return a helpful placeholder response when no LLM is configured
      const response: ChatResponse = {
        reply: `**Chat sin LLM configurado**\n\nPara habilitar el chat, configura \`USDCOP_AZURE_OPENAI_API_KEY\` o \`USDCOP_ANTHROPIC_API_KEY\` en las variables de entorno.\n\n**Contexto de la semana ${body.year}-W${weekStr}:**\n${contextSummary}`,
        tokens_used: 0,
        session_id: body.session_id,
      };
      return NextResponse.json(response);
    }

    // Try Azure OpenAI first, then Anthropic fallback
    let reply = '';
    let tokensUsed = 0;

    const systemPrompt = `Eres un asistente financiero del sistema de trading USDCOP. Tienes acceso al contexto de la semana actual incluyendo señales de modelos, indicadores macro, y noticias relevantes.\n\nReglas:\n- Responde en español\n- Se conciso (maximo 300 palabras por respuesta)\n- Basa tus respuestas en los datos proporcionados en el contexto\n- Si no tienes informacion suficiente, dilo\n- No des consejos de inversion directos — presenta datos y analisis\n\nContexto semanal:\n${contextSummary}`;

    if (azureKey) {
      try {
        const endpoint = process.env.USDCOP_AZURE_OPENAI_ENDPOINT!;
        const deployment = process.env.USDCOP_AZURE_OPENAI_DEPLOYMENT || 'gpt-4o';
        const apiVersion = process.env.USDCOP_AZURE_OPENAI_API_VERSION || '2024-10-21';
        const url = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'api-key': azureKey,
          },
          body: JSON.stringify({
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: body.message },
            ],
            max_tokens: MAX_RESPONSE_TOKENS,
            temperature: 0.7,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          reply = data.choices?.[0]?.message?.content || '';
          tokensUsed = data.usage?.total_tokens || 0;
        }
      } catch {
        // Fall through to Anthropic
      }
    }

    if (!reply && anthropicKey) {
      try {
        const res = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': anthropicKey,
            'anthropic-version': '2023-06-01',
          },
          body: JSON.stringify({
            model: 'claude-sonnet-4-5-20250514',
            max_tokens: MAX_RESPONSE_TOKENS,
            system: systemPrompt,
            messages: [{ role: 'user', content: body.message }],
          }),
        });

        if (res.ok) {
          const data = await res.json();
          reply = data.content?.[0]?.text || '';
          tokensUsed = (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0);
        }
      } catch {
        // Both providers failed
      }
    }

    if (!reply) {
      reply = 'Lo siento, no pude procesar tu mensaje en este momento. Intenta de nuevo mas tarde.';
    }

    const response: ChatResponse = {
      reply,
      tokens_used: tokensUsed,
      session_id: body.session_id,
    };

    return NextResponse.json(response);
  } catch {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
