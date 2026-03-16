# SDD-09: Chat Widget

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-09 |
| **Título** | Floating Chat Widget — Contextual LLM Assistant |
| **Versión** | 1.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🆕 NEW |
| **Depende de** | SDD-03 (analysis_chat_history), SDD-07 (LLM client), SDD-08 (page) |

---

## 1. Scope

Context-aware conversational assistant, available **only on `/analysis`**. Uses the currently viewed week's analysis + NewsEngine data as LLM context.

---

## 2. Visual Design

**Collapsed:** Fixed `bottom-6 right-6`, circular `w-14 h-14` button, `bg-cyan-600`.

**Expanded:** `w-[420px] h-[560px]` glassmorphism panel. Full-screen on mobile `< 640px`.

```
┌──────────────────────────────────────────────┐
│  💬 Chat USD/COP          [Semana 9]  [─][✕] │
├──────────────────────────────────────────────┤
│                                              │
│  🤖 Hola, tengo contexto de la Semana 9     │
│     incluyendo macro, señales y noticias.    │
│                                              │
│       ┌──────────────────────────────┐       │
│       │ ¿Por qué el modelo dio SHORT │  👤   │
│       │ si el Brent subió el jueves? │       │
│       └──────────────────────────────┘       │
│                                              │
│  🤖 El DXY rompió resistencia en 104.50     │
│     (SMA-20: 103.80), y además el           │
│     NewsEngine detectó 3 fuentes reportando  │
│     "fortaleza dólar"...                     │
│                                              │
│  [¿Qué impactó más?] [Señales] [Variables]  │
│                                              │
├──────────────────────────────────────────────┤
│  [Escribe tu pregunta...]           [Enviar] │
└──────────────────────────────────────────────┘
```

---

## 3. Context Injection

The chat system prompt includes:
- Weekly analysis summary + daily headlines (from `weekly_analysis` + `daily_analysis`)
- Macro snapshots with SMAs (from `macro_variable_snapshots`)
- Model signals H1/H5 (from existing trading DB)
- **NewsEngine data**: article counts, top categories, cross-reference topics (from `articles` + `cross_references`)

Context auto-updates when user navigates to a different week.

---

## 4. Quick Actions

Pre-defined prompts that disappear after first user message:

```
"¿Qué impactó más al USD/COP esta semana?"
"Explica las señales del modelo"
"¿Qué variables macro debo vigilar?"
"Resumen para toma de decisiones"
```

---

## 5. Rate Limiting

- 50 messages/session, 10 sessions/day
- 1s cooldown between messages (client-side)
- 2,000 max tokens per response
- Message counter in header: "12/50"

---

## 6. Implementation

- **MVP:** HTTP POST to `/api/analysis/chat` (full response, no streaming)
- **V2:** WebSocket for token-by-token streaming
- **State:** Zustand store (`useAnalysisChatStore`)
- **Persistence:** `analysis_chat_history` table (SDD-03 B4)
- **Session:** UUID generated on open, persists until widget close
