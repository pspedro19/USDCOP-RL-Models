# SDD-08: Dashboard & Frontend

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-08 |
| **Título** | Dashboard Frontend — /analysis Page |
| **Versión** | 1.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🆕 NEW |
| **Depende de** | SDD-03, SDD-07 |
| **Requerido por** | SDD-09 (Chat Widget) |

---

## 1. Page Route & Component Hierarchy

**Route:** `/analysis` in existing Next.js dashboard

```
app/analysis/page.tsx
  └── components/analysis/
        AnalysisPage.tsx                      # State orchestrator
        ├── WeekSelector.tsx                  # ◄ W08 │ SEMANA 9 │ W10 ►
        ├── WeeklySummaryHeader.tsx            # Sentiment, themes, executive summary
        ├── MacroSnapshotBar.tsx               # 4 key variable cards (DXY, VIX, Oil, EMBI)
        ├── SignalSummaryCards.tsx              # H5 weekly + H1 daily aggregate
        ├── DailyTimeline.tsx                  # Vertical timeline container
        │   └── DailyTimelineEntry.tsx         # Per-day: dot, connector, content card
        │       └── MacroEventChip.tsx         # Colored publication badge
        ├── UpcomingEventsPanel.tsx            # Next week calendar
        ├── AnalysisMarkdown.tsx              # react-markdown wrapper
        └── FloatingChatWidget.tsx             # (SDD-09)
```

---

## 2. Visual Layout

```
+------------------------------------------------------------------+
| GlobalNavbar  [Hub] [Dashboard] [Forecasting] [*Análisis*]        |
+------------------------------------------------------------------+
| [< W08]  SEMANA 9  ·  Feb 23-27, 2026  [W10 >]  [▾ Jump]        |
+------------------------------------------------------------------+
| WEEKLY SUMMARY                                                     |
| [🔴 BEARISH COP]  [#DXY]  [#VIX]  [#BanRep]                     |
| "Peso colombiano se debilita ante fortaleza del dólar..."          |
| [Open: 4,201] [Close: 4,235 +0.82%] [High: 4,250] [Low: 4,198]  |
+------------------------------------------------------------------+
| MACRO SNAPSHOT BAR                                                 |
| [DXY 104.5 SMA:103.8 ▲+0.4%] [VIX 18.3 ▼-1.2%] [WTI] [EMBI]  |
+------------------------------------------------------------------+
| SIGNALS                                                            |
| [H5 Weekly: SHORT HIGH -0.8%]  [H1 Daily: 2L / 3S / 0H]         |
+------------------------------------------------------------------+
| DAILY TIMELINE (vertical)                                          |
|  ●── Mon 23  "DXY sube tras datos de empleo"                      |
|  │   [analysis card — expandable]                                  |
|  │   [CPI: 🔴] [Employment: 🔴]  H1: SHORT                      |
|  │                                                                 |
|  ●── Tue 24  "EMBI sube 15bp, presión sobre COP"                 |
|  │   [card]                                                        |
|  │                                                                 |
|  ●── Wed 25  "Oil cae 2%, COP encuentra soporte"                 |
|  │   [card]                                                        |
|  │                                                                 |
|  ●── Thu 26  "Fed minutes dovish, DXY retrocede"                 |
|  │   [card]                                                        |
|  │                                                                 |
|  ●── Fri 27  "Cierre semanal: COP -0.8%"                         |
|       [card + H5 result]                                           |
+------------------------------------------------------------------+
| UPCOMING EVENTS                                                    |
| Mar 2: Fed Funds [🔴] | Mar 3: CPI CO [🟡] | ...                |
+------------------------------------------------------------------+
|                                               [💬 Chat] ← Float  |
+------------------------------------------------------------------+
```

---

## 3. Design System

Follows existing **glassmorphism dark theme**:

| Element | Tailwind Class |
|---------|---------------|
| Page bg | `bg-gray-950` |
| Content cards | `bg-slate-900/40 backdrop-blur-md border border-gray-800/50 rounded-xl` |
| Active glow | `ring-2 ring-cyan-500/30` |
| Timeline line | `border-l-2 border-cyan-500/30` |
| Day dot (active) | `w-4 h-4 rounded-full bg-cyan-500 shadow-lg shadow-cyan-500/20` |
| Day dot (past) | `w-3 h-3 rounded-full` + sentiment color |
| Sentiment: bullish | `#10B981` (green-500) |
| Sentiment: bearish | `#EF4444` (red-500) |
| Sentiment: neutral | `#F59E0B` (amber-500) |
| Sentiment: volatile | `#8B5CF6` (purple-500) |
| MacroEvent chip (high) | `bg-red-500/20 text-red-400 border-red-500/30` |
| MacroEvent chip (medium) | `bg-amber-500/20 text-amber-400` |
| MacroEvent chip (low) | `bg-slate-500/20 text-slate-400` |

---

## 4. Data Fetching

**File:** `hooks/useWeeklyAnalysis.ts`

```typescript
// Reads static JSON files (exported by Analysis Engine SDD-07)
export function useWeekList()        // /api/analysis/weeks → analysis_index.json
export function useWeekView(y, w)    // /api/analysis/week/Y/W → weekly_Y_WXX.json
export function useUpcomingEvents()  // /api/analysis/calendar → upcoming_events.json
```

Fallback: Direct fetch from `/data/analysis/*.json` if API route fails.

---

## 5. API Routes

| Method | Path | Source | Response |
|--------|------|--------|----------|
| GET | `/api/analysis/weeks` | `analysis_index.json` | `WeekListResponse` |
| GET | `/api/analysis/week/[year]/[week]` | `weekly_Y_WXX.json` | `WeekView` |
| GET | `/api/analysis/calendar` | `upcoming_events.json` | `UpcomingEvent[]` |
| POST | `/api/analysis/chat` | Azure OpenAI / DB | `ChatResponse` |

---

## 6. TypeScript Contract

**File:** `lib/contracts/weekly-analysis.contract.ts`

Key interfaces: `WeeklyAnalysis`, `DailyAnalysis`, `MacroSnapshot`, `MacroPublication`, `WeekView`, `UpcomingEvent`, `ChatMessage`, `ChatRequest`, `ChatResponse`, `WeekListItem`, `WeekListResponse`.

---

## 7. Responsive Design

| Breakpoint | Adaptation |
|------------|-----------|
| `< 640px` | Single column. MacroBar 2×2. Chat full-screen. |
| `640–1024px` | MacroBar 2×2. Timeline centered max-width. |
| `> 1024px` | Full layout. MacroBar 4-col. |

---

## 8. New Files (19)

| # | File | Type |
|---|------|------|
| 1 | `app/analysis/page.tsx` | Page |
| 2 | `lib/contracts/weekly-analysis.contract.ts` | Contract |
| 3 | `hooks/useWeeklyAnalysis.ts` | Hook |
| 4-7 | `app/api/analysis/*/route.ts` | API (×4) |
| 8-18 | `components/analysis/*.tsx` | Components (×11) |
| 19 | `stores/useAnalysisChatStore.ts` | Zustand store |

Modified: `GlobalNavbar.tsx` (add nav item), `hub/page.tsx` (add card).
