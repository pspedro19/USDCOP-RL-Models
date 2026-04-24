# Project: SignalBridge / Global Minds Pitch Video

> Subproject of USDCOP-RL-Models. Produces a 75-second pitch video showcasing
> the trading system using a hybrid Playwright + Remotion workflow.

---

## Mission

Deliver two polished 75s pitch MP4s (+ bonus vertical for LinkedIn) that
demonstrate the SignalBridge / Global Minds trading system to investors as a
**guided tour** through the 6 sections of the dashboard (Inicio → Dashboard →
Producción → Forecasting → Análisis → SignalBridge).

Narrative pillars:
1. **2025 OOS** = out-of-sample validation — real proof the system generalizes
2. **2026 YTD** = live production — regime gate blocking mean-reverting weeks
3. **Backtest replay** = market reproduced bar-by-bar with system decisions

Real dashboard captures (Playwright) + animated overlays (Remotion) + kinetic
typography + royalty-free music + SFX. **No voiceover.**

---

## Hard constraints (FORBIDDEN to violate)

1. **All animations MUST use `useCurrentFrame()` + `interpolate` or `spring`.**
   No CSS `transition`, no CSS `@keyframes`, no `framer-motion animate` auto,
   no `useFrame` from `@react-three/fiber`.
2. **All colors/fonts/sizes come from `src/theme/tokens.ts`.** Never hardcode.
3. **All pitch metrics come from `src/data/metrics.ts` → `PITCH_METRICS`.**
   Do not hardcode +25.63 or 3.347 in scenes — reference the SSOT.
4. **Screencasts captured at 4K/60fps by Playwright, composition at 1080p/30fps.**
   Always use `<OffthreadVideo>` for screencast playback (not `<Video>`).
5. **Every scene accepts `variant: 'raw' | 'final'` prop.**
   - `raw`: no music, no SFX, scene markers visible, for designer handoff
   - `final`: full animation with music, SFX, captions
6. **Total pitch duration = sum of `PITCH_TIMINGS`.** Never hardcode `durationInFrames={2250}`.
7. **Safe zones (minimum):**
   - Landscape (1920×1080): top/bottom 60px, sides 80px
   - Vertical (1080×1920): top 150px, bottom 170px, sides 60px
8. **Minimum font size: 28px** (`TOKENS.fontSize.caption`). No exceptions.
9. **Pitch deck is 75 seconds total.** Extending requires updating SSOT + guión in one commit.
10. **No voice track.** Narrative via kinetic typography + SFX cues. Music optional ducked.

---

## Asset conventions

```
public/
├── captures/                # Playwright MP4s (4K/60fps → re-encoded 1080p/30fps CFR)
│   ├── S01-login.mp4
│   ├── S02-hub.mp4
│   ├── S03-dashboard-scroll.mp4
│   ├── S03-replay.mp4          <- BACKTEST REPLAY (market reproduced bar by bar)
│   ├── S05-production-live.mp4
│   ├── S06-analysis-chat.mp4
│   ├── S07-forecasting-zoo.mp4
│   ├── S08-signalbridge.mp4
│   └── I04-airflow-dags.mp4
├── audio/
│   ├── music/
│   │   └── pitch-upbeat.mp3    <- Pixabay CC0 cinematic tech 120-130 BPM
│   └── sfx/                    <- Mixkit/Pixabay CC0
│       ├── whoosh.mp3
│       ├── typewriter-tick.mp3
│       ├── number-tick.mp3
│       ├── impact-boom.mp3
│       ├── notification-pop.mp3
│       └── subtle-riser.mp3
└── data/                       <- Snapshots of summary.json, summary_2025.json for import
```

---

## Brand tokens (from dashboard)

All in `src/theme/tokens.ts`:
- Background: `#050816` (deep space) / `#0A0E27` / `#030712`
- Accent firma: `linear-gradient(135deg, #06B6D4, #8B5CF6)` (cyan → purple)
- Market: `#00D395` (up) / `#FF3B69` (down)
- Semantic: `#22c55e` (success) / `#f59e0b` (warning)
- Fonts: Inter (display+body), JetBrains Mono (numbers/code)
- Style: glassmorphism, dark mode, Bloomberg-terminal aesthetic

---

## Scene inventory — Guided Tour (8 scenes, 75s)

| ID | Dur | Section | Playwright clip | Overlay focus |
|----|-----|---------|-----------------|---------------|
| **P01_Hook** | 5s | — | — | Split reveal: "Febrero 2026. El mercado USD/COP cambió." + equity teaser |
| **P02_Inicio** | 8s | `/login` → `/hub` | S01 + S02 | Typewriter "Trading cuantitativo end-to-end" + 6 módulos staggered |
| **P03_Dashboard** | 15s | `/dashboard` | S03 + **S03-replay** | OOSBadge "Validación Out-of-Sample 2025" + backtest replay + count-up +25.63%/Sharpe 3.35/p=0.006 |
| **P04_Produccion** | 12s | `/production` | S05 | RegimeBadge "Mean-Reverting H=0.28" + WeekBlockingBars 13/14 + +0.61% vs BH -2.82% |
| **P05_Forecasting** | 9s | `/forecasting` | S07 | "9 modelos ML · 63 backtests walk-forward" + model zoo ranking |
| **P06_Analisis** | 9s | `/analysis` | S06 | "IA semanal · GPT-4o · 16 semanas" + chat flotante abierto |
| **P07_SignalBridge** | 12s | `/execution/dashboard` | S08 + I04 | "Ejecución automatizada · MEXC · AES-256 · Kill switch" + Airflow DAGs |
| **P08_Outro** | 5s | — | — | BrandLogoReveal "Global Minds · SignalBridge" + disclaimer + CTA |

**Total: 75s ✓.** Authoritative source: `src/data/timings-pitch.ts`.

---

## Authoritative metrics (DO NOT hardcode, import from PITCH_METRICS)

From `src/data/metrics.ts`:

### 2025 OOS (validation — out-of-sample)
- Return: **+25.63%**
- Sharpe: **3.347**
- p-value: **0.0063**
- MaxDD: **-6.12%**
- Trades: **34** (5 LONG / 29 SHORT)
- Win rate: **82.4%**
- TP exits: **21 (62%)** · HS exits: **2**

### 2026 YTD (live production)
- Return: **+0.61%** ($10K → $10,061)
- Trades: **1** (0 losses)
- Gate blocked: **13 of 14 weeks** (mean-reverting regime)
- Buy-and-hold: **-2.82%** · Alpha: **+3.43pp**

### Regime gate evidence
- Hurst 2025 (trending): **0.532**
- Hurst 2026 Q1 (mean-reverting): **0.28**
- Threshold mean-reverting: **<0.42** → block
- Threshold trending: **>0.52** → trade

### Infrastructure
- **37** Airflow DAGs
- **25+** Docker services
- **9** ML models (Ridge, BR, ARD, 3 pure, 3 hybrid boosting)
- **63** walk-forward backtests (9 models × 7 horizons)
- **43** DB migrations applied
- **8** dashboard pages (NextAuth + real-time polling)

Source files (paths in `metrics.ts`):
- `../usdcop-trading-dashboard/public/data/production/summary_2025.json`
- `../usdcop-trading-dashboard/public/data/production/summary.json`
- `../usdcop-trading-dashboard/public/data/production/approval_state.json`

---

## Disclaimer (mandatory on P08, ≥3s at ≥28px)

> SignalBridge / Global Minds es una herramienta de análisis cuantitativo.
> Los resultados mostrados corresponden a ejecuciones reales en el período
> 2025-2026 pero no garantizan rendimiento futuro. No constituye asesoría
> financiera. Trading conlleva riesgo de pérdida del capital invertido.

---

## Render commands

```bash
npm run still:sanity                  # quick frame validation
npm run dev                           # Remotion Studio interactive

npm run render:pitch-raw              # pitch-raw-1080p.mp4
npm run render:pitch-final            # pitch-final-1080p.mp4
npm run render:pitch-vertical         # pitch-final-vertical 1080x1920
```

---

## Do NOT

- Do NOT use `<Video>` — use `<OffthreadVideo>`
- Do NOT use `<img>` — use `<Img src={staticFile(...)}>`
- Do NOT fetch fonts via CSS — use `@remotion/google-fonts`
- Do NOT use `setInterval` / `setTimeout` — use frame-based logic
- Do NOT hardcode metrics in scenes — import from `PITCH_METRICS`
- Do NOT create `PitchRaw.tsx` / `PitchFinal.tsx` as separate files — one
  `Pitch.tsx` composition with `variant` prop (prevents content drift)
- Do NOT skip disclaimer in P08 — legal requirement
- Do NOT exceed 75s without updating SSOT in a single commit
- Do NOT add voiceover — explicit scope decision (kinetic typography only)
