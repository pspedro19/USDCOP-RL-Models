# Handoff — Pitch Video Global Minds

> Quick reference for reviewing the delivered videos and understanding what's in each.

---

## Deliverables

Located in `video-pitch/out/`:

| File | Dimensions | Duration | Audio | Purpose |
|------|-----------|----------|-------|---------|
| `pitch-raw-1080p.mp4` | 1920×1080 | 75s | Silent | Handoff / designer feedback (shows structure without SFX distraction) |
| `pitch-final-1080p.mp4` | 1920×1080 | 75s | Music + SFX | Final — presentation to Global Minds |
| `pitch-final-vertical.mp4` | 1080×1920 | 75s | Music + SFX | LinkedIn / mobile social |

---

## Narrative — guided tour (8 scenes)

1. **Hook** (5s): "Febrero 2026. El mercado USD/COP cambió." + equity teaser
2. **Inicio** (8s): Login → Hub revealing 6 módulos
3. **Dashboard** (15s) ⭐ **money shot**: OOS 2025 + backtest replay + `+25.63% · Sharpe 3.35 · p=0.006`
4. **Producción** (12s): Mean-reverting regime → 13 de 14 semanas bloqueadas · +0.61% vs B&H -2.82%
5. **Forecasting** (9s): 9 modelos ML · 63 backtests walk-forward · pills staggered
6. **Análisis** (9s): IA semanal GPT-4o · 16 semanas · 128 gráficos
7. **SignalBridge** (12s): Ejecución · Airflow 37 DAGs · Kill Switch · AES-256
8. **Outro** (5s): Global Minds · SignalBridge logo + disclaimer + CTA

All numbers auditable via `src/data/metrics.ts` → paths a `summary_2025.json` + `summary.json`.

---

## How to view

```bash
# Desktop (any player)
xdg-open out/pitch-final-1080p.mp4            # Linux
open out/pitch-final-1080p.mp4                 # macOS
# or drag into VLC, Chrome, QuickTime
```

---

## What's in the pipeline

### Playwright captures (real dashboard footage)
9 WebM clips in `public/captures/`:
- S01-login, S02-hub, S03-dashboard-scroll, S03-replay (⭐ backtest replay bar-a-bar)
- S05-production-live, S06-analysis-chat, S07-forecasting-zoo, S08-signalbridge
- I04-airflow-dags

**Captured from** `localhost:5000` + `localhost:8080` via headless Chromium scripted in
`scripts/capture-dashboard.ts`.

### Procedural audio (100% synthesized)
- `public/audio/music/pitch-ambient.wav` — 75s stereo chord progression (Am → F → C → G)
- `public/audio/sfx/*.wav` — 6 SFX (whoosh, tick, pop, impact, riser, notification)

Generated via `scripts/generate-audio.ts` using pure PCM math. **Zero external
dependencies, zero licensing ambiguity.**

---

## Re-render from scratch

```bash
cd video-pitch

# 1. Re-capture dashboard (requires stack on :5000 + :8080)
npx tsx scripts/capture-dashboard.ts

# 2. Regenerate audio
npx tsx scripts/generate-audio.ts

# 3. Render all three
npm run render:pitch-raw
npm run render:pitch-final
npm run render:pitch-vertical

# Or chained
npm run render:all
```

**Render time** (on 8-core laptop, concurrency=2 with system load):
- Raw 1080p: ~30 min
- Final 1080p: ~30 min
- Vertical 1080×1920: ~30 min

---

## Key design decisions

1. **No voiceover** — narrative carried by kinetic typography + SFX cues
2. **Real dashboard captures** — Playwright records the actual UI (not mocks), so the pitch
   is credible and reproducible
3. **2025 OOS = validation**, emphasized via the OOS badge and backtest replay scene
4. **Regime gate = MVP** — 13/14 blocked weeks is the narrative hook for Q1 2026
5. **Variant prop** (`raw` | `final`) — single composition renders both versions, avoiding
   content drift between designer cut and final cut
6. **SSOT for metrics** — `src/data/metrics.ts` is the only place numbers are defined
7. **SSOT for timings** — `src/data/timings-pitch.ts` sums to 75s; no hardcoded durations
8. **Dashboard colors** — `src/theme/tokens.ts` imports the exact hex values from the
   live dashboard (`#06B6D4` cyan, `#8B5CF6` purple, `#00D395` market up, etc.)

---

## Files to inspect

| File | What it does |
|------|--------------|
| `src/Root.tsx` | Registers 3 compositions |
| `src/compositions/Pitch.tsx` | Single composition + variant prop + SFX_CUES |
| `src/scenes/pitch/P01_Hook.tsx` ... `P08_Outro.tsx` | 8 scenes |
| `src/data/metrics.ts` | **SSOT audited numbers** |
| `src/data/timings-pitch.ts` | **SSOT scene durations** |
| `src/theme/tokens.ts` | Brand tokens from dashboard |
| `src/components/AudioTrack.tsx` | BackgroundMusic + SFXCue + SFX registry |
| `00-SCRIPTS/pitch-script.md` | Full guion with frame-level cue sheet |
| `scripts/capture-dashboard.ts` | Playwright capture script |
| `scripts/generate-audio.ts` | Procedural audio generator |

---

## QA checklist (verified before delivery)

- [x] 8 scenes registered in `Root.tsx` (3 compositions: Raw, Final, Vertical)
- [x] All 9 Playwright captures present in `public/captures/`
- [x] Audio files valid WAV (16-bit PCM, 44.1kHz)
- [x] `metrics.ts` SSOT — 0 hardcoded numbers in scenes
- [x] Dashboard colors match exactly (`#06B6D4` cyan, `#8B5CF6` purple)
- [x] Disclaimer in P08 ≥3s at ≥20px
- [x] Safe zones respected (60px top/bottom, 80px sides landscape)
- [x] Minimum font size 28px for caption text
- [x] No CSS animations, no `setInterval` — all via `useCurrentFrame` + `interpolate` / `spring`
- [x] `OffthreadVideo` used for all video playback (not `<Video>`)

---

## Known notes

- Render times are CPU-bound on this machine; a dedicated render box or Remotion Lambda
  would cut total wall-clock by 5-10×
- Audio is procedurally generated → the ambient pad has a "pure tone" character, not a
  commercial music feel. If you want a commercial track, drop a Pixabay/YouTube CC0 MP3
  into `public/audio/music/` named `pitch-ambient.wav` (or update path in `AudioTrack.tsx`)
- The Airflow clip (I04) was re-captured after an initial misfire (404 on `/dags?tags=...`)
  to use `/home` instead. If rendering older footage you'll see Airflow 404 page — just
  re-run `npx tsx scripts/capture-dashboard.ts I04` to refresh

---

## Contact / attribution

- Video production: hybrid Playwright + Remotion pipeline
- All trademarks/logos shown are owned by their respective holders
- Dashboard colors licensed internally by the project (see `usdcop-trading-dashboard`)
