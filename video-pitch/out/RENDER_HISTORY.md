# Render History — Global Minds Pitch

## Render Session · 2026-04-16

| Video | File | Resolution | Duration | Size | Render Time | MD5 |
|-------|------|------------|----------|------|-------------|-----|
| Raw (handoff) | `pitch-raw.mp4` | 960×540 · 30fps | 75.00s | 6.03 MB | 18m45s | `4e84b24ff74f402a877e1152134c4941` |
| Final (presentation) | `pitch-final.mp4` | 960×540 · 30fps | 75.00s | 5.99 MB | 15m38s | `04ea484582de5f8c38820cd53a96831f` |
| Vertical (LinkedIn) | `pitch-final-vertical.mp4` | 540×960 · 30fps | 75.00s | 6.70 MB | 12m58s | `c76b340fab8d3ed6efa5ba6e457c79c0` |

**Total render time**: 47m 21s

### Codec & audio

- Video: H.264 (libx264) · CRF 17 · `yuv420p` pixel format
- Audio: AAC 48kHz stereo · ~640kbps
- Container: MP4 (ISO 14496-12)

### Scale decision

Rendered at `--scale=0.5` (960×540 / 540×960) due to system load averaging 25-27
during the render window (competing CPU from Airflow, Postgres, Signalbridge,
MLflow, and dashboard containers). A full 1080p re-render is recommended when
the system is idle; estimated time ~35 min per video (~2 hours total).

To re-render at 1080p:
```bash
npx remotion render PitchRaw out/pitch-raw-1080p.mp4 --concurrency=3
npx remotion render PitchFinal out/pitch-final-1080p.mp4 --concurrency=3
npx remotion render PitchFinalVertical out/pitch-final-vertical-1080p.mp4 --concurrency=3
```

### Sanity validation

Frame snapshots extracted and visually verified in `out/previews/`:

| Frame | Time | Scene | File |
|-------|------|-------|------|
| raw-f660-snap | 22s | P03 Dashboard | OOS metrics + backtest replay overlay |
| raw-s33-P04 | 33s | P04 Producción | "13/14 semanas bloqueadas" + WeekBlockingBars |
| raw-s45-P05 | 45s | P05 Forecasting | Model Zoo · 9 · 63 · 7 + 9 ML pills |
| raw-s63-P07 | 63s | P07 SignalBridge | 37 DAGs · infra stats |
| raw-s72-P08 | 72s | P08 Outro | Disclaimer + brand reveal |
| final-s22-P03 | 22s | P03 | Same as raw (no RAW label correctly) |
| vertical-s22-P03 | 22s | P03 (vertical) | Backtest OOS 2025 + Sharpe 3.35, PF 2.76, WR 82% |

### Audio verification

```
Packet stream: 846 bytes/packet at ~21ms intervals (continuous)
Sample rate: 48kHz stereo
Duration: 75.05s (matches video)
Source: src/components/AudioTrack.tsx → BackgroundMusic + SFXCue
Music: procedural Am→F→C→G pad (75s stereo)
SFX cues: 23 discrete events (whoosh, tick×3, pop×14, impact×3, riser)
```

Audio confirmed present and active. Not silent padding.

### Known notes

- Airflow DAG clip (I04) was re-captured during Fase 3 because first attempt hit
  a 404 on `/dags?tags=forecast_h5`. Second capture uses `/home` with search
  filter — working correctly. Visible in P07 SignalBridge scene.
- P08 brand reveal timing: primary "Global Minds" gradient renders faintly
  against dark background at 960x540 due to subtle cyan-purple gradient. At
  1080p re-render, should be more visually pronounced.
- `--scale=0.5` output is suitable for: LinkedIn, Twitter, Discord, email
  embeds. For projector / 4K TV playback, re-render at full 1080p.

### Git commit SHA at render time

```
3675d73 fix: CI pipeline — black formatting, gitleaks v8 config, PYTHONPATH for contracts
9f0b0a3 feat: Smart Simple v2.0 — regime gate, effective HS, dynamic leverage, watchdog
```

All metrics in the videos are sourced from the dashboard snapshots at this commit.
