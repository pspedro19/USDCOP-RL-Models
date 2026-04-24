/**
 * PITCH TIMINGS — Single Source of Truth for scene durations
 *
 * v4 (2026-04-16 post-feedback): extended to 120s (2 min) for a more
 * natural, pausado pacing. P07 SignalBridge gets major dwell (35s) to
 * showcase Airflow deep tour: login → DAGs → click → Code → scroll code.
 *
 * Total: 120s = 3600 frames.
 */

export const FPS = 30;

export const PITCH_TIMINGS = {
  P01_Hook: 5 * FPS,              // 150f · "Una plataforma integral"
  P02_Inicio: 10 * FPS,           // 300f · Login + Hub pausado
  P03_Dashboard: 26 * FPS,        // 780f · Caso 1 · Backtest replay completo
  P04_Produccion: 12 * FPS,       // 360f · Caso 2 · Live + régimen gate
  P05_Forecasting: 14 * FPS,      // 420f · Caso 3 · Forward Forecast + Backtest multi-week
  P06_Analisis: 13 * FPS,         // 390f · Caso 4 · Análisis macro multi-semana
  P07_SignalBridge: 35 * FPS,     // 1050f · Caso 5 · Ejecución + Airflow deep + API
  P08_Outro: 5 * FPS,             // 150f · Logo + disclaimer + CTA
} as const;

export type PitchSceneId = keyof typeof PITCH_TIMINGS;

export const PITCH_SCENE_ORDER: PitchSceneId[] = [
  "P01_Hook",
  "P02_Inicio",
  "P03_Dashboard",
  "P04_Produccion",
  "P05_Forecasting",
  "P06_Analisis",
  "P07_SignalBridge",
  "P08_Outro",
];

export const PITCH_TOTAL_FRAMES = PITCH_SCENE_ORDER.reduce(
  (acc, id) => acc + PITCH_TIMINGS[id],
  0
);
// 150+300+780+360+420+390+1050+150 = 3600 frames = 120s ✓

export const pitchSceneStart = (id: PitchSceneId): number => {
  const idx = PITCH_SCENE_ORDER.indexOf(id);
  return PITCH_SCENE_ORDER.slice(0, idx).reduce(
    (acc, k) => acc + PITCH_TIMINGS[k],
    0
  );
};

export const pitchSceneEnd = (id: PitchSceneId): number =>
  pitchSceneStart(id) + PITCH_TIMINGS[id];

export const pitchSceneAtFrame = (frame: number): PitchSceneId | undefined => {
  return PITCH_SCENE_ORDER.find((id) => {
    const start = pitchSceneStart(id);
    const end = pitchSceneEnd(id);
    return frame >= start && frame < end;
  });
};

export const localFrame = (frame: number, id: PitchSceneId): number =>
  frame - pitchSceneStart(id);
