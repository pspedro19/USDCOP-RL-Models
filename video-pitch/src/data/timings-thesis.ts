/**
 * THESIS TIMINGS — Single Source of Truth for 20-min academic defense video
 *
 * Structure: 6 actos, 36 escenas, 1200s (20 min).
 * Academic pacing: pausado, con breathing rooms obligatorios entre actos.
 * Central feature: Acto V tour de código Airflow por 10 DAGs (~4:40).
 */

export const FPS = 30;

export const THESIS_TIMINGS = {
  // ============ ACTO I · Apertura + paradoja (2:30 = 150s) ============
  T01_ColdOpen: 25 * FPS,           // 750f · "El modelo predijo mal, aún así ganó"
  T02_Caratula: 18 * FPS,           // 540f · Logo FIUBA + autor + director
  T03_ContextoMacro: 30 * FPS,      // 900f · USD/COP régimen cambiante
  T04_StateOfArt: 40 * FPS,         // 1200f · 5 citas Chicago
  T05_Hipotesis: 40 * FPS,          // 1200f · H-primaria + H1/H2/H3 + H-alt
  T06_Breathing1: 5 * FPS,          // 150f · black beat

  // ============ ACTO II · Arquitectura comprimida (2:30 = 150s) ============
  T07_OverviewArch: 30 * FPS,       // 900f · Slide 0 overview
  T08_TourL0toL8: 90 * FPS,         // 2700f · montage L0-L8 ritmado
  T09_SystemOps: 30 * FPS,          // 900f · Slide 7 watchdog radial
  T10_Breathing2: 5 * FPS,          // 150f

  // ============ ACTO III · 3 componentes del edge (5:30 = 330s) ============
  T11_AnalogiaSemaforo: 20 * FPS,   // 600f · metáfora niebla
  T12_ComponenteDA: 65 * FPS,       // 1950f · DA honesto, binomial tests
  T13_ComponenteRegimen: 75 * FPS,  // 2250f · Hurst + WeekBlocking + MVP
  T14_ComponenteStops: 50 * FPS,    // 1500f · effective HS + dyn leverage
  T15_ClimaxReplay: 90 * FPS,       // 2700f · Backtest replay 2025 (PEAK)
  T16_PostClimax: 8 * FPS,          // 240f · breathing post-pico

  // ============ ACTO IV · Validación estadística (3:15 = 195s) ============
  T17_AblationCorrected: 60 * FPS,  // 1800f · tabla 5 configs con BH-FDR
  T18_SharpeCorrections: 45 * FPS,  // 1350f · PSR + DSR
  T19_BootstrapBlock: 45 * FPS,     // 1350f · Politis-Romano
  T20_Produccion2026: 25 * FPS,     // 750f · YTD honesto (N=1)

  // ============ ACTO V · Deep tour código Airflow (4:40 = 280s) ⭐ ============
  T21_IntroCodigoDags: 15 * FPS,    // 450f · "Código reproducible"
  T22_L0_Ingesta: 45 * FPS,         // 1350f · 2 DAGs (ohlcv + macro)
  T23_L3L5_Modelado: 60 * FPS,      // 1800f · 3 DAGs (training + signal + régimen)
  T24_L7_Execution: 25 * FPS,       // 750f · 1 DAG (multiday_executor)
  T25_L6_Monitoring: 20 * FPS,      // 600f · 1 DAG (weekly_monitor)
  T26_L8_Intelligence: 45 * FPS,    // 1350f · 2 DAGs (news + analysis)
  T27_Watchdog: 20 * FPS,           // 600f · 1 DAG (core_watchdog)
  T28_PgAdmin: 30 * FPS,            // 900f · psql queries reales

  // ============ ACTO VI · Cierre (2:15 = 135s) ============
  T29_Microservicios: 35 * FPS,     // 1050f · Docker grid 5x5 + network
  T30_Breathing3: 5 * FPS,          // 150f
  T31_Limitaciones: 45 * FPS,       // 1350f · 7 bullets auto-crítica
  T32_Contribuciones: 40 * FPS,     // 1200f · 6 aportes originales
  T33_Conclusiones: 40 * FPS,       // 1200f · paradoja + sinergia + legado
  T34_TrabajoFuturo: 20 * FPS,      // 600f · 4 ejes
  T35_Callback: 25 * FPS,           // 750f · regreso al cold open + créditos
  T36_FadeMusical: 5 * FPS,         // 150f · fade out final
} as const;

export type ThesisSceneId = keyof typeof THESIS_TIMINGS;

export const THESIS_SCENE_ORDER: ThesisSceneId[] = [
  "T01_ColdOpen", "T02_Caratula", "T03_ContextoMacro", "T04_StateOfArt",
  "T05_Hipotesis", "T06_Breathing1",
  "T07_OverviewArch", "T08_TourL0toL8", "T09_SystemOps", "T10_Breathing2",
  "T11_AnalogiaSemaforo", "T12_ComponenteDA", "T13_ComponenteRegimen",
  "T14_ComponenteStops", "T15_ClimaxReplay", "T16_PostClimax",
  "T17_AblationCorrected", "T18_SharpeCorrections", "T19_BootstrapBlock",
  "T20_Produccion2026",
  "T21_IntroCodigoDags", "T22_L0_Ingesta", "T23_L3L5_Modelado",
  "T24_L7_Execution", "T25_L6_Monitoring", "T26_L8_Intelligence",
  "T27_Watchdog", "T28_PgAdmin",
  "T29_Microservicios", "T30_Breathing3",
  "T31_Limitaciones", "T32_Contribuciones", "T33_Conclusiones",
  "T34_TrabajoFuturo", "T35_Callback", "T36_FadeMusical",
];

export const THESIS_TOTAL_FRAMES = THESIS_SCENE_ORDER.reduce(
  (acc, id) => acc + THESIS_TIMINGS[id],
  0
);
// 36 scenes = 1200s × 30fps = 36,000 frames

export const thesisSceneStart = (id: ThesisSceneId): number => {
  const idx = THESIS_SCENE_ORDER.indexOf(id);
  return THESIS_SCENE_ORDER.slice(0, idx).reduce(
    (acc, k) => acc + THESIS_TIMINGS[k],
    0
  );
};

export const thesisSceneEnd = (id: ThesisSceneId): number =>
  thesisSceneStart(id) + THESIS_TIMINGS[id];

export const thesisSceneAtFrame = (frame: number): ThesisSceneId | undefined => {
  return THESIS_SCENE_ORDER.find((id) => {
    const s = thesisSceneStart(id);
    return frame >= s && frame < s + THESIS_TIMINGS[id];
  });
};

export const localFrameThesis = (frame: number, id: ThesisSceneId): number =>
  frame - thesisSceneStart(id);

/** DAG code tour timestamps within I04-airflow-code-tour.webm */
export const DAG_TOUR_TIMINGS = {
  login_and_list: { start: 0, duration: 12 },  // 0-12s
  core_l0_01_ohlcv: { start: 12, duration: 22 },
  core_l0_03_macro: { start: 34, duration: 22 },
  forecast_h5_l3: { start: 56, duration: 22 },
  forecast_h5_l5_signal: { start: 78, duration: 22 },
  forecast_h5_l5_regime: { start: 100, duration: 22 },  // MVP ⭐
  forecast_h5_l7: { start: 122, duration: 22 },
  forecast_h5_l6: { start: 144, duration: 22 },
  news_daily: { start: 166, duration: 22 },
  analysis_l8: { start: 188, duration: 22 },
  core_watchdog: { start: 210, duration: 22 },
  final_dwell: { start: 232, duration: 8 },
} as const;
