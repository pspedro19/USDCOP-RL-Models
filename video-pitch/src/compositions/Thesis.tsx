import React from "react";
import { z } from "zod";
import { AbsoluteFill, Sequence } from "remotion";
import { TOKENS } from "../theme/tokens";
import {
  THESIS_TIMINGS,
  THESIS_SCENE_ORDER,
  thesisSceneStart,
  THESIS_TOTAL_FRAMES,
} from "../data/timings-thesis";

// Import all 36 scenes (stubs for now, filled progressively)
import { T01_ColdOpen } from "../scenes/thesis/T01_ColdOpen";
import { T02_Caratula } from "../scenes/thesis/T02_Caratula";
import { T03_ContextoMacro } from "../scenes/thesis/T03_ContextoMacro";
import { T04_StateOfArt } from "../scenes/thesis/T04_StateOfArt";
import { T05_Hipotesis } from "../scenes/thesis/T05_Hipotesis";
import { T06_Breathing1 } from "../scenes/thesis/T06_Breathing1";
import { T07_OverviewArch } from "../scenes/thesis/T07_OverviewArch";
import { T08_TourL0toL8 } from "../scenes/thesis/T08_TourL0toL8";
import { T09_SystemOps } from "../scenes/thesis/T09_SystemOps";
import { T10_Breathing2 } from "../scenes/thesis/T10_Breathing2";
import { T11_AnalogiaSemaforo } from "../scenes/thesis/T11_AnalogiaSemaforo";
import { T12_ComponenteDA } from "../scenes/thesis/T12_ComponenteDA";
import { T13_ComponenteRegimen } from "../scenes/thesis/T13_ComponenteRegimen";
import { T14_ComponenteStops } from "../scenes/thesis/T14_ComponenteStops";
import { T15_ClimaxReplay } from "../scenes/thesis/T15_ClimaxReplay";
import { T16_PostClimax } from "../scenes/thesis/T16_PostClimax";
import { T17_AblationCorrected } from "../scenes/thesis/T17_AblationCorrected";
import { T18_SharpeCorrections } from "../scenes/thesis/T18_SharpeCorrections";
import { T19_BootstrapBlock } from "../scenes/thesis/T19_BootstrapBlock";
import { T20_Produccion2026 } from "../scenes/thesis/T20_Produccion2026";
import { T21_IntroCodigoDags } from "../scenes/thesis/T21_IntroCodigoDags";
import { T22_L0_Ingesta } from "../scenes/thesis/T22_L0_Ingesta";
import { T23_L3L5_Modelado } from "../scenes/thesis/T23_L3L5_Modelado";
import { T24_L7_Execution } from "../scenes/thesis/T24_L7_Execution";
import { T25_L6_Monitoring } from "../scenes/thesis/T25_L6_Monitoring";
import { T26_L8_Intelligence } from "../scenes/thesis/T26_L8_Intelligence";
import { T27_Watchdog } from "../scenes/thesis/T27_Watchdog";
import { T28_PgAdmin } from "../scenes/thesis/T28_PgAdmin";
import { T29_Microservicios } from "../scenes/thesis/T29_Microservicios";
import { T30_Breathing3 } from "../scenes/thesis/T30_Breathing3";
import { T31_Limitaciones } from "../scenes/thesis/T31_Limitaciones";
import { T32_Contribuciones } from "../scenes/thesis/T32_Contribuciones";
import { T33_Conclusiones } from "../scenes/thesis/T33_Conclusiones";
import { T34_TrabajoFuturo } from "../scenes/thesis/T34_TrabajoFuturo";
import { T35_Callback } from "../scenes/thesis/T35_Callback";
import { T36_FadeMusical } from "../scenes/thesis/T36_FadeMusical";

import { BackgroundMusic, SFXCue, SFX } from "../components/AudioTrack";

export const thesisSchema = z.object({
  variant: z.enum(["raw", "final"]).default("final"),
});

export type ThesisProps = z.infer<typeof thesisSchema>;

const SCENE_MAP = {
  T01_ColdOpen, T02_Caratula, T03_ContextoMacro, T04_StateOfArt,
  T05_Hipotesis, T06_Breathing1,
  T07_OverviewArch, T08_TourL0toL8, T09_SystemOps, T10_Breathing2,
  T11_AnalogiaSemaforo, T12_ComponenteDA, T13_ComponenteRegimen,
  T14_ComponenteStops, T15_ClimaxReplay, T16_PostClimax,
  T17_AblationCorrected, T18_SharpeCorrections, T19_BootstrapBlock,
  T20_Produccion2026,
  T21_IntroCodigoDags, T22_L0_Ingesta, T23_L3L5_Modelado,
  T24_L7_Execution, T25_L6_Monitoring, T26_L8_Intelligence,
  T27_Watchdog, T28_PgAdmin,
  T29_Microservicios, T30_Breathing3,
  T31_Limitaciones, T32_Contribuciones, T33_Conclusiones,
  T34_TrabajoFuturo, T35_Callback, T36_FadeMusical,
} as const;

/** Breathing scenes where music ducks */
const DUCK_FRAMES: number[] = [
  thesisSceneStart("T06_Breathing1"),
  thesisSceneStart("T10_Breathing2"),
  thesisSceneStart("T16_PostClimax"),
  thesisSceneStart("T30_Breathing3"),
];

/** SFX cues: whoosh between acts, impact on climax, ticks on metrics */
const SFX_CUES: Array<{ src: string; atFrame: number; volume?: number; dur?: number }> = [
  // Act transitions: whoosh
  { src: SFX.whoosh, atFrame: thesisSceneStart("T07_OverviewArch") - 10, volume: 0.55, dur: 30 },
  { src: SFX.whoosh, atFrame: thesisSceneStart("T11_AnalogiaSemaforo") - 10, volume: 0.6, dur: 30 },
  { src: SFX.whoosh, atFrame: thesisSceneStart("T17_AblationCorrected") - 10, volume: 0.6, dur: 30 },
  { src: SFX.whoosh, atFrame: thesisSceneStart("T21_IntroCodigoDags") - 10, volume: 0.6, dur: 30 },
  { src: SFX.whoosh, atFrame: thesisSceneStart("T29_Microservicios") - 10, volume: 0.55, dur: 30 },
  { src: SFX.whoosh, atFrame: thesisSceneStart("T31_Limitaciones") - 10, volume: 0.55, dur: 30 },
  // Climax: riser + impact on +25.63% reveal
  { src: SFX.riser, atFrame: thesisSceneStart("T15_ClimaxReplay") + 1800, volume: 0.55, dur: 45 },
  { src: SFX.impact, atFrame: thesisSceneStart("T15_ClimaxReplay") + 1850, volume: 0.85, dur: 60 },
  { src: SFX.numberTick, atFrame: thesisSceneStart("T15_ClimaxReplay") + 1900, volume: 0.5, dur: 10 },
  { src: SFX.numberTick, atFrame: thesisSceneStart("T15_ClimaxReplay") + 1950, volume: 0.5, dur: 10 },
  { src: SFX.numberTick, atFrame: thesisSceneStart("T15_ClimaxReplay") + 2000, volume: 0.5, dur: 10 },
  // Ablation: impact on final row reveal
  { src: SFX.impact, atFrame: thesisSceneStart("T17_AblationCorrected") + 1400, volume: 0.5, dur: 60 },
  // Outro logo
  { src: SFX.impact, atFrame: thesisSceneStart("T35_Callback") + 5, volume: 0.4, dur: 60 },
];

export const Thesis: React.FC<ThesisProps> = ({ variant }) => {
  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.primary }}>
      {THESIS_SCENE_ORDER.map((id) => {
        const Scene = SCENE_MAP[id];
        const from = thesisSceneStart(id);
        const dur = THESIS_TIMINGS[id];
        return (
          <Sequence
            key={id}
            from={from}
            durationInFrames={dur}
            layout="none"
            name={id}
          >
            <Scene variant={variant} />
          </Sequence>
        );
      })}

      {variant === "final" && (
        <>
          <BackgroundMusic
            baseVolume={0.28}
            duckAt={DUCK_FRAMES}
            totalFrames={THESIS_TOTAL_FRAMES}
          />
          {SFX_CUES.map((cue, i) => (
            <SFXCue
              key={i}
              src={cue.src}
              atFrame={Math.max(0, cue.atFrame)}
              volume={cue.volume ?? 0.6}
              durationInFrames={cue.dur ?? 60}
            />
          ))}
        </>
      )}

      {variant === "raw" && <RawSceneMarker />}
    </AbsoluteFill>
  );
};

const RawSceneMarker: React.FC = () => (
  <div
    style={{
      position: "absolute",
      top: 12,
      left: 12,
      padding: "6px 12px",
      background: "rgba(0,0,0,0.55)",
      border: `1px solid ${TOKENS.colors.accent.cyan}`,
      borderRadius: 6,
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 14,
      color: TOKENS.colors.accent.cyan,
      letterSpacing: 1,
      pointerEvents: "none",
    }}
  >
    THESIS RAW · handoff cut
  </div>
);
