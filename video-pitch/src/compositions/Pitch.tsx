import React from "react";
import { z } from "zod";
import { AbsoluteFill, Sequence } from "remotion";
import { TOKENS } from "../theme/tokens";
import {
  PITCH_TIMINGS,
  PITCH_SCENE_ORDER,
  pitchSceneStart,
  PITCH_TOTAL_FRAMES,
} from "../data/timings-pitch";

import { P01_Hook } from "../scenes/pitch/P01_Hook";
import { P02_Inicio } from "../scenes/pitch/P02_Inicio";
import { P03_Dashboard } from "../scenes/pitch/P03_Dashboard";
import { P04_Produccion } from "../scenes/pitch/P04_Produccion";
import { P05_Forecasting } from "../scenes/pitch/P05_Forecasting";
import { P06_Analisis } from "../scenes/pitch/P06_Analisis";
import { P07_SignalBridge } from "../scenes/pitch/P07_SignalBridge";
import { P08_Outro } from "../scenes/pitch/P08_Outro";
import { BackgroundMusic, SFXCue, SFX } from "../components/AudioTrack";

export const pitchSchema = z.object({
  variant: z.enum(["raw", "final"]).default("final"),
  vertical: z.boolean().default(false),
});

export type PitchProps = z.infer<typeof pitchSchema>;

const SCENE_MAP = {
  P01_Hook,
  P02_Inicio,
  P03_Dashboard,
  P04_Produccion,
  P05_Forecasting,
  P06_Analisis,
  P07_SignalBridge,
  P08_Outro,
} as const;

/**
 * Frame cues — each cue is at the start of a scene transition or key moment.
 * Used to time SFX + music ducking.
 */
const sceneStart = (id: keyof typeof PITCH_TIMINGS) => pitchSceneStart(id);

const SFX_CUES: Array<{ src: string; atFrame: number; volume?: number; dur?: number }> = [
  // P01 → P02 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P02_Inicio") - 10, volume: 0.55, dur: 30 },
  // P02 → P03 whoosh (key OOS reveal)
  { src: SFX.whoosh, atFrame: sceneStart("P03_Dashboard") - 10, volume: 0.65, dur: 30 },
  // P03 metrics reveal — impact boom on +25.63% reveal
  { src: SFX.riser, atFrame: sceneStart("P03_Dashboard") + 240, volume: 0.5, dur: 45 },
  { src: SFX.impact, atFrame: sceneStart("P03_Dashboard") + 280, volume: 0.75, dur: 60 },
  { src: SFX.numberTick, atFrame: sceneStart("P03_Dashboard") + 300, volume: 0.5, dur: 10 },
  { src: SFX.numberTick, atFrame: sceneStart("P03_Dashboard") + 320, volume: 0.5, dur: 10 },
  { src: SFX.numberTick, atFrame: sceneStart("P03_Dashboard") + 340, volume: 0.5, dur: 10 },
  // P03 → P04 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P04_Produccion") - 10, volume: 0.6, dur: 30 },
  // P04 gate blocking — pop per bar (14 bars, staggered 8 frames)
  ...Array.from({ length: 14 }).map((_, i) => ({
    src: SFX.pop,
    atFrame: sceneStart("P04_Produccion") + 90 + i * 8,
    volume: 0.25,
    dur: 8,
  })),
  // P04 → P05 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P05_Forecasting") - 10, volume: 0.6, dur: 30 },
  // P05 → P06 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P06_Analisis") - 10, volume: 0.55, dur: 30 },
  // P06 → P07 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P07_SignalBridge") - 10, volume: 0.6, dur: 30 },
  // P07 kill switch callout
  { src: SFX.impact, atFrame: sceneStart("P07_SignalBridge") + 270, volume: 0.55, dur: 60 },
  // P07 → P08 whoosh
  { src: SFX.whoosh, atFrame: sceneStart("P08_Outro") - 10, volume: 0.6, dur: 30 },
  // P08 logo reveal — soft impact
  { src: SFX.impact, atFrame: sceneStart("P08_Outro") + 5, volume: 0.4, dur: 60 },
];

const DUCK_FRAMES = SFX_CUES
  .filter((c) => c.src === SFX.impact || c.src === SFX.riser)
  .map((c) => c.atFrame);

export const Pitch: React.FC<PitchProps> = ({ variant, vertical }) => {
  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.primary }}>
      {PITCH_SCENE_ORDER.map((id) => {
        const Scene = SCENE_MAP[id];
        const from = pitchSceneStart(id);
        const dur = PITCH_TIMINGS[id];
        return (
          <Sequence
            key={id}
            from={from}
            durationInFrames={dur}
            layout="none"
            name={id}
          >
            <Scene variant={variant} vertical={vertical} />
          </Sequence>
        );
      })}

      {/* Audio ONLY in final variant */}
      {variant === "final" && (
        <>
          <BackgroundMusic
            baseVolume={0.32}
            duckAt={DUCK_FRAMES}
            totalFrames={PITCH_TOTAL_FRAMES}
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

/**
 * Only visible in RAW variant: shows active scene id + frame counter
 * as a small overlay (for designer handoff reference).
 */
const RawSceneMarker: React.FC = () => {
  return (
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
      RAW · handoff cut
    </div>
  );
};
