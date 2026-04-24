import React from "react";
import { Audio, Sequence, staticFile, interpolate } from "remotion";

/**
 * Background music — plays across the full composition with fade in/out
 * and peak-duck points where SFX impact.
 */
export const BackgroundMusic: React.FC<{
  src?: string;
  baseVolume?: number;
  /** frame numbers (global) where music ducks to low */
  duckAt?: number[];
  /** total composition duration in frames */
  totalFrames: number;
}> = ({
  src = "audio/music/pitch-ambient.wav",
  baseVolume = 0.35,
  duckAt = [],
  totalFrames,
}) => {
  // Volume function: base, ducks to 30% around duck points (±6 frames)
  const volumeFn = (frame: number) => {
    // Overall fade in/out
    const fadeIn = interpolate(frame, [0, 20], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
    const fadeOut = interpolate(
      frame,
      [totalFrames - 30, totalFrames],
      [1, 0],
      { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
    );
    let duck = 1.0;
    for (const d of duckAt) {
      const dist = Math.abs(frame - d);
      if (dist < 12) {
        const factor = interpolate(dist, [0, 12], [0.3, 1.0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        duck = Math.min(duck, factor);
      }
    }
    return baseVolume * fadeIn * fadeOut * duck;
  };

  return <Audio src={staticFile(src)} volume={volumeFn} />;
};

/**
 * One-shot SFX — plays a short clip at a specific frame (global).
 * Cleanly positioned via <Sequence from=...>.
 */
export const SFXCue: React.FC<{
  src: string;
  atFrame: number;
  volume?: number;
  /** how long the SFX lasts (frames) — used to size the Sequence */
  durationInFrames?: number;
}> = ({ src, atFrame, volume = 0.7, durationInFrames = 60 }) => {
  return (
    <Sequence from={atFrame} durationInFrames={durationInFrames} layout="none">
      <Audio src={staticFile(src)} volume={volume} />
    </Sequence>
  );
};

/** SFX paths (relative to public/) */
export const SFX = {
  whoosh: "audio/sfx/whoosh.wav",
  tick: "audio/sfx/typewriter-tick.wav",
  numberTick: "audio/sfx/number-tick.wav",
  impact: "audio/sfx/impact-boom.wav",
  pop: "audio/sfx/notification-pop.wav",
  riser: "audio/sfx/subtle-riser.wav",
} as const;
