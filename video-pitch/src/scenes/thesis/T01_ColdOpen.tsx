import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T01 · Cold Open · 25s
 *
 * The paradox hook. Three acts in 750 frames:
 *   0-180f   : Black. Single line appears letter-by-letter.
 *              "El modelo predijo mal."   (red)
 *   180-360f : Beat of silence, line lingers.
 *   360-540f : Second line appears below, building tension.
 *              "Y aún así, ganó."        (green)
 *   540-750f : Both lines lock, tiny sub-line fades in:
 *              "Esta es la historia de por qué."
 */
export const T01_ColdOpen: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Line 1 typewriter: 0 → 90f
  const line1 = "El modelo predijo mal.";
  const line1Chars = Math.floor(
    interpolate(frame, [0, 90], [0, line1.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );
  const line1Visible = line1.slice(0, line1Chars);

  // Line 1 glow ramp
  const line1Glow = interpolate(frame, [30, 150, 360, 540], [0, 1, 1, 0.6], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 2 springs in at frame 360
  const line2Progress = spring({
    frame: frame - 360,
    fps,
    config: { damping: 18, stiffness: 140, mass: 0.8 },
  });
  const line2Y = interpolate(line2Progress, [0, 1], [30, 0]);
  const line2Opacity = interpolate(line2Progress, [0, 1], [0, 1]);

  // Sub-line fades in at frame 600
  const subProgress = interpolate(frame, [600, 680], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Vignette pulse synced with second line
  const vignetteOpacity = interpolate(
    frame,
    [0, 120, 360, 480, 750],
    [0.3, 0.5, 0.5, 0.75, 0.9],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Final fade-out on last 30 frames
  const endFade = interpolate(frame, [720, 750], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: TOKENS.colors.bg.deep,
        opacity: endFade,
      }}
    >
      {/* Vignette */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at center, transparent 20%, rgba(0,0,0,${vignetteOpacity}) 75%)`,
          pointerEvents: "none",
        }}
      />

      {/* Stack container */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 32,
          padding: "0 120px",
        }}
      >
        {/* Line 1: "El modelo predijo mal." */}
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            fontSize: 96,
            letterSpacing: -2,
            color: TOKENS.colors.market.down,
            textAlign: "center",
            textShadow: `0 0 ${20 * line1Glow}px rgba(255,59,105,${0.6 * line1Glow})`,
            minHeight: 120,
          }}
        >
          {line1Visible}
          {line1Chars < line1.length && frame < 90 && (
            <span style={{ opacity: Math.floor(frame / 8) % 2 }}>|</span>
          )}
        </div>

        {/* Line 2: "Y aún así, ganó." */}
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            fontSize: 120,
            letterSpacing: -3,
            color: TOKENS.colors.market.up,
            textAlign: "center",
            transform: `translateY(${line2Y}px)`,
            opacity: line2Opacity,
            textShadow: `0 0 32px rgba(0,211,149,0.65)`,
            minHeight: 140,
          }}
        >
          Y aún así, ganó.
        </div>

        {/* Sub-line */}
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 400,
            fontSize: 32,
            letterSpacing: 4,
            textTransform: "uppercase",
            color: TOKENS.colors.text.secondary,
            opacity: subProgress,
            marginTop: 40,
          }}
        >
          Esta es la historia de por qué.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
