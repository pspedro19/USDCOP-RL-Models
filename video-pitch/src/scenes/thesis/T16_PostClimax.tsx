import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T16 · Post climax breathing · 8s (240f)
 * Let the audience digest after the +25.63% reveal.
 */
export const T16_PostClimax: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const fadeIn = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(frame, [210, 240], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const caption = interpolate(frame, [60, 120, 200, 240], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 14,
          opacity: fadeIn * fadeOut,
        }}
      >
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontSize: 18,
            color: TOKENS.colors.text.muted,
            letterSpacing: 10,
            textTransform: "uppercase",
            opacity: caption,
          }}
        >
          Pero no es solo el número.
        </div>
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontSize: 32,
            fontWeight: 600,
            color: "#fff",
            letterSpacing: -0.5,
            opacity: caption,
          }}
        >
          Es cómo se obtuvo.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
