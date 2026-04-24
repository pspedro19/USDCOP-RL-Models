import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T06 · Breathing beat 1 · 5s (150f)
 * Black beat between Act I and Act II.
 */
export const T06_Breathing1: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const fadeIn = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(frame, [130, 150], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // Tiny dot pulse
  const pulse = 0.6 + 0.4 * Math.sin(frame * 0.08);
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          opacity: fadeIn * fadeOut,
        }}
      >
        <div
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: TOKENS.colors.accent.cyan,
            opacity: pulse,
            boxShadow: `0 0 ${20 * pulse}px ${TOKENS.colors.accent.cyan}`,
          }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
