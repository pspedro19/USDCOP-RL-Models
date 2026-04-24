import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T36 · Fade out · 5s (150f)
 * Silent fade. Attribution line for the FIUBA logo (CC BY-SA 3.0).
 */
export const T36_FadeMusical: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const op = interpolate(frame, [0, 30, 120, 150], [0, 0.85, 0.4, 0], {
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
          gap: 6,
          opacity: op,
        }}
      >
        <div
          style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 12,
            color: TOKENS.colors.text.muted,
            letterSpacing: 3,
            textTransform: "uppercase",
          }}
        >
          Logo FIUBA · CC BY-SA 3.0 · Wikimedia Commons
        </div>
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontSize: 12,
            color: TOKENS.colors.text.muted,
            letterSpacing: 2,
          }}
        >
          Generado con Remotion · Captures via Playwright · Abril 2026
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
