import React from "react";
import { AbsoluteFill, Sequence, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { BrandLogoReveal } from "../../components/BrandLogoReveal";
import { DisclaimerRoll } from "../../components/DisclaimerRoll";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P08 · Outro (5s)
 * Logo reveal + disclaimer + CTA
 */
export const P08_Outro: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at center, ${TOKENS.colors.bg.secondary}, ${TOKENS.colors.bg.deep})`,
      }}
    >
      {/* Gradient mesh decoration */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 25% 35%, ${TOKENS.colors.accent.cyan}15, transparent 40%), radial-gradient(circle at 75% 65%, ${TOKENS.colors.accent.purple}15, transparent 40%)`,
        }}
      />

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 48,
          padding: 80,
        }}
      >
        <BrandLogoReveal delay={2} />
        <Sequence from={55}>
          <DisclaimerRoll delay={0} />
        </Sequence>
        <Sequence from={110}>
          <div
            style={{
              opacity: interpolate(frame - 110, [0, 18], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
              fontSize: 22,
              color: TOKENS.colors.accent.cyan,
              fontFamily: "JetBrains Mono, monospace",
              letterSpacing: 3,
            }}
          >
            globalminds.ai · 2026
          </div>
        </Sequence>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
