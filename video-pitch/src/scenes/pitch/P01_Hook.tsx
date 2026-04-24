import React from "react";
import { AbsoluteFill, Sequence, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { Typewriter } from "../../components/Typewriter";
import { EquityCurve } from "../../components/EquityCurve";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P01 · Hook (4s) — PLATFORM POWER, not market event
 * "Una plataforma integral." + "Datos. Modelos. Ejecución." + equity teaser
 */
export const P01_Hook: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at center, ${TOKENS.colors.bg.secondary}, ${TOKENS.colors.bg.deep})`,
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 32,
      }}
    >
      {/* Gradient overlay blob */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 25% 35%, ${TOKENS.colors.accent.cyan}18, transparent 45%), radial-gradient(circle at 75% 65%, ${TOKENS.colors.accent.purple}18, transparent 45%)`,
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "relative",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 18,
          zIndex: 2,
        }}
      >
        <Typewriter
          text="Una plataforma."
          speed={22}
          delay={4}
          fontSize={88}
          fontWeight={800}
          gradient={[TOKENS.colors.accent.cyan, TOKENS.colors.accent.purple]}
          letterSpacing={-1}
          cursor={false}
        />
        <Typewriter
          text="Datos · Modelos · Ejecución."
          speed={28}
          delay={36}
          fontSize={48}
          fontWeight={600}
          color={TOKENS.colors.text.secondary}
          letterSpacing={2}
          cursor={false}
        />
      </div>

      <Sequence from={70}>
        <div
          style={{
            position: "relative",
            zIndex: 2,
            opacity: interpolate(frame, [70, 90], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <EquityCurve
            width={720}
            height={150}
            durationInFrames={50}
            strokeWidth={3}
            showFinalValue={false}
          />
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
