import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";

export interface StubProps {
  id: string;
  label: string;
  variant: string;
  color?: string;
}

/**
 * Placeholder used during scaffold phase.
 * Each final scene will replace this with the real implementation.
 */
export const ThesisStub: React.FC<StubProps> = ({ id, label, variant, color }) => {
  const frame = useCurrentFrame();
  const accent = color ?? TOKENS.colors.accent.cyan;
  return (
    <AbsoluteFill
      style={{
        background: TOKENS.colors.bg.primary,
        color: TOKENS.colors.text.primary,
        fontFamily: "Inter, system-ui",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <div
        style={{
          fontSize: 80,
          fontWeight: 800,
          background: `linear-gradient(135deg, ${accent}, ${TOKENS.colors.accent.purple})`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: -1,
        }}
      >
        {id}
      </div>
      <div
        style={{
          fontSize: 32,
          color: TOKENS.colors.text.secondary,
          letterSpacing: 3,
          textAlign: "center",
          maxWidth: 1200,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 20,
          fontFamily: "JetBrains Mono, monospace",
          color: accent,
          marginTop: 12,
        }}
      >
        {variant.toUpperCase()} · local frame {frame}
      </div>
    </AbsoluteFill>
  );
};
