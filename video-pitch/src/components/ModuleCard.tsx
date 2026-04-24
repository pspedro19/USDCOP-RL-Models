import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface ModuleCardProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  delay?: number;
  accent?: string;
  width?: number;
  height?: number;
}

export const ModuleCard: React.FC<ModuleCardProps> = ({
  title,
  subtitle,
  icon,
  delay = 0,
  accent = TOKENS.colors.accent.cyan,
  width = 280,
  height = 160,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);
  const appear = spring({
    frame: local,
    fps,
    config: { damping: 14, stiffness: 180, mass: 0.6 },
    durationInFrames: 25,
  });
  const slide = (1 - appear) * 30;
  return (
    <div
      style={{
        width,
        height,
        background: `linear-gradient(145deg, ${TOKENS.colors.bg.card}, ${TOKENS.colors.bg.secondary})`,
        border: `1px solid ${accent}40`,
        borderRadius: 16,
        padding: 22,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        gap: 8,
        transform: `translateY(${slide}px) scale(${appear})`,
        opacity: appear,
        boxShadow: `0 8px 32px ${accent}22`,
      }}
    >
      <div style={{ fontSize: 40, color: accent, lineHeight: 1 }}>{icon}</div>
      <div>
        <div
          style={{
            fontSize: 26,
            fontWeight: 700,
            color: TOKENS.colors.text.primary,
            fontFamily: "Inter, system-ui",
            marginBottom: 4,
          }}
        >
          {title}
        </div>
        {subtitle && (
          <div
            style={{
              fontSize: 16,
              color: TOKENS.colors.text.secondary,
              letterSpacing: 1,
            }}
          >
            {subtitle}
          </div>
        )}
      </div>
    </div>
  );
};
