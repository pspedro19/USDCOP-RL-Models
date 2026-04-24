import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface RegimeBadgeProps {
  regime: "mean-reverting" | "trending" | "indeterminate";
  hurst: number;
  delay?: number;
}

export const RegimeBadge: React.FC<RegimeBadgeProps> = ({ regime, hurst, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);

  const appear = spring({
    frame: local,
    fps,
    config: { damping: 14, stiffness: 180 },
    durationInFrames: 20,
  });

  const color =
    regime === "mean-reverting"
      ? TOKENS.colors.semantic.danger
      : regime === "trending"
        ? TOKENS.colors.semantic.success
        : TOKENS.colors.semantic.warning;

  const label =
    regime === "mean-reverting"
      ? "MEAN-REVERTING"
      : regime === "trending"
        ? "TRENDING"
        : "INDETERMINATE";

  // Slow color pulse: 0.7 → 1.0 opacity cycle
  const pulseOpacity = interpolate(
    Math.sin(local / 10),
    [-1, 1],
    [0.7, 1],
  );

  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        padding: "12px 28px",
        border: `2px solid ${color}`,
        borderRadius: 14,
        background: `${color}20`,
        color,
        transform: `scale(${appear})`,
        opacity: pulseOpacity,
        boxShadow: `0 0 32px ${color}40`,
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          letterSpacing: 5,
          fontFamily: "Inter, system-ui",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 28,
          fontWeight: 800,
          fontFamily: "JetBrains Mono, monospace",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        H = {hurst.toFixed(3)}
      </div>
    </div>
  );
};
