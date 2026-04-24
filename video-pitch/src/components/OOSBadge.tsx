import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export const OOSBadge: React.FC<{ label?: string; delay?: number; color?: string }> = ({
  label = "Validación Out-of-Sample 2025",
  delay = 0,
  color = TOKENS.colors.market.up,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);
  const appear = spring({
    frame: local,
    fps,
    config: { damping: 14, stiffness: 180, mass: 0.6 },
    durationInFrames: 20,
  });
  const pulse = 1 + Math.sin(local / 12) * 0.04;

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 24px",
        border: `2px solid ${color}`,
        borderRadius: 999,
        background: `${color}15`,
        color,
        fontFamily: "Inter, system-ui",
        fontSize: 22,
        fontWeight: 600,
        letterSpacing: 3,
        textTransform: "uppercase",
        transform: `scale(${appear * pulse})`,
        boxShadow: `0 0 24px ${color}40`,
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 999,
          background: color,
          boxShadow: `0 0 8px ${color}`,
        }}
      />
      {label}
    </div>
  );
};
