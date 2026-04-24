import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface Gate {
  label: string;
  value: string;
  threshold: string;
  passed: boolean;
}

export const GateChecklist: React.FC<{
  gates: Gate[];
  delay?: number;
  staggerFrames?: number;
}> = ({ gates, delay = 0, staggerFrames = 8 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 520 }}>
      {gates.map((g, i) => {
        const start = i * staggerFrames;
        const appear = spring({
          frame: Math.max(0, local - start),
          fps,
          config: { damping: 14, stiffness: 180 },
          durationInFrames: 18,
        });
        const color = g.passed ? TOKENS.colors.market.up : TOKENS.colors.market.down;
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
              padding: "12px 18px",
              background: `${color}14`,
              border: `1px solid ${color}66`,
              borderRadius: 10,
              transform: `translateX(${(1 - appear) * 40}px) scale(${appear})`,
              opacity: appear,
            }}
          >
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 999,
                background: color,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white",
                fontSize: 18,
                fontWeight: 800,
                boxShadow: `0 0 12px ${color}`,
              }}
            >
              ✓
            </div>
            <div style={{ flex: 1, fontSize: 20, fontWeight: 600, color: "white" }}>
              {g.label}
            </div>
            <div
              style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 18,
                fontWeight: 700,
                color,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {g.value}
            </div>
          </div>
        );
      })}
    </div>
  );
};
