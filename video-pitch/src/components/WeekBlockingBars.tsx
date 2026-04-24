import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface WeekBlockingBarsProps {
  totalWeeks: number;
  blockedWeeks: number;
  /** how many frames between each bar revealing */
  staggerFrames?: number;
  delay?: number;
  /** total frames from first bar to last bar complete */
  perBarDuration?: number;
}

export const WeekBlockingBars: React.FC<WeekBlockingBarsProps> = ({
  totalWeeks,
  blockedWeeks,
  staggerFrames = 6,
  delay = 0,
  perBarDuration = 20,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);
  const barW = 42;
  const barH = 80;
  const gap = 10;

  return (
    <div style={{ display: "flex", gap, alignItems: "end" }}>
      {Array.from({ length: totalWeeks }).map((_, i) => {
        const isBlocked = i < blockedWeeks;
        const revealStart = i * staggerFrames;
        const revealLocal = Math.max(0, local - revealStart);
        const appear = spring({
          frame: revealLocal,
          fps,
          config: { damping: 14, stiffness: 180 },
          durationInFrames: perBarDuration,
        });
        // X overlay shows up slightly AFTER bar appears (for blocked)
        const xAppear = spring({
          frame: Math.max(0, revealLocal - 8),
          fps,
          config: { damping: 16, stiffness: 220 },
          durationInFrames: 12,
        });

        const barColor = isBlocked
          ? TOKENS.colors.semantic.danger
          : TOKENS.colors.market.up;

        return (
          <div
            key={i}
            style={{
              position: "relative",
              width: barW,
              height: barH,
              transform: `scaleY(${appear}) translateY(${(1 - appear) * 20}px)`,
              transformOrigin: "bottom center",
            }}
          >
            <div
              style={{
                width: "100%",
                height: "100%",
                background: `linear-gradient(180deg, ${barColor}aa, ${barColor}44)`,
                border: `1px solid ${barColor}`,
                borderRadius: 4,
                boxShadow: `0 0 12px ${barColor}66`,
              }}
            />
            {isBlocked && xAppear > 0.1 && (
              <svg
                width={barW}
                height={barH}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  transform: `scale(${xAppear})`,
                }}
              >
                <line
                  x1={barW * 0.15}
                  y1={barH * 0.2}
                  x2={barW * 0.85}
                  y2={barH * 0.8}
                  stroke="white"
                  strokeWidth={3}
                  strokeLinecap="round"
                />
                <line
                  x1={barW * 0.85}
                  y1={barH * 0.2}
                  x2={barW * 0.15}
                  y2={barH * 0.8}
                  stroke="white"
                  strokeWidth={3}
                  strokeLinecap="round"
                />
              </svg>
            )}
            {/* Week label */}
            <div
              style={{
                position: "absolute",
                bottom: -28,
                left: 0,
                width: "100%",
                textAlign: "center",
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 14,
                color: TOKENS.colors.text.muted,
              }}
            >
              W{(i + 1).toString().padStart(2, "0")}
            </div>
          </div>
        );
      })}
    </div>
  );
};
