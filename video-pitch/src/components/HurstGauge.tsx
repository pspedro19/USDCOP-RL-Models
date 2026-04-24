import React from "react";
import { Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface HurstGaugeProps {
  from?: number;
  to: number;
  durationInFrames?: number;
  delay?: number;
  size?: number;
  showLabel?: boolean;
}

export const HurstGauge: React.FC<HurstGaugeProps> = ({
  from = 0.5,
  to,
  durationInFrames = 45,
  delay = 0,
  size = 360,
  showLabel = true,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);
  const value = interpolate(local, [0, durationInFrames], [from, to], {
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });

  const appear = spring({
    frame: local,
    fps,
    config: { damping: 18, stiffness: 120 },
    durationInFrames: 25,
  });

  // Arc from 180° (left, value=0) to 0° (right, value=1) — half circle upper
  const angle = 180 - value * 180; // value 0 → 180°, value 1 → 0°
  const rad = (angle * Math.PI) / 180;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 14;
  const tipX = cx + r * Math.cos(rad);
  const tipY = cy - r * Math.sin(rad);

  // Zone colors — mean-reverting (<0.42) red, indeterminate (0.42-0.52) amber, trending (>0.52) green
  const zoneColor =
    value < 0.42
      ? TOKENS.colors.semantic.danger
      : value < 0.52
        ? TOKENS.colors.semantic.warning
        : TOKENS.colors.semantic.success;
  const zoneLabel =
    value < 0.42 ? "Mean-Reverting" : value < 0.52 ? "Indeterminate" : "Trending";

  // Build the arc path (semicircle)
  const left = { x: cx - r, y: cy };
  const right = { x: cx + r, y: cy };

  return (
    <div
      style={{
        transform: `scale(${appear})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 18,
      }}
    >
      <svg width={size} height={size / 2 + 30} style={{ overflow: "visible" }}>
        {/* Background arc */}
        <path
          d={`M ${left.x} ${left.y} A ${r} ${r} 0 0 1 ${right.x} ${right.y}`}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={18}
          strokeLinecap="round"
        />
        {/* Zone markers */}
        {[
          { v: 0.42, color: TOKENS.colors.semantic.danger },
          { v: 0.52, color: TOKENS.colors.semantic.success },
        ].map((m) => {
          const a = ((180 - m.v * 180) * Math.PI) / 180;
          const x = cx + r * Math.cos(a);
          const y = cy - r * Math.sin(a);
          return (
            <line
              key={m.v}
              x1={x}
              y1={y}
              x2={cx + (r + 12) * Math.cos(a)}
              y2={cy - (r + 12) * Math.sin(a)}
              stroke={m.color}
              strokeWidth={3}
            />
          );
        })}
        {/* Progress arc */}
        <path
          d={`M ${left.x} ${left.y} A ${r} ${r} 0 0 1 ${tipX} ${tipY}`}
          fill="none"
          stroke={zoneColor}
          strokeWidth={18}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${zoneColor}80)` }}
        />
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={tipX}
          y2={tipY}
          stroke={TOKENS.colors.text.primary}
          strokeWidth={3}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={10} fill={zoneColor} />
      </svg>
      <div
        style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: size * 0.16,
          fontWeight: 800,
          color: zoneColor,
          fontVariantNumeric: "tabular-nums",
          letterSpacing: -1,
        }}
      >
        H = {value.toFixed(3)}
      </div>
      {showLabel && (
        <div
          style={{
            fontSize: size * 0.06,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            letterSpacing: 4,
            textTransform: "uppercase",
          }}
        >
          {zoneLabel}
        </div>
      )}
    </div>
  );
};
