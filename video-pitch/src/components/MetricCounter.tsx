import React from "react";
import { Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface MetricCounterProps {
  target: number;
  /** frames to fully reach target */
  durationInFrames?: number;
  /** delay before starting */
  delay?: number;
  /** formatting type */
  format?: "pct" | "ratio" | "int" | "pvalue" | "money";
  /** digits after decimal (ratio/pct) */
  digits?: number;
  /** label below the number */
  label?: string;
  /** font size (number) */
  fontSize?: number;
  /** color of the number — defaults to cyan→purple gradient */
  gradient?: [string, string];
  /** solid color override (skips gradient) */
  color?: string;
  /** label font size */
  labelSize?: number;
  /** align self */
  align?: "center" | "flex-start" | "flex-end";
}

const fmt = (v: number, type: MetricCounterProps["format"], digits: number): string => {
  switch (type) {
    case "pct":
      return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
    case "ratio":
      return v.toFixed(digits);
    case "int":
      return Math.round(v).toLocaleString("en-US");
    case "pvalue":
      return v < 0.01 ? `p=${v.toFixed(4)}` : `p=${v.toFixed(3)}`;
    case "money":
      return `$${Math.round(v).toLocaleString("en-US")}`;
    default:
      return v.toString();
  }
};

export const MetricCounter: React.FC<MetricCounterProps> = ({
  target,
  durationInFrames = 45,
  delay = 0,
  format = "ratio",
  digits = 2,
  label,
  fontSize = 96,
  gradient = [TOKENS.colors.accent.cyan, TOKENS.colors.accent.purple],
  color,
  labelSize = 22,
  align = "center",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const localFrame = Math.max(0, frame - delay);

  const value = interpolate(localFrame, [0, durationInFrames], [0, target], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  const scale = spring({
    frame: localFrame,
    fps,
    config: { damping: 15, stiffness: 180, mass: 0.6 },
    durationInFrames: 20,
  });

  const text = fmt(value, format, digits);

  const numberStyle: React.CSSProperties = color
    ? { color }
    : {
        background: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`,
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
      };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: align, gap: 6 }}>
      <div
        style={{
          fontSize,
          fontWeight: 800,
          fontFamily: "JetBrains Mono, monospace",
          letterSpacing: -1,
          fontVariantNumeric: "tabular-nums",
          transform: `scale(${scale})`,
          ...numberStyle,
        }}
      >
        {text}
      </div>
      {label && (
        <div
          style={{
            fontSize: labelSize,
            color: TOKENS.colors.text.secondary,
            letterSpacing: 2,
            textTransform: "uppercase",
            fontFamily: "Inter, system-ui",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
};
