import React from "react";
import { Easing, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface EquityCurveProps {
  /** list of equity values, first = initial, last = final */
  values?: number[];
  width?: number;
  height?: number;
  /** stroke draw duration (frames) */
  durationInFrames?: number;
  delay?: number;
  stroke?: string;
  strokeWidth?: number;
  showFinalValue?: boolean;
}

/** Synthetic equity curve mimicking +25.63% 2025 pattern */
const DEFAULT_VALUES = (() => {
  const pts: number[] = [];
  const n = 52;
  for (let i = 0; i < n; i++) {
    // Blend noisy + overall up trend to 25.63%
    const trend = i / (n - 1); // 0 → 1
    const vol = Math.sin(i * 0.6) * 0.02 + Math.sin(i * 0.22) * 0.015;
    const level = 1 + trend * 0.2563 + vol * (1 - trend * 0.5);
    pts.push(level);
  }
  return pts;
})();

export const EquityCurve: React.FC<EquityCurveProps> = ({
  values = DEFAULT_VALUES,
  width = 900,
  height = 320,
  durationInFrames = 60,
  delay = 0,
  stroke,
  strokeWidth = 4,
  showFinalValue = true,
}) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - delay);

  const pad = 20;
  const maxV = Math.max(...values);
  const minV = Math.min(...values);
  const range = maxV - minV || 1;

  // Build path
  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - ((v - minV) / range) * (height - pad * 2);
    return { x, y };
  });
  const d = points
    .map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
    .join(" ");

  // Compute total length (approximation) for dashoffset animation
  let totalLen = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    totalLen += Math.sqrt(dx * dx + dy * dy);
  }

  const progress = interpolate(local, [0, durationInFrames], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const dashOffset = totalLen * (1 - progress);

  const strokeColor = stroke ?? TOKENS.colors.market.up;

  const strokeId = React.useId().replace(/:/g, "-");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <defs>
        <linearGradient id={`areaGrad-${strokeId}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity="0.45" />
          <stop offset="100%" stopColor={strokeColor} stopOpacity="0" />
        </linearGradient>
        <filter id={`glow-${strokeId}`}>
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* Filled area under curve, revealed with the line */}
      <clipPath id={`reveal-${strokeId}`}>
        <rect
          x={0}
          y={0}
          width={progress * width}
          height={height}
        />
      </clipPath>
      <path
        d={`${d} L ${points[points.length - 1].x} ${height - pad} L ${points[0].x} ${height - pad} Z`}
        fill={`url(#areaGrad-${strokeId})`}
        clipPath={`url(#reveal-${strokeId})`}
      />
      {/* Line path with stroke-dashoffset animation */}
      <path
        d={d}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={totalLen}
        strokeDashoffset={dashOffset}
        filter={`url(#glow-${strokeId})`}
      />
      {/* Final value pin */}
      {showFinalValue && progress > 0.95 && (
        <g>
          <circle
            cx={points[points.length - 1].x}
            cy={points[points.length - 1].y}
            r={8}
            fill={strokeColor}
          />
          <text
            x={points[points.length - 1].x - 10}
            y={points[points.length - 1].y - 16}
            fontFamily="JetBrains Mono, monospace"
            fontSize={28}
            fontWeight={700}
            fill={strokeColor}
            textAnchor="end"
          >
            +25.63%
          </text>
        </g>
      )}
    </svg>
  );
};
