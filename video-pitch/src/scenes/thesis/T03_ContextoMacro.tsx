import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T03 · Contexto macro · 30s (900f)
 * Sinusoidal "USD/COP" chart + 3 regime bands (trending/indeterm/mean-reverting)
 * + macro anchors (tasa Fed, EMBI, WTI) as sticky notes.
 */
const CHART_W = 1520;
const CHART_H = 380;
const POINTS = 180;

// Procedural path: trending ↑, then mean-reverting, then trending ↓.
const makePath = (): number[] => {
  const ys: number[] = [];
  for (let i = 0; i < POINTS; i++) {
    const t = i / POINTS;
    if (t < 0.4) {
      ys.push(0.5 + t * 0.9 + 0.05 * Math.sin(i * 0.5));
    } else if (t < 0.7) {
      ys.push(0.85 + 0.12 * Math.sin(i * 0.45));
    } else {
      ys.push(0.85 - (t - 0.7) * 1.1 + 0.04 * Math.sin(i * 0.55));
    }
  }
  return ys;
};
const PATH = makePath();

export const T03_ContextoMacro: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const drawProgress = interpolate(frame, [0, 450], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const nDraw = Math.floor(drawProgress * POINTS);

  const titleOp = interpolate(frame, [0, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bandOp = interpolate(frame, [480, 580], [0, 0.6], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const takeawayOp = interpolate(frame, [780, 860], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [870, 900], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Build SVG path
  const toX = (i: number) => 40 + (i / (POINTS - 1)) * (CHART_W - 80);
  const toY = (y: number) => CHART_H - 20 - y * (CHART_H - 60);
  let d = "";
  for (let i = 0; i < nDraw; i++) {
    d += (i === 0 ? "M" : "L") + ` ${toX(i)} ${toY(PATH[i])} `;
  }

  return (
    <AbsoluteFill
      style={{ background: TOKENS.colors.bg.primary, opacity: exitFade }}
    >
      {/* Header */}
      <div
        style={{
          position: "absolute",
          top: 50,
          left: 80,
          opacity: titleOp,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 6,
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO I · MOTIVACIÓN
        </div>
        <div
          style={{
            fontSize: 48,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -1,
            marginTop: 4,
          }}
        >
          USD/COP · tres regímenes en 14 meses
        </div>
      </div>

      {/* Chart area */}
      <div
        style={{
          position: "absolute",
          top: 180,
          left: 80,
          width: CHART_W,
          height: CHART_H,
          background: "rgba(15,23,42,0.5)",
          border: `1px solid ${TOKENS.colors.accent.cyan}33`,
          borderRadius: 10,
          padding: 0,
          overflow: "hidden",
        }}
      >
        <svg width={CHART_W} height={CHART_H} style={{ display: "block" }}>
          {/* Regime bands */}
          <rect
            x={toX(0)}
            y={0}
            width={toX(72) - toX(0)}
            height={CHART_H}
            fill={TOKENS.colors.market.up}
            opacity={bandOp * 0.25}
          />
          <rect
            x={toX(72)}
            y={0}
            width={toX(126) - toX(72)}
            height={CHART_H}
            fill={TOKENS.colors.semantic.warning}
            opacity={bandOp * 0.3}
          />
          <rect
            x={toX(126)}
            y={0}
            width={toX(POINTS - 1) - toX(126)}
            height={CHART_H}
            fill={TOKENS.colors.market.down}
            opacity={bandOp * 0.3}
          />

          {/* Line */}
          <path
            d={d}
            fill="none"
            stroke={TOKENS.colors.accent.cyan}
            strokeWidth={3}
            strokeLinecap="round"
          />
          {/* Dot at current */}
          {nDraw > 0 && nDraw < POINTS && (
            <circle
              cx={toX(nDraw - 1)}
              cy={toY(PATH[nDraw - 1])}
              r={5}
              fill={TOKENS.colors.accent.cyan}
            />
          )}
        </svg>

        {/* Band labels */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            opacity: bandOp * 1.5,
          }}
        >
          <BandLabel text="Trending ↑ H=0.53" left={80} color={TOKENS.colors.market.up} />
          <BandLabel
            text="Indeterm. H=0.49"
            left={700}
            color={TOKENS.colors.semantic.warning}
          />
          <BandLabel
            text="Mean-rev. H=0.28"
            left={1180}
            color={TOKENS.colors.market.down}
          />
        </div>
      </div>

      {/* Macro anchor tags */}
      <div
        style={{
          position: "absolute",
          top: 600,
          left: 80,
          right: 80,
          display: "flex",
          gap: 20,
          opacity: bandOp * 1.5,
        }}
      >
        <MacroChip label="Fed Funds" value="5.33% → 4.50%" />
        <MacroChip label="EMBI Col" value="255 → 302 bps" />
        <MacroChip label="WTI" value="USD 78 → 69" />
        <MacroChip label="TPM BanRep" value="13.00% → 9.25%" />
      </div>

      {/* Takeaway */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 80,
          right: 80,
          padding: "16px 24px",
          background: `${TOKENS.colors.accent.cyan}18`,
          border: `1px solid ${TOKENS.colors.accent.cyan}`,
          borderRadius: 10,
          opacity: takeawayOp,
          fontSize: 22,
          fontFamily: "Inter, system-ui",
          color: "#fff",
          lineHeight: 1.4,
        }}
      >
        <b style={{ color: TOKENS.colors.accent.cyan }}>Takeaway:</b> un
        modelo estacionario es insuficiente. La tesis propone detectar el
        régimen y adaptar la operación — incluida la decisión de{" "}
        <i>no operar</i>.
      </div>
    </AbsoluteFill>
  );
};

const BandLabel: React.FC<{ text: string; left: number; color: string }> = ({
  text,
  left,
  color,
}) => (
  <div
    style={{
      position: "absolute",
      top: 12,
      left,
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 14,
      color,
      letterSpacing: 1,
      background: "rgba(0,0,0,0.5)",
      padding: "4px 10px",
      borderRadius: 4,
      border: `1px solid ${color}`,
    }}
  >
    {text}
  </div>
);

const MacroChip: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div
    style={{
      flex: 1,
      padding: "14px 20px",
      background: "rgba(139,92,246,0.1)",
      border: `1px solid ${TOKENS.colors.accent.purple}66`,
      borderRadius: 8,
    }}
  >
    <div
      style={{
        fontSize: 12,
        letterSpacing: 3,
        color: TOKENS.colors.text.secondary,
        fontFamily: "Inter, system-ui",
        fontWeight: 600,
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
    <div
      style={{
        fontSize: 22,
        fontFamily: "JetBrains Mono, monospace",
        color: "#fff",
        marginTop: 4,
        letterSpacing: -0.3,
      }}
    >
      {value}
    </div>
  </div>
);
