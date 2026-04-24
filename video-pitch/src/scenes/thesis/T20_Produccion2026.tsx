import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  staticFile,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import { PITCH_METRICS } from "../../data/metrics";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T20 · Producción 2026 YTD · 25s (750f)
 * Split-view: production capture on the left, metrics on the right.
 * Honest acknowledgment: N = 1 trade so far. Complement, not proof.
 */
export const T20_Produccion2026: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const fadeIn = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const metricsOp = interpolate(frame, [90, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const honestyOp = interpolate(frame, [450, 540], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [720, 750], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const ytd = PITCH_METRICS.ytd_2026;

  return (
    <AbsoluteFill
      style={{
        background: "#000",
        opacity: fadeIn * exitFade,
      }}
    >
      {/* Left capture */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          right: "44%",
          overflow: "hidden",
        }}
      >
        <OffthreadVideo
          src={staticFile("captures/S05-production-live.webm")}
          muted
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: "brightness(0.85)",
          }}
        />
      </div>

      {/* Right panel */}
      <div
        style={{
          position: "absolute",
          top: 0,
          right: 0,
          bottom: 0,
          width: "44%",
          background: `linear-gradient(135deg, ${TOKENS.colors.bg.primary} 0%, ${TOKENS.colors.bg.secondary} 100%)`,
          padding: "60px 56px",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 6,
            color: TOKENS.colors.market.up,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          PRODUCCIÓN 2026 · YEAR-TO-DATE
        </div>
        <div
          style={{
            fontSize: 36,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -0.5,
          }}
        >
          Evidencia complementaria
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
            opacity: metricsOp,
            marginTop: 10,
          }}
        >
          <Metric label="Retorno" value={`+${ytd.return_pct.toFixed(2)}%`} color={TOKENS.colors.market.up} />
          <Metric label="Trades" value={String(ytd.trades)} color={TOKENS.colors.accent.cyan} />
          <Metric label="Pérdidas" value={String(ytd.losses)} color={TOKENS.colors.market.up} />
          <Metric
            label="Semanas bloqueadas"
            value={`${ytd.gate_blocked_weeks}/${ytd.total_weeks}`}
            color={TOKENS.colors.accent.purple}
          />
          <Metric
            label="Buy & Hold"
            value={`${ytd.buy_and_hold_pct.toFixed(2)}%`}
            color={TOKENS.colors.market.down}
          />
          <Metric label="Alpha" value={`+${ytd.alpha_pp.toFixed(2)} pp`} color={TOKENS.colors.market.up} />
        </div>

        <div
          style={{
            marginTop: 20,
            padding: "14px 20px",
            background: "rgba(245,158,11,0.1)",
            border: `1px solid ${TOKENS.colors.semantic.warning}`,
            borderRadius: 8,
            opacity: honestyOp,
            fontSize: 15,
            fontFamily: "Inter, system-ui",
            color: "#fff",
            lineHeight: 1.4,
          }}
        >
          <div
            style={{
              fontSize: 12,
              letterSpacing: 4,
              color: TOKENS.colors.semantic.warning,
              fontFamily: "Inter, system-ui",
              fontWeight: 700,
              textTransform: "uppercase",
            }}
          >
            Honestidad epistemológica
          </div>
          <div style={{ marginTop: 4 }}>
            N = 1 trade. Esto NO valida la tesis independientemente. Es un dato
            consistente con la hipótesis: cuando el régimen es mean-reverting,
            el sistema simplemente se queda en efectivo. El alpha proviene de
            NO haber perdido con Buy-and-Hold.
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Metric: React.FC<{ label: string; value: string; color: string }> = ({
  label,
  value,
  color,
}) => (
  <div
    style={{
      padding: "16px 18px",
      background: `${color}14`,
      border: `1px solid ${color}55`,
      borderRadius: 8,
    }}
  >
    <div
      style={{
        fontSize: 11,
        letterSpacing: 2.5,
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
        fontSize: 32,
        fontFamily: "JetBrains Mono, monospace",
        fontWeight: 700,
        color,
        letterSpacing: -0.5,
        marginTop: 6,
      }}
    >
      {value}
    </div>
  </div>
);
