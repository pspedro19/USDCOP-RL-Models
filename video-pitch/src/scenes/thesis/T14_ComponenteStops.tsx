import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T14 · Effective HS + Dynamic Leverage · 50s (1500f)
 *
 * Left: Formula for effective HS = min(HS_base, 3.5% / leverage)
 * Right: DL scaling based on rolling WR + drawdown
 */
export const T14_ComponenteStops: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const leftOp = interpolate(frame, [90, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rightOp = interpolate(frame, [450, 540], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bottomOp = interpolate(frame, [900, 1020], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1470, 1500], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // DL scaling curve point (animated)
  const dlProgress = interpolate(frame, [540, 1200], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rollingWR = 0.3 + 0.7 * (1 - Math.cos(dlProgress * Math.PI * 2)) * 0.5;
  const leverage = Math.max(0.25, Math.min(1.0, rollingWR));

  return (
    <AbsoluteFill
      style={{
        background: TOKENS.colors.bg.primary,
        opacity: exitFade,
      }}
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
            color: TOKENS.colors.market.up,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO III · COMPONENTE 3 — EJECUCIÓN
        </div>
        <div
          style={{
            fontSize: 44,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -1,
            marginTop: 4,
          }}
        >
          Effective HS + Dynamic Leverage
        </div>
      </div>

      {/* LEFT · Effective HS */}
      <div
        style={{
          position: "absolute",
          top: 220,
          left: 80,
          width: 840,
          padding: "24px 28px",
          background: "rgba(0,211,149,0.10)",
          border: `1px solid ${TOKENS.colors.market.up}66`,
          borderRadius: 12,
          opacity: leftOp,
        }}
      >
        <div
          style={{
            fontSize: 16,
            letterSpacing: 5,
            color: TOKENS.colors.market.up,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Effective Hard Stop
        </div>
        <div
          style={{
            fontSize: 36,
            fontFamily: "JetBrains Mono, monospace",
            color: "#fff",
            marginTop: 12,
            letterSpacing: 0.5,
            lineHeight: 1.4,
          }}
        >
          HS_eff = min(HS_base, <br />
          <span style={{ color: TOKENS.colors.market.up }}>3.5%</span> /
          leverage)
        </div>
        <div
          style={{
            marginTop: 20,
            padding: "14px 18px",
            background: "rgba(0,0,0,0.35)",
            borderRadius: 8,
            fontSize: 18,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            lineHeight: 1.5,
          }}
        >
          Si lev = 2.0, HS efectivo ≤ 1.75%. <br />
          Si lev = 0.25, HS efectivo ≤ HS_base (hasta 3.5%). <br />
          <span style={{ color: TOKENS.colors.market.up }}>
            La pérdida máxima por trade se capa al 3.5% del portafolio
          </span>{" "}
          sin importar el apalancamiento.
        </div>
      </div>

      {/* RIGHT · Dynamic Leverage */}
      <div
        style={{
          position: "absolute",
          top: 220,
          left: 960,
          width: 880,
          padding: "24px 28px",
          background: "rgba(139,92,246,0.10)",
          border: `1px solid ${TOKENS.colors.accent.purple}66`,
          borderRadius: 12,
          opacity: rightOp,
        }}
      >
        <div
          style={{
            fontSize: 16,
            letterSpacing: 5,
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Dynamic Leverage
        </div>
        <div
          style={{
            fontSize: 28,
            fontFamily: "JetBrains Mono, monospace",
            color: "#fff",
            marginTop: 12,
            letterSpacing: 0.5,
            lineHeight: 1.35,
          }}
        >
          lev = clip(<br />
          &nbsp;&nbsp;f(rolling_WR, rolling_DD), <br />
          &nbsp;&nbsp;0.25, 1.0<br />
          )
        </div>

        <div
          style={{
            marginTop: 16,
            padding: "10px 16px",
            background: "rgba(0,0,0,0.35)",
            borderRadius: 8,
            fontSize: 16,
            fontFamily: "JetBrains Mono, monospace",
            color: TOKENS.colors.text.secondary,
          }}
        >
          rolling WR actual ={" "}
          <span style={{ color: TOKENS.colors.accent.purple }}>
            {(rollingWR * 100).toFixed(0)}%
          </span>{" "}
          → lev ={" "}
          <span style={{ color: TOKENS.colors.accent.purple, fontWeight: 700 }}>
            {leverage.toFixed(2)}x
          </span>
        </div>

        <div
          style={{
            marginTop: 14,
            fontSize: 17,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            lineHeight: 1.5,
          }}
        >
          Cuando la racha reciente es fuerte, el sistema se apoya. Cuando es
          débil, se repliega automáticamente. <br />
          <span style={{ color: TOKENS.colors.accent.purple }}>
            Gestión procíclica del riesgo sin intervención manual.
          </span>
        </div>
      </div>

      {/* Bottom evidence */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          left: 80,
          right: 80,
          padding: "18px 28px",
          background: "rgba(6,182,212,0.08)",
          border: `1px solid ${TOKENS.colors.accent.cyan}88`,
          borderRadius: 10,
          opacity: bottomOp,
          display: "flex",
          gap: 40,
          alignItems: "center",
        }}
      >
        <Metric value="-6.12%" label="MaxDD 2025 OOS" color={TOKENS.colors.market.up} />
        <Metric value="2" label="Hard stops ejecutados" color="#fff" />
        <Metric value="21" label="Take-profit exits (62%)" color={TOKENS.colors.accent.cyan} />
        <div
          style={{
            flex: 1,
            fontSize: 18,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            fontStyle: "italic",
          }}
        >
          HS efectivo + DL transforman señales marginales en un perfil
          asimétrico (ganancias &gt; pérdidas) sin añadir predictibilidad.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Metric: React.FC<{ value: string; label: string; color: string }> = ({
  value,
  label,
  color,
}) => (
  <div>
    <div
      style={{
        fontSize: 40,
        fontFamily: "JetBrains Mono, monospace",
        fontWeight: 700,
        color,
        letterSpacing: -1,
      }}
    >
      {value}
    </div>
    <div
      style={{
        fontSize: 13,
        fontFamily: "Inter, system-ui",
        color: TOKENS.colors.text.secondary,
        letterSpacing: 2,
        textTransform: "uppercase",
        marginTop: 2,
      }}
    >
      {label}
    </div>
  </div>
);
