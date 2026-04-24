import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T05 · Hipótesis · 40s (1200f)
 *
 * Academic structure:
 *   Primary hypothesis (top)
 *   H1, H2, H3 sub-hypotheses (staggered cards)
 *   H-alternative (null/falsifier)
 */
const HYPOTHESES = [
  {
    id: "H1",
    title: "Predictibilidad (debilitada)",
    body: "Un ensemble ML sobre features macro + técnicas exhibe DA marginalmente superior a 50% a horizonte semanal, aunque no significativo con corrección múltiple.",
    color: TOKENS.colors.accent.cyan,
  },
  {
    id: "H2",
    title: "Detección de régimen (MVP)",
    body: "El coeficiente de Hurst R/S clasifica semanas en trending / mean-reverting / indeterminate con potencia suficiente para decidir cuándo NO operar.",
    color: "#00D395",
  },
  {
    id: "H3",
    title: "Ejecución adaptativa",
    body: "Stops efectivos acoplados al apalancamiento dinámico protegen el capital cuando el régimen fuerza entrada en zona indeterminada.",
    color: TOKENS.colors.accent.purple,
  },
];

export const T05_Hipotesis: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const primaryOp = interpolate(frame, [60, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const primaryY = interpolate(frame, [60, 150], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const altOp = interpolate(frame, [960, 1050], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1170, 1200], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

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
          top: 60,
          left: 80,
          opacity: titleOp,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 6,
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO I · SECCIÓN 3 · HIPÓTESIS
        </div>
        <div
          style={{
            fontSize: 48,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -1,
            marginTop: 6,
          }}
        >
          Hipótesis tripartita
        </div>
      </div>

      {/* Primary hypothesis */}
      <div
        style={{
          position: "absolute",
          top: 210,
          left: 80,
          right: 80,
          padding: "22px 28px",
          background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}22, ${TOKENS.colors.accent.purple}22)`,
          border: `1px solid ${TOKENS.colors.accent.cyan}`,
          borderRadius: 10,
          opacity: primaryOp,
          transform: `translateY(${primaryY}px)`,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 5,
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Hipótesis primaria (H₀)
        </div>
        <div
          style={{
            fontSize: 26,
            fontFamily: "Inter, system-ui",
            fontWeight: 600,
            color: "#fff",
            letterSpacing: -0.3,
            marginTop: 8,
            lineHeight: 1.3,
          }}
        >
          La combinación de <span style={{ color: TOKENS.colors.accent.cyan }}>
            predicción
          </span>
          ,{" "}
          <span style={{ color: "#00D395" }}>filtrado de régimen</span>
          {" "}y{" "}
          <span style={{ color: TOKENS.colors.accent.purple }}>
            ejecución adaptativa
          </span>{" "}
          produce un sistema rentable en USD/COP con significancia estadística,
          incluso cuando cada componente aislado resulta marginal.
        </div>
      </div>

      {/* H1, H2, H3 cards (staggered) */}
      <div
        style={{
          position: "absolute",
          top: 430,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 24,
        }}
      >
        {HYPOTHESES.map((h, idx) => {
          const appearAt = 240 + idx * 120;
          const op = interpolate(frame, [appearAt, appearAt + 90], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const y = interpolate(frame, [appearAt, appearAt + 90], [40, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={h.id}
              style={{
                opacity: op,
                transform: `translateY(${y}px)`,
                padding: "24px 22px",
                background: `${h.color}18`,
                border: `1px solid ${h.color}99`,
                borderRadius: 10,
                minHeight: 280,
              }}
            >
              <div
                style={{
                  fontSize: 42,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: h.color,
                  letterSpacing: -1,
                  marginBottom: 8,
                }}
              >
                {h.id}
              </div>
              <div
                style={{
                  fontSize: 22,
                  fontFamily: "Inter, system-ui",
                  fontWeight: 700,
                  color: "#fff",
                  letterSpacing: -0.3,
                  marginBottom: 12,
                }}
              >
                {h.title}
              </div>
              <div
                style={{
                  fontSize: 18,
                  fontFamily: "Inter, system-ui",
                  color: TOKENS.colors.text.secondary,
                  lineHeight: 1.5,
                }}
              >
                {h.body}
              </div>
            </div>
          );
        })}
      </div>

      {/* Alternative hypothesis (falsifier) */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 80,
          right: 80,
          padding: "16px 24px",
          background: `${TOKENS.colors.market.down}15`,
          border: `1px dashed ${TOKENS.colors.market.down}`,
          borderRadius: 8,
          opacity: altOp,
        }}
      >
        <div
          style={{
            fontSize: 13,
            letterSpacing: 5,
            color: TOKENS.colors.market.down,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          H-alternativa · Criterio de falsación
        </div>
        <div
          style={{
            fontSize: 18,
            fontFamily: "Inter, system-ui",
            color: "#fff",
            marginTop: 4,
            lineHeight: 1.4,
          }}
        >
          La tesis se rechaza si el retorno 2025 OOS no supera 0 % con p &lt; 0.05
          tras corrección Benjamini-Hochberg para k = 70 experimentos previos.
        </div>
      </div>
    </AbsoluteFill>
  );
};
