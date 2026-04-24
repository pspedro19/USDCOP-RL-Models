import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T33 · Conclusiones · 40s (1200f)
 * Three big takeaways: paradoja, sinergia, legado.
 */
const CONCLUSIONS = [
  {
    id: "1",
    tag: "PARADOJA",
    title: "El modelo no predice bien.",
    body: "Y aún así, el sistema gana. El DA aislado es marginal (53% con p=0.17). Pero el sistema completo produce +25.63% OOS con p-BH = 0.03.",
    color: TOKENS.colors.accent.cyan,
  },
  {
    id: "2",
    tag: "SINERGIA",
    title: "El edge vive en la unión.",
    body: "Predicción débil + régimen fuerte + ejecución adaptativa = sistema rentable. La ablación muestra que cada componente aportó monotónicamente al resultado.",
    color: TOKENS.colors.market.up,
  },
  {
    id: "3",
    tag: "LEGADO",
    title: "La arquitectura es reproducible.",
    body: "Cualquier investigador con los mismos datos puede replicar la tesis: configs frozen, DAGs versionados, migraciones numeradas, métricas auditables desde psql.",
    color: TOKENS.colors.accent.purple,
  },
];

export const T33_Conclusiones: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
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
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO VI · CONCLUSIONES
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
          Tres formas de leer esta tesis
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          top: 240,
          left: 80,
          right: 80,
          display: "flex",
          flexDirection: "column",
          gap: 18,
        }}
      >
        {CONCLUSIONS.map((c, i) => {
          const appearAt = 100 + i * 280;
          const op = interpolate(frame, [appearAt, appearAt + 90], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const x = interpolate(frame, [appearAt, appearAt + 90], [-80, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                opacity: op,
                transform: `translateX(${x}px)`,
                padding: "24px 30px",
                background: `${c.color}12`,
                border: `1px solid ${c.color}66`,
                borderLeft: `6px solid ${c.color}`,
                borderRadius: 10,
                display: "flex",
                gap: 30,
                alignItems: "center",
              }}
            >
              <div
                style={{
                  fontSize: 88,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: c.color,
                  letterSpacing: -2,
                  minWidth: 120,
                }}
              >
                {c.id}
              </div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 14,
                    letterSpacing: 6,
                    color: c.color,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    textTransform: "uppercase",
                  }}
                >
                  {c.tag}
                </div>
                <div
                  style={{
                    fontSize: 28,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    color: "#fff",
                    letterSpacing: -0.5,
                    marginTop: 4,
                  }}
                >
                  {c.title}
                </div>
                <div
                  style={{
                    fontSize: 18,
                    fontFamily: "Inter, system-ui",
                    color: TOKENS.colors.text.secondary,
                    marginTop: 8,
                    lineHeight: 1.45,
                  }}
                >
                  {c.body}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
