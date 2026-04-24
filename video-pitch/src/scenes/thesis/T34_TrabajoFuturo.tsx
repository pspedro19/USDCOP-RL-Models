import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T34 · Trabajo futuro · 20s (600f)
 * 4 research axes for continuation.
 */
const AXES = [
  {
    id: "01",
    icon: "🌎",
    title: "Multi-par",
    body: "Generalización a USD/MXN y USD/BRL con datasets ya en el sistema.",
  },
  {
    id: "02",
    icon: "⏱️",
    title: "Intraday adaptativo",
    body: "H=1 con entrada/salida dinámica acoplada al régimen del día.",
  },
  {
    id: "03",
    icon: "🧠",
    title: "Transformer régimen-aware",
    body: "Arquitectura que conditioneda la predicción en el estado Hurst.",
  },
  {
    id: "04",
    icon: "📡",
    title: "Fusión con señales de noticias",
    body: "Pipeline de news engine (361 artículos) integrado como feature.",
  },
];

export const T34_TrabajoFuturo: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [570, 600], [1, 0], {
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
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO VI · TRABAJO FUTURO
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
          Cuatro ejes de continuación
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          top: 220,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
        }}
      >
        {AXES.map((a, i) => {
          const appearAt = 60 + i * 90;
          const op = interpolate(frame, [appearAt, appearAt + 60], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const scale = interpolate(
            frame,
            [appearAt, appearAt + 60],
            [0.9, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={a.id}
              style={{
                opacity: op,
                transform: `scale(${scale})`,
                padding: "28px 32px",
                background: `linear-gradient(135deg, ${TOKENS.colors.accent.purple}18, transparent)`,
                border: `1px solid ${TOKENS.colors.accent.purple}66`,
                borderRadius: 12,
                display: "flex",
                gap: 24,
                alignItems: "center",
                minHeight: 180,
              }}
            >
              <div
                style={{
                  fontSize: 72,
                  fontFamily: "Inter, system-ui",
                  fontWeight: 800,
                  color: TOKENS.colors.accent.purple,
                  letterSpacing: -2,
                  minWidth: 100,
                }}
              >
                {a.id}
              </div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 26,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    color: "#fff",
                    letterSpacing: -0.3,
                    marginBottom: 6,
                  }}
                >
                  {a.title}
                </div>
                <div
                  style={{
                    fontSize: 17,
                    fontFamily: "Inter, system-ui",
                    color: TOKENS.colors.text.secondary,
                    lineHeight: 1.4,
                  }}
                >
                  {a.body}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
