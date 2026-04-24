import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { THESIS_LIMITATIONS } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T31 · Limitaciones · 45s (1350f)
 * 7 honest limitations revealed in two columns.
 */
export const T31_Limitaciones: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const footerOp = interpolate(frame, [1160, 1250], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1320, 1350], [1, 0], {
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
          top: 50,
          left: 80,
          opacity: titleOp,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 6,
            color: TOKENS.colors.semantic.warning,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO VI · AUTO-CRÍTICA
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
          Lo que esta tesis no puede probar
        </div>
      </div>

      {/* Grid of limitations */}
      <div
        style={{
          position: "absolute",
          top: 200,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        {THESIS_LIMITATIONS.map((l, i) => {
          const appearAt = 100 + i * 130;
          const op = interpolate(frame, [appearAt, appearAt + 60], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const y = interpolate(frame, [appearAt, appearAt + 60], [30, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                opacity: op,
                transform: `translateY(${y}px)`,
                padding: "16px 22px",
                background: "rgba(245,158,11,0.08)",
                border: `1px solid ${TOKENS.colors.semantic.warning}55`,
                borderRadius: 8,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 6,
                }}
              >
                <div
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: 6,
                    background: TOKENS.colors.semantic.warning,
                    color: "#000",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    fontFamily: "JetBrains Mono, monospace",
                    fontWeight: 700,
                  }}
                >
                  {i + 1}
                </div>
                <div
                  style={{
                    fontSize: 20,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    color: "#fff",
                  }}
                >
                  {l.issue}
                </div>
              </div>
              <div
                style={{
                  fontSize: 16,
                  fontFamily: "Inter, system-ui",
                  color: TOKENS.colors.text.secondary,
                  lineHeight: 1.45,
                }}
              >
                {l.detail}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          padding: "14px 24px",
          background: `${TOKENS.colors.accent.cyan}14`,
          border: `1px solid ${TOKENS.colors.accent.cyan}`,
          borderRadius: 10,
          opacity: footerOp,
          fontSize: 18,
          fontFamily: "Inter, system-ui",
          color: "#fff",
          fontStyle: "italic",
          textAlign: "center",
        }}
      >
        Reconocer estos límites no debilita la tesis — los convierte en el
        itinerario del trabajo futuro.
      </div>
    </AbsoluteFill>
  );
};
