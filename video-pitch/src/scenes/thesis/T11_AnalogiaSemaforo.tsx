import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T11 · Analogía del semáforo · 20s (600f)
 * Three traffic-light circles illuminated sequentially, with captions.
 */
export const T11_AnalogiaSemaforo: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const redLight = interpolate(frame, [60, 120, 300, 360], [0.1, 1, 1, 0.1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const yellowLight = interpolate(
    frame,
    [180, 240, 420, 480],
    [0.1, 1, 1, 0.1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const greenLight = interpolate(frame, [300, 360, 560, 600], [0.1, 1, 1, 0.3], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [570, 600], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const textOp = (showAt: number) =>
    interpolate(frame, [showAt, showAt + 40], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });

  return (
    <AbsoluteFill
      style={{
        background: "#0b1020",
        opacity: exitFade,
      }}
    >
      {/* Fog gradient */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at 50% 60%, rgba(30,41,59,0.3) 0%, rgba(15,23,42,0.8) 70%)",
          pointerEvents: "none",
        }}
      />

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
            color: TOKENS.colors.semantic.warning,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO III · ANALOGÍA
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
          Un semáforo en la niebla
        </div>
      </div>

      {/* Traffic light */}
      <div
        style={{
          position: "absolute",
          left: 200,
          top: 280,
          width: 220,
          height: 580,
          background: "#1e293b",
          borderRadius: 24,
          border: "3px solid #334155",
          padding: 22,
          display: "flex",
          flexDirection: "column",
          gap: 18,
        }}
      >
        <Light color={TOKENS.colors.market.down} opacity={redLight} />
        <Light color={TOKENS.colors.semantic.warning} opacity={yellowLight} />
        <Light color={TOKENS.colors.market.up} opacity={greenLight} />
      </div>

      {/* Captions */}
      <div
        style={{
          position: "absolute",
          left: 520,
          top: 300,
          display: "flex",
          flexDirection: "column",
          gap: 42,
          width: 1120,
        }}
      >
        <Caption
          op={textOp(120)}
          color={TOKENS.colors.market.down}
          label="ROJO · Mean-reverting"
          body="Hurst < 0.42. El mercado revierte a la media. El régimen gate ORDENA no operar."
        />
        <Caption
          op={textOp(240)}
          color={TOKENS.colors.semantic.warning}
          label="AMARILLO · Indeterminate"
          body="Hurst 0.42–0.52. Reducción de sizing 40%. Stop efectivo y apalancamiento dinámico asumen el resto."
        />
        <Caption
          op={textOp(360)}
          color={TOKENS.colors.market.up}
          label="VERDE · Trending"
          body="Hurst > 0.52. El mercado extiende tendencia. Sizing completo. El modelo predice con ventaja."
        />
      </div>

      {/* Bottom line */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          fontFamily: "Inter, system-ui",
          fontStyle: "italic",
          fontSize: 22,
          color: TOKENS.colors.text.secondary,
          textAlign: "center",
          letterSpacing: 0.5,
          opacity: textOp(440),
        }}
      >
        “Saber cuándo no cruzar es la primera forma de inteligencia.”
      </div>
    </AbsoluteFill>
  );
};

const Light: React.FC<{ color: string; opacity: number }> = ({
  color,
  opacity,
}) => (
  <div
    style={{
      flex: 1,
      borderRadius: "50%",
      background: color,
      opacity,
      boxShadow: `0 0 ${60 * opacity}px ${color}`,
      border: "2px solid rgba(255,255,255,0.05)",
    }}
  />
);

const Caption: React.FC<{
  op: number;
  color: string;
  label: string;
  body: string;
}> = ({ op, color, label, body }) => (
  <div style={{ opacity: op }}>
    <div
      style={{
        fontSize: 16,
        letterSpacing: 5,
        color,
        fontFamily: "Inter, system-ui",
        fontWeight: 700,
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
    <div
      style={{
        fontSize: 26,
        fontFamily: "Inter, system-ui",
        color: "#fff",
        marginTop: 6,
        lineHeight: 1.35,
      }}
    >
      {body}
    </div>
  </div>
);
