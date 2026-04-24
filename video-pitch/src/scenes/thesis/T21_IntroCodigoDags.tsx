import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T21 · Intro Código DAGs · 15s (450f)
 * Transition into Acto V (the 10-DAG code tour).
 */
export const T21_IntroCodigoDags: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const chapterOp = interpolate(frame, [0, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const headlineOp = interpolate(frame, [90, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const captionOp = interpolate(frame, [180, 270], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const countOp = interpolate(frame, [270, 360], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [420, 450], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: TOKENS.colors.bg.deep,
        opacity: exitFade,
      }}
    >
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 45%, ${TOKENS.colors.accent.cyan}11 0%, transparent 60%)`,
          pointerEvents: "none",
        }}
      />

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 28,
          padding: "0 120px",
        }}
      >
        <div
          style={{
            fontSize: 18,
            letterSpacing: 10,
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            opacity: chapterOp,
            textTransform: "uppercase",
          }}
        >
          CAPÍTULO V · Código reproducible
        </div>

        <div
          style={{
            fontSize: 78,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            textAlign: "center",
            letterSpacing: -2,
            lineHeight: 1.1,
            opacity: headlineOp,
          }}
        >
          Cada decisión tiene <br />
          un archivo Python detrás.
        </div>

        <div
          style={{
            fontSize: 24,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            textAlign: "center",
            maxWidth: 1200,
            letterSpacing: 0.5,
            opacity: captionOp,
            lineHeight: 1.5,
          }}
        >
          Airflow no es un detalle de implementación. Es la materialización del
          método científico en producción: tareas versionadas, horarios
          explícitos, trazabilidad completa.
        </div>

        <div
          style={{
            display: "flex",
            gap: 40,
            marginTop: 20,
            opacity: countOp,
          }}
        >
          <Stat value="10" label="DAGs auditados" />
          <Stat value="~22s" label="Por DAG · código completo" />
          <Stat value="4:40" label="Deep tour total" />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const Stat: React.FC<{ value: string; label: string }> = ({ value, label }) => (
  <div style={{ textAlign: "center" }}>
    <div
      style={{
        fontSize: 56,
        fontFamily: "JetBrains Mono, monospace",
        fontWeight: 700,
        color: TOKENS.colors.accent.cyan,
        letterSpacing: -1,
      }}
    >
      {value}
    </div>
    <div
      style={{
        fontSize: 14,
        fontFamily: "Inter, system-ui",
        color: TOKENS.colors.text.secondary,
        letterSpacing: 3,
        textTransform: "uppercase",
        marginTop: 4,
      }}
    >
      {label}
    </div>
  </div>
);
