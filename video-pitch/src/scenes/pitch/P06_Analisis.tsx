import React from "react";
import {
  AbsoluteFill,
  staticFile,
  OffthreadVideo,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P06 · Análisis (10s · 300 frames)
 *
 * Design: video breathes with multi-week navigation visible.
 * Only small corner labels (no large centered overlay).
 *
 * Background is S06 navigating 3 weeks of macro analysis. startFrom=120
 * skips initial load, viewers see the week navigation + scroll.
 */
export const P06_Analisis: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
      {/* Background — full brightness */}
      <AbsoluteFill>
        <OffthreadVideo
          src={staticFile("captures/S06-analysis-chat.webm")}
          muted
          startFrom={120}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: "brightness(0.95)",
          }}
        />
      </AbsoluteFill>

      {/* Gradient edges only */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, transparent 15%, transparent 82%, rgba(0,0,0,0.55) 100%)",
          pointerEvents: "none",
        }}
      />

      {/* TOP-LEFT badge */}
      <div
        style={{
          position: "absolute",
          top: 32,
          left: 48,
          display: "flex",
          flexDirection: "column",
          gap: 6,
          opacity: interpolate(frame, [0, 15], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div
          style={{
            fontSize: 14,
            color: TOKENS.colors.accent.purple,
            letterSpacing: 5,
            textTransform: "uppercase",
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          Caso 4 · IA Semanal
        </div>
        <div
          style={{
            fontSize: 28,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            background: `linear-gradient(135deg, ${TOKENS.colors.accent.purple}, ${TOKENS.colors.accent.cyan})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Análisis macro · GPT-4o
        </div>
      </div>

      {/* BOTTOM-CENTER footer: tiny stats */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: 36,
          padding: "12px 26px",
          background: "rgba(0,0,0,0.75)",
          borderRadius: 12,
          border: `1px solid ${TOKENS.colors.accent.purple}44`,
          backdropFilter: "blur(6px)",
          opacity: interpolate(frame, [25, 45], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {[
          { v: "16", l: "Semanas" },
          { v: "128", l: "Gráficos" },
          { v: "13", l: "Variables macro" },
          { v: "5", l: "Días / semana" },
        ].map((s) => (
          <div
            key={s.l}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 2,
            }}
          >
            <div
              style={{
                fontSize: 28,
                fontWeight: 800,
                fontFamily: "JetBrains Mono, monospace",
                color: TOKENS.colors.accent.cyan,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {s.v}
            </div>
            <div
              style={{
                fontSize: 11,
                color: TOKENS.colors.text.secondary,
                letterSpacing: 2,
                textTransform: "uppercase",
                fontFamily: "Inter, system-ui",
              }}
            >
              {s.l}
            </div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
