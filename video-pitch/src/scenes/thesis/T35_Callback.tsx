import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import { THESIS_COLORS, THESIS_AUTHOR } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T35 · Callback + Credits · 25s (750f)
 *
 * 0-180f   : Three lines appear, echoing T01 cold open
 *              "El modelo predijo mal."  (faded red, same position)
 *              "Y aún así, ganó."         (bright green)
 *              "Porque aprendió cuándo callarse." (cyan, new line)
 * 180-420f : Lines lock, author card springs in
 * 420-660f : FIUBA badge + Chicago credit line
 * 660-750f : Fade to T36 silence
 */
export const T35_Callback: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Echo lines
  const line1Op = interpolate(frame, [0, 45], [0, 0.85], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line2Op = interpolate(frame, [50, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line3Spring = spring({
    frame: frame - 130,
    fps,
    config: { damping: 16, stiffness: 110, mass: 0.9 },
  });

  // Author card
  const authorOp = interpolate(frame, [220, 310], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const authorY = interpolate(frame, [220, 310], [40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // FIUBA badge
  const fiubaOp = interpolate(frame, [460, 540], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Bottom disclaimer
  const discOp = interpolate(frame, [540, 620], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Exit fade
  const exitFade = interpolate(frame, [690, 750], [1, 0], {
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
      {/* Subtle brand gradient */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 30%, ${THESIS_COLORS.fiuba_blue}22 0%, transparent 60%)`,
          pointerEvents: "none",
        }}
      />

      {/* Stack */}
      <div
        style={{
          position: "absolute",
          top: 140,
          left: 0,
          right: 0,
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          gap: 18,
          padding: "0 80px",
        }}
      >
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            fontSize: 60,
            letterSpacing: -1,
            color: TOKENS.colors.market.down,
            opacity: line1Op,
          }}
        >
          El modelo predijo mal.
        </div>
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            fontSize: 72,
            letterSpacing: -2,
            color: TOKENS.colors.market.up,
            textShadow: "0 0 24px rgba(0,211,149,0.55)",
            opacity: line2Op,
          }}
        >
          Y aún así, ganó.
        </div>
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            fontSize: 48,
            letterSpacing: -0.5,
            background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.accent.purple})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            opacity: line3Spring,
            transform: `translateY(${(1 - line3Spring) * 30}px)`,
            marginTop: 8,
          }}
        >
          Porque aprendió cuándo callarse.
        </div>
      </div>

      {/* Author card */}
      <div
        style={{
          position: "absolute",
          bottom: 260,
          left: "50%",
          transform: `translate(-50%, ${authorY}px)`,
          opacity: authorOp,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: 13,
            letterSpacing: 6,
            color: TOKENS.colors.text.secondary,
            fontFamily: "Inter, system-ui",
            fontWeight: 600,
            textTransform: "uppercase",
            marginBottom: 6,
          }}
        >
          Tesis de grado
        </div>
        <div
          style={{
            fontSize: 34,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            color: "#fff",
            letterSpacing: -0.5,
          }}
        >
          {THESIS_AUTHOR.full_name}
        </div>
        <div
          style={{
            fontSize: 18,
            fontFamily: "JetBrains Mono, monospace",
            color: TOKENS.colors.text.secondary,
            marginTop: 8,
            letterSpacing: 1,
          }}
        >
          Director · {THESIS_AUTHOR.director}
        </div>
      </div>

      {/* FIUBA badge */}
      <div
        style={{
          position: "absolute",
          bottom: 140,
          left: "50%",
          transform: "translateX(-50%)",
          opacity: fiubaOp,
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "12px 28px",
          background: `${THESIS_COLORS.fiuba_blue}33`,
          border: `1px solid ${THESIS_COLORS.fiuba_blue}`,
          borderRadius: 999,
        }}
      >
        <div
          style={{
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: THESIS_COLORS.fiuba_blue,
            boxShadow: `0 0 12px ${THESIS_COLORS.fiuba_blue}`,
          }}
        />
        <div
          style={{
            fontSize: 18,
            fontFamily: "Inter, system-ui",
            fontWeight: 600,
            color: "#fff",
            letterSpacing: 1.5,
          }}
        >
          {THESIS_AUTHOR.institution}
        </div>
      </div>

      {/* Academic disclaimer */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: "50%",
          transform: "translateX(-50%)",
          opacity: discOp,
          fontSize: 14,
          fontFamily: "Inter, system-ui",
          color: TOKENS.colors.text.muted,
          letterSpacing: 1.2,
          textAlign: "center",
          maxWidth: 960,
        }}
      >
        Resultados corresponden a ejecuciones 2025-2026. No constituye asesoría
        financiera. Rendimientos pasados no garantizan resultados futuros.
      </div>
    </AbsoluteFill>
  );
};
