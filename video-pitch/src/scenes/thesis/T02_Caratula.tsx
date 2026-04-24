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
 * T02 · Carátula · 18s (540f)
 *
 * Title card in the style of an academic cover page:
 *   - FIUBA crest (dot mark, CC BY-SA 3.0 attribution in T36)
 *   - Thesis title + subtitle
 *   - Author + director
 *   - Institution + year
 */
export const T02_Caratula: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const crestSpring = spring({
    frame,
    fps,
    config: { damping: 14, stiffness: 90, mass: 0.9 },
  });
  const titleOp = interpolate(frame, [20, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const subtitleOp = interpolate(frame, [50, 100], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const authorOp = interpolate(frame, [90, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const instOp = interpolate(frame, [130, 190], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Horizontal rule draws
  const rulePct = interpolate(frame, [60, 120], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [510, 540], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(180deg, ${TOKENS.colors.bg.deep} 0%, ${TOKENS.colors.bg.secondary} 100%)`,
        opacity: exitFade,
      }}
    >
      {/* Subtle FIUBA wash */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 40%, ${THESIS_COLORS.fiuba_blue}22 0%, transparent 55%)`,
          pointerEvents: "none",
        }}
      />

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          textAlign: "center",
          padding: "0 120px",
          gap: 18,
        }}
      >
        {/* FIUBA mark */}
        <div
          style={{
            transform: `scale(${crestSpring})`,
            opacity: crestSpring,
            marginBottom: 12,
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 10,
              background: THESIS_COLORS.fiuba_blue,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: `0 0 32px ${THESIS_COLORS.fiuba_blue}99`,
              fontFamily: "Inter, system-ui",
              fontWeight: 800,
              fontSize: 24,
              color: "#fff",
              letterSpacing: -0.5,
            }}
          >
            ƒ
          </div>
          <div
            style={{
              fontSize: 14,
              letterSpacing: 6,
              color: TOKENS.colors.text.secondary,
              fontFamily: "Inter, system-ui",
              fontWeight: 700,
              textTransform: "uppercase",
            }}
          >
            FIUBA · Tesis de grado
          </div>
        </div>

        {/* Horizontal rule */}
        <div
          style={{
            width: 420,
            height: 2,
            background: `linear-gradient(90deg, transparent 0%, ${TOKENS.colors.accent.cyan} 50%, transparent 100%)`,
            transform: `scaleX(${rulePct / 100})`,
            transformOrigin: "center",
            marginBottom: 12,
          }}
        />

        {/* Title */}
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            fontSize: 62,
            letterSpacing: -1.5,
            lineHeight: 1.15,
            color: "#fff",
            opacity: titleOp,
            maxWidth: 1400,
          }}
        >
          {THESIS_AUTHOR.thesis_title}
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontFamily: "Inter, system-ui",
            fontWeight: 400,
            fontSize: 30,
            letterSpacing: 0,
            color: TOKENS.colors.text.secondary,
            opacity: subtitleOp,
            maxWidth: 1200,
            marginTop: 4,
            fontStyle: "italic",
          }}
        >
          {THESIS_AUTHOR.thesis_subtitle}
        </div>

        {/* Author + director */}
        <div
          style={{
            marginTop: 40,
            opacity: authorOp,
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div
            style={{
              fontSize: 13,
              letterSpacing: 6,
              color: TOKENS.colors.text.muted,
              fontFamily: "Inter, system-ui",
              fontWeight: 600,
              textTransform: "uppercase",
            }}
          >
            Autor
          </div>
          <div
            style={{
              fontSize: 38,
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
              fontSize: 20,
              fontFamily: "JetBrains Mono, monospace",
              color: TOKENS.colors.accent.cyan,
              letterSpacing: 1,
              marginTop: 6,
            }}
          >
            Director · {THESIS_AUTHOR.director}
          </div>
        </div>

        {/* Institution + year */}
        <div
          style={{
            marginTop: 32,
            opacity: instOp,
            fontSize: 16,
            fontFamily: "Inter, system-ui",
            color: TOKENS.colors.text.secondary,
            letterSpacing: 2.5,
            textTransform: "uppercase",
          }}
        >
          {THESIS_AUTHOR.institution} · {THESIS_AUTHOR.year}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
