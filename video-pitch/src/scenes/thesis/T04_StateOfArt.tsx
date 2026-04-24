import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { STATE_OF_ART } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T04 · Estado del arte · 40s (1200f)
 * Citation cards for 5 foundational references.
 */
export const T04_StateOfArt: React.FC<ThesisProps> = ({ variant: _variant }) => {
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
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          MARCO TEÓRICO · REFERENCIAS FUNDACIONALES
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
          Sobre los hombros de gigantes
        </div>
      </div>

      {/* Cards */}
      <div
        style={{
          position: "absolute",
          top: 200,
          left: 80,
          right: 80,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {STATE_OF_ART.map((cite, idx) => {
          const appearAt = 120 + idx * 180;
          const lingerUntil = appearAt + 120 * 5;
          const op = interpolate(
            frame,
            [appearAt, appearAt + 60, lingerUntil - 60, lingerUntil],
            [0, 1, 1, 0.7],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const x = interpolate(frame, [appearAt, appearAt + 60], [-60, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={idx}
              style={{
                opacity: op,
                transform: `translateX(${x}px)`,
                padding: "20px 28px",
                background: "rgba(15,23,42,0.7)",
                border: `1px solid ${TOKENS.colors.accent.purple}55`,
                borderRadius: 10,
                display: "flex",
                alignItems: "center",
                gap: 24,
              }}
            >
              <div
                style={{
                  width: 56,
                  height: 56,
                  borderRadius: 8,
                  background: TOKENS.colors.accent.purple + "33",
                  border: `1px solid ${TOKENS.colors.accent.purple}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 24,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: TOKENS.colors.accent.purple,
                }}
              >
                {idx + 1}
              </div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 22,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    color: "#fff",
                    letterSpacing: -0.3,
                  }}
                >
                  {cite.title}
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: "JetBrains Mono, monospace",
                    color: TOKENS.colors.accent.cyan,
                    marginTop: 4,
                  }}
                >
                  {cite.authors} · {cite.year} · {cite.venue}
                </div>
                <div
                  style={{
                    fontSize: 15,
                    fontFamily: "Inter, system-ui",
                    color: TOKENS.colors.text.secondary,
                    marginTop: 8,
                    fontStyle: "italic",
                  }}
                >
                  {cite.relevance}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer line */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          fontSize: 15,
          fontFamily: "Inter, system-ui",
          color: TOKENS.colors.text.muted,
          textAlign: "center",
          letterSpacing: 2,
          opacity: interpolate(frame, [1000, 1080], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Chicago style · bibliografía extendida en apéndice A
      </div>
    </AbsoluteFill>
  );
};
