import React from "react";
import {
  AbsoluteFill,
  Sequence,
  staticFile,
  OffthreadVideo,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import { Typewriter } from "../../components/Typewriter";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P02 · Inicio (10s · 300 frames)
 *
 * The S01-login.webm now contains the FULL login flow:
 *   - type credentials (admin / admin123)
 *   - submit
 *   - redirect to /hub
 *   - 6 módulos visible in the destination
 *
 * Playback strategy: let the clip speak for itself with a minimal overlay
 * "Caso · Inicio" at the top + "6 módulos integrados" at the bottom once
 * the hub is visible.
 */
export const P02_Inicio: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.primary }}>
      {/* Full login flow — play the captured clip full-bleed */}
      <AbsoluteFill>
        <OffthreadVideo
          src={staticFile("captures/S01-login.webm")}
          muted
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            // Subtle zoom out so login UI starts large and hub shows full
            transform: `scale(${interpolate(frame, [0, 300], [1.03, 1.0])})`,
            transformOrigin: "center center",
          }}
        />
      </AbsoluteFill>

      {/* Top gradient bar for title readability */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, transparent 18%)",
          pointerEvents: "none",
        }}
      />

      {/* Top overlay: "Caso · Inicio" */}
      <Sequence from={0} durationInFrames={300} layout="none">
        <div
          style={{
            position: "absolute",
            top: 40,
            left: 60,
            display: "flex",
            alignItems: "center",
            gap: 14,
            padding: "10px 18px",
            background: "rgba(0,0,0,0.65)",
            border: `1px solid ${TOKENS.colors.accent.cyan}66`,
            borderRadius: 999,
            opacity: interpolate(frame, [0, 15], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: 999,
              background: TOKENS.colors.accent.cyan,
              boxShadow: `0 0 12px ${TOKENS.colors.accent.cyan}`,
            }}
          />
          <span
            style={{
              fontSize: 18,
              color: TOKENS.colors.accent.cyan,
              letterSpacing: 4,
              textTransform: "uppercase",
              fontFamily: "Inter, system-ui",
              fontWeight: 600,
            }}
          >
            Inicio · Acceso seguro
          </span>
        </div>
      </Sequence>

      {/* Bottom overlay appears once the hub is visible (~after 7s = 210f) */}
      <Sequence from={210} durationInFrames={90} layout="none">
        <AbsoluteFill
          style={{
            alignItems: "center",
            justifyContent: "flex-end",
            paddingBottom: 60,
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 10,
              padding: "16px 32px",
              background: "rgba(0,0,0,0.82)",
              borderRadius: 14,
              border: `1px solid ${TOKENS.colors.accent.cyan}55`,
              boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
              opacity: interpolate(frame - 210, [0, 18], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <Typewriter
              text="Trading cuantitativo end-to-end"
              speed={30}
              delay={8}
              fontSize={38}
              fontWeight={700}
              gradient={[TOKENS.colors.accent.cyan, TOKENS.colors.accent.purple]}
              cursor={false}
            />
            <div
              style={{
                fontSize: 18,
                color: TOKENS.colors.text.secondary,
                letterSpacing: 4,
                textTransform: "uppercase",
                fontFamily: "Inter, system-ui",
              }}
            >
              6 módulos integrados
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
