import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  staticFile,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import { PITCH_METRICS } from "../../data/metrics";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T15 · Climax Replay · 90s (2700 frames)
 *
 * 0-60f     : fade in replay
 * 60-1500f  : replay plays with month counter + title
 * 1500-1800f: tension build (vignette closes, brightness drops)
 * 1800-2100f: staggered number reveals
 * 2100-2400f: lock caption
 * 2400-2700f: fade to post-climax
 */
export const T15_ClimaxReplay: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stats = PITCH_METRICS.oos_2025;

  const videoOpacity = interpolate(frame, [0, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleOpacity = interpolate(
    frame,
    [30, 90, 1500, 1600],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const vignetteStrength = interpolate(frame, [1500, 1800], [0, 0.85], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const videoBrightness = interpolate(frame, [1500, 1800], [1, 0.35], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const returnSpring = spring({
    frame: frame - 1800,
    fps,
    config: { damping: 12, stiffness: 90, mass: 1 },
  });
  const sharpeSpring = spring({
    frame: frame - 1900,
    fps,
    config: { damping: 14, stiffness: 100, mass: 0.9 },
  });
  const pvalueSpring = spring({
    frame: frame - 2000,
    fps,
    config: { damping: 14, stiffness: 110, mass: 0.9 },
  });

  const returnValue = interpolate(frame, [1800, 2000], [0, 25.63], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const captionOpacity = interpolate(frame, [2100, 2200], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [2600, 2700], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const approxMonth = Math.min(12, Math.floor(1 + (frame / 1500) * 11));
  const monthName = [
    "ENE",
    "FEB",
    "MAR",
    "ABR",
    "MAY",
    "JUN",
    "JUL",
    "AGO",
    "SEP",
    "OCT",
    "NOV",
    "DIC",
  ][Math.max(0, Math.min(11, approxMonth - 1))];

  return (
    <AbsoluteFill style={{ background: "#000", opacity: exitFade }}>
      {/* Replay video */}
      <AbsoluteFill style={{ opacity: videoOpacity }}>
        <OffthreadVideo
          src={staticFile("captures/S03-replay.webm")}
          muted
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: `brightness(${videoBrightness})`,
          }}
        />
      </AbsoluteFill>

      {/* Vignette */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 50%, transparent 25%, rgba(0,0,0,${vignetteStrength}) 90%)`,
          pointerEvents: "none",
        }}
      />

      {/* Top gradient */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.72) 0%, transparent 18%, transparent 75%, rgba(0,0,0,0.85) 100%)",
          pointerEvents: "none",
        }}
      />

      {/* Top-left title */}
      <div
        style={{
          position: "absolute",
          top: 36,
          left: 56,
          opacity: titleOpacity,
          display: "flex",
          flexDirection: "column",
          gap: 6,
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
          CLIMAX · SECCIÓN IV
        </div>
        <div
          style={{
            fontSize: 44,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -1,
            textShadow: "0 2px 10px rgba(0,0,0,0.9)",
          }}
        >
          Backtest 2025 · Out-of-Sample
        </div>
        <div
          style={{
            fontSize: 20,
            fontFamily: "JetBrains Mono, monospace",
            color: TOKENS.colors.text.secondary,
            letterSpacing: 2,
            marginTop: 4,
          }}
        >
          Réplica bar-by-bar · 2025-01 → 2025-12
        </div>
      </div>

      {/* Month counter top-right */}
      <div
        style={{
          position: "absolute",
          top: 44,
          right: 56,
          opacity: titleOpacity,
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 72,
          fontWeight: 700,
          color: TOKENS.colors.market.up,
          letterSpacing: 4,
          textShadow: "0 4px 16px rgba(0,211,149,0.5)",
        }}
      >
        {monthName} 2025
      </div>

      {/* Number reveals */}
      {frame >= 1800 && (
        <AbsoluteFill
          style={{
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 40,
          }}
        >
          <div
            style={{
              transform: `scale(${returnSpring})`,
              opacity: returnSpring,
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontSize: 36,
                color: TOKENS.colors.text.secondary,
                letterSpacing: 10,
                fontFamily: "Inter, system-ui",
                marginBottom: 8,
                textTransform: "uppercase",
              }}
            >
              Retorno
            </div>
            <div
              style={{
                fontSize: 220,
                fontFamily: "JetBrains Mono, monospace",
                fontWeight: 700,
                lineHeight: 0.95,
                background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.market.up})`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                letterSpacing: -6,
                filter: "drop-shadow(0 6px 24px rgba(0,211,149,0.55))",
              }}
            >
              +{returnValue.toFixed(2)}%
            </div>
          </div>

          <div style={{ display: "flex", gap: 120, marginTop: 32 }}>
            <div
              style={{
                opacity: sharpeSpring,
                transform: `translateY(${(1 - sharpeSpring) * 30}px)`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 22,
                  color: TOKENS.colors.text.secondary,
                  letterSpacing: 6,
                  fontFamily: "Inter, system-ui",
                  textTransform: "uppercase",
                }}
              >
                Sharpe
              </div>
              <div
                style={{
                  fontSize: 96,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: "#fff",
                  letterSpacing: -2,
                  marginTop: 8,
                }}
              >
                {stats.sharpe.toFixed(2)}
              </div>
            </div>

            <div
              style={{
                opacity: pvalueSpring,
                transform: `translateY(${(1 - pvalueSpring) * 30}px)`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 22,
                  color: TOKENS.colors.text.secondary,
                  letterSpacing: 6,
                  fontFamily: "Inter, system-ui",
                  textTransform: "uppercase",
                }}
              >
                p-value
              </div>
              <div
                style={{
                  fontSize: 96,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: TOKENS.colors.market.up,
                  letterSpacing: -2,
                  marginTop: 8,
                }}
              >
                {stats.p_value.toFixed(3)}
              </div>
            </div>
          </div>

          <div
            style={{
              opacity: captionOpacity,
              marginTop: 40,
              padding: "14px 32px",
              background: "rgba(0,211,149,0.15)",
              border: `1px solid ${TOKENS.colors.market.up}`,
              borderRadius: 10,
              fontSize: 26,
              fontFamily: "Inter, system-ui",
              fontWeight: 600,
              color: TOKENS.colors.market.up,
              letterSpacing: 2,
              textTransform: "uppercase",
            }}
          >
            Validación estadística significativa · N = 34 trades
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
