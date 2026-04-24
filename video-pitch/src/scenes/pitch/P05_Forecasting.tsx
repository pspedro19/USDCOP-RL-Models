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
 * P05 · Forward Forecast + Backtest (11s · 330 frames)
 *
 * Design: video breathes — NO large overlay covering the center.
 * Only small corner badges (top + bottom) identify the scene.
 *
 * Background video S07 is a multi-week navigation (Forward W16 → switch
 * to Backtest → cycle Ridge/XGBoost models). startFrom=90 skips the
 * initial page load so viewers see the navigation from the start.
 */
export const P05_Forecasting: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
      {/* Background video — FULL BRIGHTNESS, no heavy filter */}
      <AbsoluteFill>
        <OffthreadVideo
          src={staticFile("captures/S07-forecasting-zoo.webm")}
          muted
          startFrom={90}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: "brightness(0.95)",
          }}
        />
      </AbsoluteFill>

      {/* Subtle top gradient only (for badge legibility) */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, transparent 15%, transparent 82%, rgba(0,0,0,0.55) 100%)",
          pointerEvents: "none",
        }}
      />

      {/* TOP-LEFT badge: Caso 3 · Forward Forecast */}
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
            color: TOKENS.colors.accent.cyan,
            letterSpacing: 5,
            textTransform: "uppercase",
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          Caso 3
        </div>
        <div
          style={{
            fontSize: 28,
            color: TOKENS.colors.text.primary,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.accent.purple})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Forward Forecast · Backtest
        </div>
      </div>

      {/* BOTTOM-CENTER footer: stats compact */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: 40,
          padding: "12px 26px",
          background: "rgba(0,0,0,0.75)",
          borderRadius: 12,
          border: `1px solid ${TOKENS.colors.accent.cyan}44`,
          backdropFilter: "blur(6px)",
          opacity: interpolate(frame, [20, 40], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {[
          { v: "9", l: "Modelos ML", color: TOKENS.colors.accent.cyan },
          { v: "63", l: "Backtests WF", color: TOKENS.colors.accent.purple },
          { v: "7", l: "Horizontes", color: TOKENS.colors.market.up },
          { v: "3", l: "Ensembles", color: TOKENS.colors.semantic.warning },
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
                color: s.color,
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
