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
import { OOSBadge } from "../../components/OOSBadge";
import { MetricCounter } from "../../components/MetricCounter";
import { PITCH_METRICS } from "../../data/metrics";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P03 · Dashboard (22s · 660 frames) — MONEY SHOT · Replay completo
 *
 * The S03-replay.webm has been re-captured with choreographed scrolling
 * (controls → chart up → scroll down to equity → back up → KPIs top) over
 * 22 seconds while the backtest replay animates.
 *
 * Stages:
 *   0–60f    (0-2s)   : OOS badge intro
 *   60–570f  (2-19s)  : S03-replay plays full-bleed with native scroll motion
 *   570–660f (19-22s) : Final metrics count-up on dark overlay
 */
export const P03_Dashboard: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();
  const m = PITCH_METRICS.oos_2025;

  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
      {/* Background: always show the replay capture */}
      <AbsoluteFill>
        <OffthreadVideo
          src={staticFile("captures/S03-replay.webm")}
          muted
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            filter: "brightness(0.95)",
          }}
        />
      </AbsoluteFill>

      {/* Stage 1 (0-75f): OOS badge intro — overlays the replay start */}
      <Sequence from={0} durationInFrames={80} layout="none">
        <AbsoluteFill
          style={{
            background: "rgba(5, 8, 22, 0.55)",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 28,
            opacity: interpolate(frame, [65, 80], [1, 0], {
              extrapolateRight: "clamp",
              extrapolateLeft: "clamp",
            }),
          }}
        >
          <OOSBadge label="Caso 1 · Validación histórica OOS" delay={2} />
          <div
            style={{
              fontSize: 42,
              fontWeight: 700,
              color: TOKENS.colors.text.primary,
              fontFamily: "Inter, system-ui",
              textShadow: "0 2px 20px rgba(0,0,0,0.8)",
              maxWidth: 1200,
              textAlign: "center",
              lineHeight: 1.3,
            }}
          >
            Backtest reproducible · mercado 2025 bar-a-bar.
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* Stage 2 (60-570f): Replay runs full-bleed with a persistent lower-third label */}
      <Sequence from={60} durationInFrames={510} layout="none">
        <div
          style={{
            position: "absolute",
            left: 72,
            bottom: 56,
            padding: "14px 22px",
            background: "rgba(0,0,0,0.72)",
            border: `1px solid ${TOKENS.colors.accent.cyan}`,
            borderRadius: 10,
            color: TOKENS.colors.accent.cyan,
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 22,
            letterSpacing: 2,
            boxShadow: `0 0 20px ${TOKENS.colors.accent.cyan}30`,
            opacity: interpolate(
              frame,
              [60, 85, 540, 570],
              [0, 1, 1, 0],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            ),
          }}
        >
          ▶ Replay · mercado 2025 reproducido bar-a-bar
        </div>

        {/* Subtle vignette for text legibility at bottom */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(180deg, transparent 60%, rgba(0,0,0,0.3) 100%)",
            pointerEvents: "none",
          }}
        />
      </Sequence>

      {/* Stage 3 (570-660f): Metrics count-up on dark overlay */}
      <Sequence from={570} layout="none">
        <AbsoluteFill
          style={{
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 36,
            background: `radial-gradient(ellipse at center, ${TOKENS.colors.bg.deep}f2, ${TOKENS.colors.bg.primary}f5)`,
            opacity: interpolate(frame, [570, 590], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <OOSBadge delay={0} label="Resultados · Backtest OOS 2025" />
          <div style={{ display: "flex", gap: 80, alignItems: "center" }}>
            <MetricCounter
              target={m.return_pct}
              format="pct"
              digits={2}
              durationInFrames={50}
              delay={8}
              label="Retorno"
              fontSize={108}
              color={TOKENS.colors.market.up}
              labelSize={20}
            />
            <MetricCounter
              target={m.sharpe}
              format="ratio"
              digits={2}
              durationInFrames={50}
              delay={24}
              label="Sharpe Ratio"
              fontSize={108}
            />
            <MetricCounter
              target={m.p_value}
              format="pvalue"
              durationInFrames={50}
              delay={40}
              label="Significancia"
              fontSize={88}
              color={TOKENS.colors.accent.cyan}
            />
          </div>
          <div
            style={{
              fontSize: 22,
              color: TOKENS.colors.text.secondary,
              fontFamily: "Inter, system-ui",
              letterSpacing: 4,
              textTransform: "uppercase",
              marginTop: 12,
            }}
          >
            {m.trades_total} trades · {m.win_rate}% win rate · MaxDD {m.max_dd_pct}%
          </div>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
