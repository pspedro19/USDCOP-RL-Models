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
import { RegimeBadge } from "../../components/RegimeBadge";
import { WeekBlockingBars } from "../../components/WeekBlockingBars";
import { MetricCounter } from "../../components/MetricCounter";
import { PITCH_METRICS } from "../../data/metrics";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P04 · Producción (10s · 300 frames)
 *
 * New framing: "Caso 2 · Monitoreo en vivo + decisión por régimen"
 * (reframed from 2026-specific event to generic platform capability)
 *
 *   0–80f    (0-2.7s): split screen — live monitoring + regime detection
 *   80–220f  (2.7-7.3s): WeekBlockingBars staggered reveal
 *   220–300f (7.3-10s): Comparison +0.61% vs -2.82%
 */
export const P04_Produccion: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();
  const m = PITCH_METRICS.ytd_2026;
  const rg = PITCH_METRICS.regime_gate;

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(145deg, ${TOKENS.colors.bg.primary}, ${TOKENS.colors.bg.deep})`,
      }}
    >
      {/* Stage 1 (0-85f): split screen intro */}
      <Sequence from={0} durationInFrames={90} layout="none">
        <AbsoluteFill
          style={{
            display: "flex",
            flexDirection: "row",
            opacity: interpolate(frame, [75, 88], [1, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
            <OffthreadVideo
              src={staticFile("captures/S05-production-live.webm")}
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                filter: "brightness(0.75)",
              }}
            />
            <div
              style={{
                position: "absolute",
                bottom: 40,
                left: 40,
                padding: "10px 18px",
                background: "rgba(0,0,0,0.7)",
                border: `1px solid ${TOKENS.colors.market.up}`,
                color: TOKENS.colors.market.up,
                borderRadius: 6,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 18,
                letterSpacing: 2,
              }}
            >
              ● LIVE · /production
            </div>
          </div>
          <div
            style={{
              flex: 1,
              background: TOKENS.colors.bg.deep,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              gap: 28,
              padding: 56,
            }}
          >
            <div
              style={{
                fontSize: 22,
                color: TOKENS.colors.accent.cyan,
                letterSpacing: 5,
                textTransform: "uppercase",
                fontFamily: "Inter, system-ui",
                fontWeight: 600,
              }}
            >
              Caso 2 · Monitoreo + régimen gate
            </div>
            <RegimeBadge regime="mean-reverting" hurst={rg.hurst_2026_q1} delay={12} />
            <div
              style={{
                fontSize: 28,
                color: TOKENS.colors.text.primary,
                fontFamily: "Inter, system-ui",
                fontWeight: 600,
                textAlign: "center",
                lineHeight: 1.35,
                maxWidth: 620,
              }}
            >
              Monitoreo en vivo con decisión automática por régimen.
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* Stage 2 (85-220f): WeekBlockingBars */}
      <Sequence from={85} durationInFrames={140} layout="none">
        <AbsoluteFill
          style={{
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 32,
            opacity: interpolate(frame, [85, 105, 200, 220], [0, 1, 1, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <div
            style={{
              fontSize: 28,
              color: TOKENS.colors.text.secondary,
              fontFamily: "Inter, system-ui",
              letterSpacing: 4,
              textTransform: "uppercase",
            }}
          >
            Semanas filtradas por régimen · Q1
          </div>
          <WeekBlockingBars
            totalWeeks={m.total_weeks}
            blockedWeeks={m.gate_blocked_weeks}
            delay={10}
            staggerFrames={7}
          />
          <div style={{ marginTop: 32, display: "flex", gap: 48, alignItems: "baseline" }}>
            <div
              style={{
                fontSize: 60,
                fontFamily: "JetBrains Mono, monospace",
                color: TOKENS.colors.semantic.danger,
                fontWeight: 800,
              }}
            >
              {m.gate_blocked_weeks} / {m.total_weeks}
            </div>
            <div
              style={{
                fontSize: 22,
                color: TOKENS.colors.text.primary,
                fontFamily: "Inter, system-ui",
                letterSpacing: 3,
                textTransform: "uppercase",
              }}
            >
              decisión automática de no operar
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* Stage 3 (220-300f): comparison */}
      <Sequence from={220} layout="none">
        <AbsoluteFill
          style={{
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 28,
            background: `radial-gradient(ellipse at center, ${TOKENS.colors.bg.primary}ee, ${TOKENS.colors.bg.deep}ee)`,
          }}
        >
          <div
            style={{
              fontSize: 24,
              color: TOKENS.colors.text.secondary,
              letterSpacing: 4,
              textTransform: "uppercase",
              fontFamily: "Inter, system-ui",
            }}
          >
            Resultado · Q1 2026 en vivo
          </div>
          <div style={{ display: "flex", gap: 100, alignItems: "center" }}>
            <MetricCounter
              target={m.return_pct}
              format="pct"
              digits={2}
              delay={5}
              label="Con gate"
              fontSize={108}
              color={TOKENS.colors.market.up}
              labelSize={18}
            />
            <div
              style={{
                fontSize: 48,
                color: TOKENS.colors.text.muted,
                fontFamily: "Inter, system-ui",
              }}
            >
              vs
            </div>
            <MetricCounter
              target={m.buy_and_hold_pct}
              format="pct"
              digits={2}
              delay={20}
              label="Buy & Hold"
              fontSize={108}
              color={TOKENS.colors.market.down}
              labelSize={18}
            />
          </div>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
