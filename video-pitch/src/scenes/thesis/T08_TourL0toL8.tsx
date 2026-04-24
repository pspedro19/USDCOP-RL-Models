import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  staticFile,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T08 · Tour montage L0→L8 · 90s (2700f)
 * Fast cuts through the dashboard sections while labeling each layer.
 * Reuses existing captures from the pitch.
 */
type Clip = {
  src: string;
  layer: string;
  title: string;
  desc: string;
  start: number;
  duration: number;
};

const CLIPS: Clip[] = [
  {
    src: "captures/S01-login.webm",
    layer: "L0 · ACCESO",
    title: "Auth + NextAuth",
    desc: "Gateway al sistema",
    start: 0,
    duration: 240,
  },
  {
    src: "captures/S02-hub.webm",
    layer: "L8 · HUB",
    title: "6 módulos",
    desc: "Dashboard · Producción · Forecasting · Análisis · SignalBridge",
    start: 240,
    duration: 300,
  },
  {
    src: "captures/S03-dashboard-scroll.webm",
    layer: "L4 · BACKTEST",
    title: "/dashboard · 2025 OOS",
    desc: "Gates 5/5 · +25.63% · Sharpe 3.35",
    start: 540,
    duration: 360,
  },
  {
    src: "captures/S05-production-live.webm",
    layer: "L5+L7 · PRODUCCIÓN",
    title: "/production · 2026 YTD",
    desc: "Regime gate activo · 13/14 semanas bloqueadas",
    start: 900,
    duration: 360,
  },
  {
    src: "captures/S07-forecasting-zoo.webm",
    layer: "L3 · FORWARD",
    title: "/forecasting · 9 modelos",
    desc: "Walk-forward · 7 horizontes · ensemble",
    start: 1260,
    duration: 360,
  },
  {
    src: "captures/S06-analysis-chat.webm",
    layer: "L8 · INTELLIGENCE",
    title: "/analysis · GPT-4o",
    desc: "16 semanas narrativa · chat contextual",
    start: 1620,
    duration: 300,
  },
  {
    src: "captures/S08-signalbridge.webm",
    layer: "L7 · SIGNAL BRIDGE",
    title: "/execution · OMS",
    desc: "MEXC CCXT · AES-256 · kill switch",
    start: 1920,
    duration: 420,
  },
  {
    src: "captures/I04-airflow-dags.webm",
    layer: "L0-L8 · ORQUESTACIÓN",
    title: "Airflow · 37 DAGs",
    desc: "Schedules UTC · sensors · guardrails",
    start: 2340,
    duration: 360,
  },
];

export const T08_TourL0toL8: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  // Find active clip
  const active = CLIPS.find(
    (c) => frame >= c.start && frame < c.start + c.duration
  );

  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {CLIPS.map((clip, i) => {
        const localFrame = frame - clip.start;
        const isInRange =
          frame >= clip.start - 10 && frame < clip.start + clip.duration + 10;
        if (!isInRange) return null;

        const op = interpolate(
          localFrame,
          [-10, 15, clip.duration - 20, clip.duration],
          [0, 1, 1, 0],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );

        // Ken Burns zoom
        const zoom = interpolate(localFrame, [0, clip.duration], [1.02, 1.12], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });

        return (
          <AbsoluteFill key={i} style={{ opacity: op }}>
            <OffthreadVideo
              src={staticFile(clip.src)}
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: `scale(${zoom})`,
                filter: "brightness(0.9)",
              }}
            />
          </AbsoluteFill>
        );
      })}

      {/* Dark top/bottom for readability */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.78) 0%, transparent 18%, transparent 80%, rgba(0,0,0,0.85) 100%)",
          pointerEvents: "none",
        }}
      />

      {/* Labels */}
      {active && (
        <>
          <div
            style={{
              position: "absolute",
              top: 36,
              left: 56,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div
              style={{
                fontSize: 13,
                letterSpacing: 6,
                color: TOKENS.colors.accent.cyan,
                fontFamily: "Inter, system-ui",
                fontWeight: 700,
              }}
            >
              {active.layer}
            </div>
            <div
              style={{
                fontSize: 36,
                fontFamily: "Inter, system-ui",
                fontWeight: 800,
                color: "#fff",
                letterSpacing: -0.5,
                textShadow: "0 2px 8px rgba(0,0,0,0.8)",
              }}
            >
              {active.title}
            </div>
          </div>

          <div
            style={{
              position: "absolute",
              bottom: 36,
              left: "50%",
              transform: "translateX(-50%)",
              padding: "10px 20px",
              background: "rgba(0,0,0,0.7)",
              border: `1px solid ${TOKENS.colors.accent.cyan}66`,
              borderRadius: 8,
              fontSize: 18,
              fontFamily: "Inter, system-ui",
              color: "#fff",
              letterSpacing: 1,
              textAlign: "center",
            }}
          >
            {active.desc}
          </div>
        </>
      )}

      {/* Section progress */}
      <div
        style={{
          position: "absolute",
          top: 40,
          right: 60,
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 13,
          color: TOKENS.colors.text.secondary,
          letterSpacing: 2,
        }}
      >
        {active
          ? `${CLIPS.indexOf(active) + 1} / ${CLIPS.length}`
          : `${CLIPS.length} / ${CLIPS.length}`}
      </div>
    </AbsoluteFill>
  );
};
