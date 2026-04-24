import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T09 · SystemOps · 30s (900f)
 * Radial watchdog: central core + orbiting service/DAG nodes.
 */
const SERVICES = [
  "postgres",
  "redis",
  "airflow",
  "mlflow",
  "grafana",
  "prometheus",
  "signalbridge",
  "minio",
  "loki",
  "jaeger",
  "pgadmin",
  "promtail",
  "vault",
  "nextjs",
  "alertmgr",
  "trading-api",
  "news-api",
  "feast",
];

const CENTER_X = 960;
const CENTER_Y = 540;
const RADIUS = 280;

export const T09_SystemOps: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const nodesOp = interpolate(frame, [60, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [870, 900], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Rotation
  const rotation = (frame / 30) * 10; // 10 deg/sec

  // Pulse
  const pulse = 0.85 + 0.15 * Math.sin(frame * 0.12);

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
            color: TOKENS.colors.market.up,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO II · OPERACIÓN
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
          Watchdog · 25 servicios orbitando
        </div>
      </div>

      {/* Orbit SVG */}
      <svg
        width={1920}
        height={1080}
        style={{ position: "absolute", inset: 0 }}
      >
        {/* Orbit rings */}
        {[1, 2, 3].map((r) => (
          <circle
            key={r}
            cx={CENTER_X}
            cy={CENTER_Y}
            r={r * 90}
            fill="none"
            stroke={TOKENS.colors.accent.cyan}
            strokeWidth={0.5}
            opacity={0.12 * nodesOp}
          />
        ))}

        {/* Center core */}
        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={60 * pulse}
          fill={TOKENS.colors.market.up}
          opacity={0.15}
        />
        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={38}
          fill={TOKENS.colors.market.up}
        />
        <text
          x={CENTER_X}
          y={CENTER_Y + 8}
          textAnchor="middle"
          fontSize={22}
          fontFamily="Inter"
          fontWeight={700}
          fill="#000"
        >
          CORE
        </text>

        {/* Orbiting services */}
        {SERVICES.map((svc, i) => {
          const angle = (i / SERVICES.length) * 360 + rotation;
          const rad = (angle * Math.PI) / 180;
          const r = RADIUS + (i % 3) * 40;
          const x = CENTER_X + r * Math.cos(rad);
          const y = CENTER_Y + r * Math.sin(rad);

          const appearAt = 80 + i * 6;
          const op = interpolate(frame, [appearAt, appearAt + 30], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          return (
            <g key={svc} opacity={op}>
              <line
                x1={CENTER_X}
                y1={CENTER_Y}
                x2={x}
                y2={y}
                stroke={TOKENS.colors.accent.cyan}
                strokeWidth={0.5}
                opacity={0.2}
              />
              <circle
                cx={x}
                cy={y}
                r={10}
                fill={TOKENS.colors.accent.cyan}
                opacity={0.9}
              />
              <text
                x={x}
                y={y + 28}
                textAnchor="middle"
                fontSize={14}
                fontFamily="JetBrains Mono"
                fill="#fff"
                opacity={0.85}
              >
                {svc}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Stats bar */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 80,
          right: 80,
          display: "flex",
          gap: 24,
          justifyContent: "center",
          opacity: interpolate(frame, [600, 700], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {[
          { value: "37", label: "DAGs" },
          { value: "25+", label: "Servicios Docker" },
          { value: "4", label: "Grafana dashboards" },
          { value: "37+", label: "Prometheus alerts" },
        ].map((s, i) => (
          <div
            key={i}
            style={{
              padding: "14px 28px",
              background: "rgba(0,211,149,0.12)",
              border: `1px solid ${TOKENS.colors.market.up}66`,
              borderRadius: 10,
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontSize: 36,
                fontFamily: "JetBrains Mono, monospace",
                fontWeight: 700,
                color: TOKENS.colors.market.up,
              }}
            >
              {s.value}
            </div>
            <div
              style={{
                fontSize: 12,
                fontFamily: "Inter, system-ui",
                color: TOKENS.colors.text.secondary,
                letterSpacing: 3,
                textTransform: "uppercase",
                marginTop: 4,
              }}
            >
              {s.label}
            </div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
