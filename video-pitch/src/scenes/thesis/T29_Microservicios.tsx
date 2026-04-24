import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T29 · Microservicios · 35s (1050f)
 * 5×5 Docker container grid with category colors + network hint.
 */
type Svc = { name: string; category: "data" | "compute" | "ops" | "observability" | "ui" };

const SERVICES: Svc[] = [
  { name: "postgres", category: "data" },
  { name: "timescaledb", category: "data" },
  { name: "redis", category: "data" },
  { name: "minio", category: "data" },
  { name: "vault", category: "data" },

  { name: "airflow-web", category: "compute" },
  { name: "airflow-sched", category: "compute" },
  { name: "airflow-worker", category: "compute" },
  { name: "mlflow", category: "compute" },
  { name: "feast-server", category: "compute" },

  { name: "signalbridge", category: "ops" },
  { name: "trading-api", category: "ops" },
  { name: "news-api", category: "ops" },
  { name: "inference-api", category: "ops" },
  { name: "analytics-api", category: "ops" },

  { name: "prometheus", category: "observability" },
  { name: "grafana", category: "observability" },
  { name: "alertmgr", category: "observability" },
  { name: "loki", category: "observability" },
  { name: "jaeger", category: "observability" },

  { name: "nextjs", category: "ui" },
  { name: "pgadmin", category: "ui" },
  { name: "promtail", category: "observability" },
  { name: "nginx", category: "ui" },
  { name: "traefik", category: "ui" },
];

const COLORS: Record<Svc["category"], string> = {
  data: TOKENS.colors.accent.cyan,
  compute: TOKENS.colors.market.up,
  ops: TOKENS.colors.accent.purple,
  observability: TOKENS.colors.semantic.warning,
  ui: "#a78bfa",
};

const LABELS: Record<Svc["category"], string> = {
  data: "Data layer",
  compute: "Compute",
  ops: "Ops / APIs",
  observability: "Observability",
  ui: "UI / Gateway",
};

export const T29_Microservicios: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const legendOp = interpolate(frame, [840, 960], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [1020, 1050], [1, 0], {
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
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO VI · INFRAESTRUCTURA
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
          25 servicios · aislamiento por Docker
        </div>
      </div>

      {/* Grid */}
      <div
        style={{
          position: "absolute",
          top: 220,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gridAutoRows: "110px",
          gap: 14,
        }}
      >
        {SERVICES.map((s, i) => {
          const appearAt = 60 + i * 28;
          const op = interpolate(frame, [appearAt, appearAt + 30], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const scale = interpolate(frame, [appearAt, appearAt + 30], [0.8, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={s.name}
              style={{
                opacity: op,
                transform: `scale(${scale})`,
                background: `${COLORS[s.category]}22`,
                border: `1px solid ${COLORS[s.category]}`,
                borderRadius: 8,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: 8,
                gap: 4,
              }}
            >
              <div
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: 6,
                  background: COLORS[s.category],
                  boxShadow: `0 0 16px ${COLORS[s.category]}88`,
                }}
              />
              <div
                style={{
                  fontSize: 14,
                  fontFamily: "JetBrains Mono, monospace",
                  color: "#fff",
                  letterSpacing: 0.5,
                }}
              >
                {s.name}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 80,
          right: 80,
          display: "flex",
          justifyContent: "center",
          gap: 28,
          opacity: legendOp,
        }}
      >
        {(Object.keys(LABELS) as Svc["category"][]).map((cat) => (
          <div
            key={cat}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 14,
              fontFamily: "JetBrains Mono, monospace",
              color: TOKENS.colors.text.secondary,
              letterSpacing: 1,
            }}
          >
            <div
              style={{
                width: 14,
                height: 14,
                borderRadius: 4,
                background: COLORS[cat],
              }}
            />
            {LABELS[cat]}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
