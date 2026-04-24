import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T07 · Overview arquitectura · 30s (900f)
 * L0 → L8 layered diagram with progressive reveal.
 */
const LAYERS = [
  {
    id: "L0",
    name: "Ingesta",
    desc: "OHLCV 5-min · 3 pares FX · 40 macro",
    color: "#06B6D4",
  },
  {
    id: "L1",
    name: "Feature Store",
    desc: "FEATURE_ORDER hash · norm_stats train-only",
    color: "#8B5CF6",
  },
  {
    id: "L2",
    name: "Dataset",
    desc: "Train / Val / Test fijo · anti-leakage",
    color: "#6366f1",
  },
  {
    id: "L3",
    name: "Training",
    desc: "Ridge + BayesianRidge + XGBoost · expanding window",
    color: "#22c55e",
  },
  {
    id: "L4",
    name: "Backtest",
    desc: "Walk-forward · gates 5/5",
    color: "#00D395",
  },
  {
    id: "L5",
    name: "Signal + Regime",
    desc: "Hurst gate · confidence · DL · effective HS",
    color: "#f59e0b",
  },
  {
    id: "L6",
    name: "Monitor",
    desc: "Paper trading · guardrails · circuit breaker",
    color: "#FF3B69",
  },
  {
    id: "L7",
    name: "Execution",
    desc: "TP/HS · Friday close · MEXC CCXT",
    color: "#ef4444",
  },
  {
    id: "L8",
    name: "Intelligence",
    desc: "LLM análisis · news ingestion · dashboard",
    color: "#a78bfa",
  },
];

export const T07_OverviewArch: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [870, 900], [1, 0], {
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
          CAPÍTULO II · ARQUITECTURA
        </div>
        <div
          style={{
            fontSize: 48,
            fontFamily: "Inter, system-ui",
            fontWeight: 800,
            color: "#fff",
            letterSpacing: -1,
            marginTop: 4,
          }}
        >
          Pipeline L0 → L8
        </div>
      </div>

      {/* Stack */}
      <div
        style={{
          position: "absolute",
          top: 200,
          left: 80,
          right: 80,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {LAYERS.map((layer, idx) => {
          const appearAt = 60 + idx * 45;
          const op = interpolate(frame, [appearAt, appearAt + 50], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const x = interpolate(frame, [appearAt, appearAt + 50], [-40, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={layer.id}
              style={{
                opacity: op,
                transform: `translateX(${x}px)`,
                padding: "14px 24px",
                background: `${layer.color}18`,
                border: `1px solid ${layer.color}88`,
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                gap: 24,
              }}
            >
              <div
                style={{
                  width: 64,
                  height: 50,
                  background: layer.color,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 22,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  color: "#000",
                }}
              >
                {layer.id}
              </div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 22,
                    fontFamily: "Inter, system-ui",
                    fontWeight: 700,
                    color: "#fff",
                  }}
                >
                  {layer.name}
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontFamily: "JetBrains Mono, monospace",
                    color: TOKENS.colors.text.secondary,
                    marginTop: 2,
                  }}
                >
                  {layer.desc}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer count */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          textAlign: "center",
          fontSize: 18,
          fontFamily: "JetBrains Mono, monospace",
          color: TOKENS.colors.text.secondary,
          letterSpacing: 2,
          opacity: interpolate(frame, [600, 680], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        37 DAGs · 25+ servicios Docker · 9 modelos ML · 63 backtests walk-forward
      </div>
    </AbsoluteFill>
  );
};
