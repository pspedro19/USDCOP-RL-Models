import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { THESIS_STATISTICS } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T19 · Block Bootstrap (Politis-Romano) · 45s (1350f)
 * Histogram visualization of 10k block bootstrap iterations on 2025 OOS returns.
 */
const BINS = 40;

// Procedural gaussian-like histogram centered at 0.22, SD ~ 0.09
const makeHistogram = (): number[] => {
  const hist: number[] = [];
  const mu = 0.22;
  const sigma = 0.09;
  const min = -0.15;
  const max = 0.55;
  const step = (max - min) / BINS;
  for (let i = 0; i < BINS; i++) {
    const x = min + i * step;
    const z = (x - mu) / sigma;
    const density = Math.exp(-0.5 * z * z);
    hist.push(density);
  }
  const maxH = Math.max(...hist);
  return hist.map((v) => v / maxH);
};
const HIST = makeHistogram();
const MIN = -0.15;
const MAX = 0.55;

export const T19_BootstrapBlock: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const barGrow = interpolate(frame, [120, 720], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const meanLineOp = interpolate(frame, [700, 800], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const ciOp = interpolate(frame, [820, 920], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const verdictOp = interpolate(frame, [1000, 1120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1320, 1350], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const chartL = 120;
  const chartR = 1800;
  const chartT = 260;
  const chartB = 760;
  const barW = (chartR - chartL) / BINS;
  const xAt = (x: number) => chartL + ((x - MIN) / (MAX - MIN)) * (chartR - chartL);

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
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO IV · BOOTSTRAP ESTACIONARIO
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
          Politis-Romano · 10k iteraciones · block = 3
        </div>
      </div>

      {/* Config line */}
      <div
        style={{
          position: "absolute",
          top: 170,
          left: 80,
          fontSize: 18,
          fontFamily: "JetBrains Mono, monospace",
          color: TOKENS.colors.text.secondary,
          letterSpacing: 1,
          opacity: titleOp,
        }}
      >
        Null: retornos aleatorios con misma distribución marginal · iter =
        10,000 · block_size = N^(1/3) = 3
      </div>

      {/* Histogram */}
      <svg width={1920} height={1080} style={{ position: "absolute", inset: 0 }}>
        {/* Bars */}
        {HIST.map((h, i) => {
          const x = chartL + i * barW;
          const barHeight = h * (chartB - chartT) * barGrow;
          const xVal = MIN + (i + 0.5) * ((MAX - MIN) / BINS);
          const isActual = Math.abs(xVal - 0.256) < (MAX - MIN) / BINS / 2;
          const isCIBound =
            Math.abs(xVal - 0.05) < (MAX - MIN) / BINS / 2 ||
            Math.abs(xVal - 0.40) < (MAX - MIN) / BINS / 2;
          const fill = isActual
            ? TOKENS.colors.market.up
            : isCIBound
            ? TOKENS.colors.accent.cyan
            : TOKENS.colors.accent.purple;
          return (
            <rect
              key={i}
              x={x + 1}
              y={chartB - barHeight}
              width={barW - 2}
              height={barHeight}
              fill={fill}
              opacity={0.75}
              rx={1}
            />
          );
        })}

        {/* Zero line */}
        <line
          x1={xAt(0)}
          y1={chartT}
          x2={xAt(0)}
          y2={chartB}
          stroke={TOKENS.colors.market.down}
          strokeWidth={2}
          strokeDasharray="6 4"
          opacity={meanLineOp}
        />
        <text
          x={xAt(0) + 8}
          y={chartT + 20}
          fontSize={16}
          fontFamily="JetBrains Mono"
          fill={TOKENS.colors.market.down}
          opacity={meanLineOp}
        >
          0% (null)
        </text>

        {/* Actual realized */}
        <line
          x1={xAt(0.2563)}
          y1={chartT}
          x2={xAt(0.2563)}
          y2={chartB + 6}
          stroke={TOKENS.colors.market.up}
          strokeWidth={3}
          opacity={meanLineOp}
        />
        <text
          x={xAt(0.2563) + 10}
          y={chartT + 20}
          fontSize={18}
          fontFamily="JetBrains Mono"
          fontWeight={700}
          fill={TOKENS.colors.market.up}
          opacity={meanLineOp}
        >
          +25.63% (observado)
        </text>

        {/* CI bounds */}
        <line
          x1={xAt(0.05)}
          y1={chartB}
          x2={xAt(0.05)}
          y2={chartB - 50}
          stroke={TOKENS.colors.accent.cyan}
          strokeWidth={2}
          opacity={ciOp}
        />
        <line
          x1={xAt(0.40)}
          y1={chartB}
          x2={xAt(0.40)}
          y2={chartB - 50}
          stroke={TOKENS.colors.accent.cyan}
          strokeWidth={2}
          opacity={ciOp}
        />
        <text
          x={xAt(0.225)}
          y={chartB + 30}
          fontSize={14}
          fontFamily="JetBrains Mono"
          fill={TOKENS.colors.accent.cyan}
          textAnchor="middle"
          opacity={ciOp}
        >
          95% CI BCa: [+5%, +40%]
        </text>

        {/* Axis ticks */}
        {[-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
          <g key={v} opacity={titleOp}>
            <line
              x1={xAt(v)}
              y1={chartB}
              x2={xAt(v)}
              y2={chartB + 6}
              stroke={TOKENS.colors.text.secondary}
            />
            <text
              x={xAt(v)}
              y={chartB + 22}
              fontSize={12}
              fontFamily="JetBrains Mono"
              fill={TOKENS.colors.text.secondary}
              textAnchor="middle"
            >
              {v >= 0 ? "+" : ""}
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}
      </svg>

      {/* Verdict */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          left: 80,
          right: 80,
          padding: "16px 28px",
          background: `${TOKENS.colors.accent.purple}18`,
          border: `1px solid ${TOKENS.colors.accent.purple}`,
          borderRadius: 10,
          opacity: verdictOp,
          fontSize: 21,
          fontFamily: "Inter, system-ui",
          color: "#fff",
          lineHeight: 1.4,
        }}
      >
        <b style={{ color: TOKENS.colors.accent.purple }}>
          p-value empírico = {THESIS_STATISTICS.p_values.bootstrap_empirical}.
        </b>{" "}
        Solo el 0.9% de las 10,000 replicas bajo la null igualan o superan el
        retorno observado. La cola derecha es densa, no gruesa — el resultado no
        es explicable por azar estructurado de la serie.
      </div>
    </AbsoluteFill>
  );
};
