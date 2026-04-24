import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DA_BY_MODEL } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T12 · Componente DA · 65s (1950f)
 * Direction accuracy per model + binomial test honest reveal.
 */
const MODELS = [
  { key: "ridge_h1", label: "Ridge", ...DA_BY_MODEL.ridge_h1 },
  { key: "bayesian_ridge_h1", label: "BayesianRidge", ...DA_BY_MODEL.bayesian_ridge_h1 },
  { key: "ard_h1", label: "ARD", ...DA_BY_MODEL.ard_h1 },
  { key: "xgboost_pure_h1", label: "XGBoost", ...DA_BY_MODEL.xgboost_pure_h1 },
  { key: "lightgbm_pure_h1", label: "LightGBM", ...DA_BY_MODEL.lightgbm_pure_h1 },
  { key: "catboost_pure_h1", label: "CatBoost", ...DA_BY_MODEL.catboost_pure_h1 },
  { key: "ensemble_top3_h1", label: "Ensemble Top-3", ...DA_BY_MODEL.ensemble_top3_h1 },
  { key: "ensemble_top6_h1", label: "Ensemble Top-6", ...DA_BY_MODEL.ensemble_top6_h1 },
  { key: "ensemble_best_of_breed_h1", label: "Best-of-Breed", ...DA_BY_MODEL.ensemble_best_of_breed_h1 },
];

export const T12_ComponenteDA: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const randomLineOp = interpolate(frame, [120, 180], [0, 0.9], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const verdictOp = interpolate(frame, [1440, 1560], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const exitFade = interpolate(frame, [1920, 1950], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Chart layout
  const chartLeft = 120;
  const chartRight = 1800;
  const chartTop = 240;
  const chartBottom = 800;
  const barWidth = (chartRight - chartLeft - 40) / MODELS.length;
  const yAt = (da: number) =>
    chartBottom - ((da - 45) / 15) * (chartBottom - chartTop);

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
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO III · COMPONENTE 1 — PREDICCIÓN
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
          Direction Accuracy honesto · N = 250
        </div>
      </div>

      {/* Chart */}
      <svg
        width={1920}
        height={1080}
        style={{ position: "absolute", inset: 0 }}
      >
        {/* Grid lines */}
        {[45, 50, 55, 60].map((da) => {
          const y = yAt(da);
          return (
            <g key={da}>
              <line
                x1={chartLeft}
                y1={y}
                x2={chartRight}
                y2={y}
                stroke="#334155"
                strokeWidth={0.5}
                strokeDasharray="4 6"
                opacity={0.6}
              />
              <text
                x={chartLeft - 10}
                y={y + 5}
                fontSize={16}
                fontFamily="JetBrains Mono"
                fill={TOKENS.colors.text.secondary}
                textAnchor="end"
              >
                {da}%
              </text>
            </g>
          );
        })}

        {/* Random line = 50% */}
        <line
          x1={chartLeft}
          y1={yAt(50)}
          x2={chartRight}
          y2={yAt(50)}
          stroke={TOKENS.colors.market.down}
          strokeWidth={2}
          strokeDasharray="10 6"
          opacity={randomLineOp}
        />
        <text
          x={chartRight + 10}
          y={yAt(50) + 6}
          fontSize={16}
          fontFamily="JetBrains Mono"
          fill={TOKENS.colors.market.down}
          opacity={randomLineOp}
        >
          random = 50%
        </text>

        {/* Bars */}
        {MODELS.map((m, i) => {
          const x = chartLeft + 20 + i * barWidth;
          const barHeight = chartBottom - yAt(m.da);
          const appearAt = 180 + i * 80;
          const rise = interpolate(frame, [appearAt, appearAt + 60], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          const isSignificant = m.binomial_p < 0.05;
          const color = isSignificant
            ? TOKENS.colors.market.up
            : TOKENS.colors.accent.cyan;

          return (
            <g key={m.key}>
              <rect
                x={x}
                y={chartBottom - barHeight * rise}
                width={barWidth - 16}
                height={barHeight * rise}
                fill={color}
                opacity={0.6}
                rx={4}
              />
              <rect
                x={x}
                y={chartBottom - barHeight * rise}
                width={barWidth - 16}
                height={3}
                fill={color}
                rx={2}
              />
              <text
                x={x + (barWidth - 16) / 2}
                y={chartBottom - barHeight * rise - 10}
                fontSize={16}
                fontFamily="JetBrains Mono"
                fontWeight={700}
                fill="#fff"
                textAnchor="middle"
                opacity={rise}
              >
                {m.da.toFixed(1)}%
              </text>
              <text
                x={x + (barWidth - 16) / 2}
                y={chartBottom + 24}
                fontSize={14}
                fontFamily="Inter"
                fill={TOKENS.colors.text.secondary}
                textAnchor="middle"
                opacity={rise}
              >
                {m.label}
              </text>
              <text
                x={x + (barWidth - 16) / 2}
                y={chartBottom + 46}
                fontSize={12}
                fontFamily="JetBrains Mono"
                fill={TOKENS.colors.text.muted}
                textAnchor="middle"
                opacity={rise}
              >
                p={m.binomial_p.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Verdict */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 80,
          right: 80,
          padding: "18px 28px",
          background: `${TOKENS.colors.semantic.warning}18`,
          border: `1px solid ${TOKENS.colors.semantic.warning}`,
          borderRadius: 10,
          opacity: verdictOp,
        }}
      >
        <div
          style={{
            fontSize: 14,
            letterSpacing: 5,
            color: TOKENS.colors.semantic.warning,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Honestidad metodológica
        </div>
        <div
          style={{
            fontSize: 22,
            fontFamily: "Inter, system-ui",
            color: "#fff",
            marginTop: 6,
            lineHeight: 1.4,
          }}
        >
          Ningún modelo rechaza H₀ &ldquo;DA = 50%&rdquo; con α = 0.05 (test binomial,
          N = 250). El mejor ensemble alcanza DA = 53.0% con p = 0.17. <br />
          <b style={{ color: TOKENS.colors.accent.cyan }}>
            El edge NO proviene del DA aislado.
          </b>{" "}
          Proviene de combinar predicción débil con régimen fuerte.
        </div>
      </div>
    </AbsoluteFill>
  );
};
