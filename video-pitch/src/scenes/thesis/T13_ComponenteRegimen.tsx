import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { PITCH_METRICS } from "../../data/metrics";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T13 · Componente Régimen (MVP) · 75s (2250f)
 * Hurst R/S explanation + weekly blocking grid for Q1 2026.
 */
const WEEKS_Q1_2026: Array<{
  label: string;
  hurst: number;
  regime: "MEAN_REVERT" | "INDETERM" | "TRENDING";
  action: "SKIP" | "TRADE" | "REDUCED";
}> = [
  { label: "W1", hurst: 0.16, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W2", hurst: 0.22, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W3", hurst: 0.18, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W4", hurst: 0.31, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W5", hurst: 0.27, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W6", hurst: 0.35, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W7", hurst: 0.41, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W8", hurst: 0.29, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W9", hurst: 0.33, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W10", hurst: 0.44, regime: "INDETERM", action: "REDUCED" },
  { label: "W11", hurst: 0.38, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W12", hurst: 0.36, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W13", hurst: 0.41, regime: "MEAN_REVERT", action: "SKIP" },
  { label: "W14", hurst: 0.39, regime: "MEAN_REVERT", action: "SKIP" },
];

export const T13_ComponenteRegimen: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const formulaOp = interpolate(frame, [90, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const gridTitleOp = interpolate(frame, [720, 810], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const verdictOp = interpolate(frame, [1920, 2040], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [2220, 2250], [1, 0], {
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
            color: TOKENS.colors.market.down,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO III · COMPONENTE 2 — RÉGIMEN (MVP) ★
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
          Hurst R/S · clasificar antes de predecir
        </div>
      </div>

      {/* Formula block */}
      <div
        style={{
          position: "absolute",
          top: 220,
          left: 80,
          right: 80,
          padding: "20px 32px",
          background: "rgba(255,59,105,0.10)",
          border: `1px solid ${TOKENS.colors.market.down}66`,
          borderRadius: 10,
          opacity: formulaOp,
          display: "flex",
          gap: 40,
          alignItems: "center",
        }}
      >
        <div style={{ flex: "0 0 auto" }}>
          <div
            style={{
              fontSize: 56,
              fontFamily: "JetBrains Mono, monospace",
              color: "#fff",
              letterSpacing: 1,
              fontWeight: 700,
            }}
          >
            H = log(R/S) / log(n)
          </div>
          <div
            style={{
              fontSize: 16,
              fontFamily: "JetBrains Mono, monospace",
              color: TOKENS.colors.text.secondary,
              marginTop: 6,
              letterSpacing: 0.5,
            }}
          >
            ventana = 60 días · window_size tuned on 2020-2024
          </div>
        </div>

        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <ThresholdRow
            condition="H > 0.52"
            label="TRENDING"
            action="sizing × 1.0"
            color={TOKENS.colors.market.up}
          />
          <ThresholdRow
            condition="0.42 ≤ H ≤ 0.52"
            label="INDETERMINATE"
            action="sizing × 0.40"
            color={TOKENS.colors.semantic.warning}
          />
          <ThresholdRow
            condition="H < 0.42"
            label="MEAN-REVERTING"
            action="skip_trade = True"
            color={TOKENS.colors.market.down}
          />
        </div>
      </div>

      {/* Weekly blocking grid */}
      <div
        style={{
          position: "absolute",
          top: 520,
          left: 80,
          right: 80,
          opacity: gridTitleOp,
        }}
      >
        <div
          style={{
            fontSize: 16,
            letterSpacing: 5,
            color: TOKENS.colors.accent.cyan,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            textTransform: "uppercase",
          }}
        >
          Q1 2026 · 14 semanas evaluadas
        </div>
        <div
          style={{
            fontSize: 22,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
            color: "#fff",
            marginTop: 2,
            marginBottom: 14,
          }}
        >
          El gate decidió NO operar 13 de 14 veces.
        </div>

        {/* Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(14, 1fr)",
            gap: 8,
          }}
        >
          {WEEKS_Q1_2026.map((w, i) => {
            const appearAt = 810 + i * 50;
            const op = interpolate(frame, [appearAt, appearAt + 40], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const color =
              w.action === "SKIP"
                ? TOKENS.colors.market.down
                : w.action === "REDUCED"
                ? TOKENS.colors.semantic.warning
                : TOKENS.colors.market.up;
            return (
              <div
                key={i}
                style={{
                  opacity: op,
                  padding: "14px 8px",
                  background: `${color}20`,
                  border: `1px solid ${color}`,
                  borderRadius: 8,
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontFamily: "JetBrains Mono, monospace",
                    color,
                    fontWeight: 700,
                    letterSpacing: 1,
                  }}
                >
                  {w.label}
                </div>
                <div
                  style={{
                    fontSize: 20,
                    fontFamily: "JetBrains Mono, monospace",
                    color: "#fff",
                    marginTop: 6,
                    fontWeight: 700,
                  }}
                >
                  {w.hurst.toFixed(2)}
                </div>
                <div
                  style={{
                    fontSize: 10,
                    fontFamily: "Inter, system-ui",
                    color: TOKENS.colors.text.secondary,
                    letterSpacing: 1,
                    marginTop: 4,
                  }}
                >
                  {w.action}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Verdict */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 80,
          right: 80,
          padding: "18px 28px",
          background: `${TOKENS.colors.market.up}18`,
          border: `2px solid ${TOKENS.colors.market.up}`,
          borderRadius: 10,
          opacity: verdictOp,
          display: "flex",
          gap: 24,
          alignItems: "center",
        }}
      >
        <div
          style={{
            fontSize: 72,
            fontFamily: "JetBrains Mono, monospace",
            fontWeight: 700,
            color: TOKENS.colors.market.up,
            letterSpacing: -2,
          }}
        >
          {PITCH_METRICS.ytd_2026.alpha_pp.toFixed(2)} pp
        </div>
        <div
          style={{
            flex: 1,
            fontSize: 22,
            fontFamily: "Inter, system-ui",
            color: "#fff",
            lineHeight: 1.35,
          }}
        >
          Alpha YTD 2026 frente a Buy-and-Hold (-2.82%). <br />
          <b style={{ color: TOKENS.colors.market.up }}>
            El gate convierte un modelo perdedor en un sistema rentable,
          </b>{" "}
          no porque acierte más, sino porque rehúsa jugar cuando el tablero no lo
          favorece.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const ThresholdRow: React.FC<{
  condition: string;
  label: string;
  action: string;
  color: string;
}> = ({ condition, label, action, color }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      gap: 16,
      padding: "6px 12px",
      background: `${color}14`,
      border: `1px solid ${color}66`,
      borderRadius: 6,
    }}
  >
    <div
      style={{
        fontSize: 16,
        fontFamily: "JetBrains Mono, monospace",
        color: "#fff",
        minWidth: 170,
      }}
    >
      {condition}
    </div>
    <div
      style={{
        fontSize: 14,
        letterSpacing: 3,
        color,
        fontFamily: "Inter, system-ui",
        fontWeight: 700,
        minWidth: 170,
      }}
    >
      {label}
    </div>
    <div
      style={{
        fontSize: 14,
        fontFamily: "JetBrains Mono, monospace",
        color: TOKENS.colors.text.secondary,
      }}
    >
      → {action}
    </div>
  </div>
);
