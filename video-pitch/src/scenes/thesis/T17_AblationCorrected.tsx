import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { ABLATION_ANALYSIS } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T17 · Ablation table con corrección BH-FDR · 60s (1800f)
 */
export const T17_AblationCorrected: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const verdictOp = interpolate(frame, [1400, 1520], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1770, 1800], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const colWidths = ["28%", "12%", "10%", "10%", "10%", "14%", "16%"];

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
            color: TOKENS.colors.accent.purple,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO IV · VALIDACIÓN ESTADÍSTICA
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
          Análisis de ablación con corrección BH-FDR
        </div>
      </div>

      {/* Table header */}
      <div
        style={{
          position: "absolute",
          top: 220,
          left: 80,
          right: 80,
          display: "flex",
          padding: "16px 20px",
          background: "rgba(139,92,246,0.15)",
          border: `1px solid ${TOKENS.colors.accent.purple}66`,
          borderRadius: 8,
          fontSize: 15,
          fontFamily: "JetBrains Mono, monospace",
          color: TOKENS.colors.accent.purple,
          letterSpacing: 1.5,
          textTransform: "uppercase",
          fontWeight: 700,
        }}
      >
        <div style={{ width: colWidths[0] }}>Configuración</div>
        <div style={{ width: colWidths[1], textAlign: "right" }}>Return</div>
        <div style={{ width: colWidths[2], textAlign: "right" }}>Sharpe</div>
        <div style={{ width: colWidths[3], textAlign: "right" }}>MaxDD</div>
        <div style={{ width: colWidths[4], textAlign: "right" }}>Trades</div>
        <div style={{ width: colWidths[5], textAlign: "right" }}>p-raw</div>
        <div style={{ width: colWidths[6], textAlign: "right" }}>p-BH(k=70)</div>
      </div>

      {/* Rows */}
      <div
        style={{
          position: "absolute",
          top: 300,
          left: 80,
          right: 80,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {ABLATION_ANALYSIS.map((row, i) => {
          const appearAt = 120 + i * 220;
          const op = interpolate(frame, [appearAt, appearAt + 90], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const highlight = i === ABLATION_ANALYSIS.length - 1;
          const sigBg = row.p_bh_fdr !== null && row.p_bh_fdr < 0.05;
          return (
            <div
              key={i}
              style={{
                opacity: op,
                display: "flex",
                padding: "14px 20px",
                background: highlight
                  ? "rgba(0,211,149,0.15)"
                  : sigBg
                  ? "rgba(34,197,94,0.08)"
                  : "rgba(15,23,42,0.6)",
                border: highlight
                  ? `2px solid ${TOKENS.colors.market.up}`
                  : `1px solid ${TOKENS.colors.text.muted}`,
                borderRadius: 8,
                fontSize: 18,
                fontFamily: "JetBrains Mono, monospace",
                color: "#fff",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: colWidths[0],
                  fontFamily: "Inter, system-ui",
                  fontWeight: highlight ? 700 : 500,
                  fontSize: 18,
                }}
              >
                {row.config}
              </div>
              <div
                style={{
                  width: colWidths[1],
                  textAlign: "right",
                  color:
                    row.return_pct >= 0
                      ? TOKENS.colors.market.up
                      : TOKENS.colors.market.down,
                  fontWeight: 700,
                }}
              >
                {row.return_pct >= 0 ? "+" : ""}
                {row.return_pct.toFixed(2)}%
              </div>
              <div style={{ width: colWidths[2], textAlign: "right" }}>
                {row.sharpe === null ? "—" : row.sharpe.toFixed(2)}
              </div>
              <div
                style={{
                  width: colWidths[3],
                  textAlign: "right",
                  color: TOKENS.colors.market.down,
                }}
              >
                {row.max_dd_pct.toFixed(1)}%
              </div>
              <div style={{ width: colWidths[4], textAlign: "right" }}>
                {row.trades ?? "—"}
              </div>
              <div
                style={{
                  width: colWidths[5],
                  textAlign: "right",
                  color: TOKENS.colors.text.secondary,
                }}
              >
                {row.p_raw === null ? "—" : row.p_raw.toFixed(3)}
              </div>
              <div
                style={{
                  width: colWidths[6],
                  textAlign: "right",
                  color:
                    row.p_bh_fdr !== null && row.p_bh_fdr < 0.05
                      ? TOKENS.colors.market.up
                      : TOKENS.colors.semantic.warning,
                  fontWeight: 700,
                }}
              >
                {row.p_bh_fdr === null ? "—" : row.p_bh_fdr.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Verdict */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          left: 80,
          right: 80,
          padding: "16px 28px",
          background: `${TOKENS.colors.market.up}18`,
          border: `1px solid ${TOKENS.colors.market.up}`,
          borderRadius: 10,
          opacity: verdictOp,
          fontSize: 21,
          fontFamily: "Inter, system-ui",
          color: "#fff",
          lineHeight: 1.4,
        }}
      >
        <b style={{ color: TOKENS.colors.market.up }}>Sistema completo:</b>{" "}
        +25.63% con p-BH = 0.03 &lt; 0.05 tras corrección para k = 70
        experimentos. Cada componente añadido mejora monotónicamente el Sharpe
        y reduce el drawdown. <b>La sinergia es lo que es significativo.</b>
      </div>
    </AbsoluteFill>
  );
};
