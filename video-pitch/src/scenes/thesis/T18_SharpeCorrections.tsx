import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { THESIS_STATISTICS } from "../../data/metrics-thesis";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T18 · Sharpe corrections · 45s (1350f)
 * Three boxes: Raw Sharpe → PSR → DSR with formulas.
 */
export const T18_SharpeCorrections: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const box1Op = interpolate(frame, [90, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const box2Op = interpolate(frame, [360, 480], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const box3Op = interpolate(frame, [660, 780], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const interpOp = interpolate(frame, [960, 1080], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1320, 1350], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const s = THESIS_STATISTICS.sharpe;

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
          CAPÍTULO IV · CORRECCIONES DEL SHARPE
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
          Raw → PSR → DSR · Bailey &amp; López de Prado
        </div>
      </div>

      {/* Three sharpe boxes */}
      <div
        style={{
          position: "absolute",
          top: 240,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "1fr 80px 1fr 80px 1fr",
          alignItems: "center",
          gap: 0,
        }}
      >
        <SharpeBox
          op={box1Op}
          label="RAW SHARPE"
          value={s.raw.toFixed(3)}
          description="N = 34 trades · anualizado × √(252/5)"
          color="#fff"
        />

        <ArrowBlock op={box2Op} />

        <SharpeBox
          op={box2Op}
          label="PSR (Probabilistic)"
          value={s.psr.toFixed(2)}
          description={`P(SR_real > 0 | N=34) · Bailey 2012`}
          color={TOKENS.colors.accent.cyan}
          subValue="93% de probabilidad"
        />

        <ArrowBlock op={box3Op} />

        <SharpeBox
          op={box3Op}
          label="DSR (Deflated)"
          value={s.dsr.toFixed(2)}
          description={`descuenta k=${s.n_trials_evaluated} trials previos`}
          color={TOKENS.colors.market.up}
          subValue="78% de probabilidad"
        />
      </div>

      {/* Formulas row */}
      <div
        style={{
          position: "absolute",
          top: 530,
          left: 80,
          right: 80,
          opacity: interpOp,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
        }}
      >
        <div
          style={{
            padding: "18px 24px",
            background: "rgba(6,182,212,0.08)",
            border: `1px solid ${TOKENS.colors.accent.cyan}55`,
            borderRadius: 10,
          }}
        >
          <div
            style={{
              fontSize: 14,
              letterSpacing: 4,
              color: TOKENS.colors.accent.cyan,
              fontFamily: "Inter, system-ui",
              fontWeight: 700,
              textTransform: "uppercase",
            }}
          >
            Bootstrap CI 95% (BCa, 10k iter, block=3)
          </div>
          <div
            style={{
              fontSize: 32,
              fontFamily: "JetBrains Mono, monospace",
              color: "#fff",
              marginTop: 8,
              letterSpacing: 0.5,
            }}
          >
            [{s.ci_95_bca[0].toFixed(2)}, {s.ci_95_bca[1].toFixed(2)}]
          </div>
          <div
            style={{
              fontSize: 14,
              fontFamily: "Inter, system-ui",
              color: TOKENS.colors.text.secondary,
              marginTop: 4,
            }}
          >
            Rango robusto — el lower bound 2.08 ya supera benchmarks
            académicos (&gt; 1.0).
          </div>
        </div>

        <div
          style={{
            padding: "18px 24px",
            background: "rgba(0,211,149,0.08)",
            border: `1px solid ${TOKENS.colors.market.up}55`,
            borderRadius: 10,
          }}
        >
          <div
            style={{
              fontSize: 14,
              letterSpacing: 4,
              color: TOKENS.colors.market.up,
              fontFamily: "Inter, system-ui",
              fontWeight: 700,
              textTransform: "uppercase",
            }}
          >
            Interpretación
          </div>
          <div
            style={{
              fontSize: 17,
              fontFamily: "Inter, system-ui",
              color: "#fff",
              marginTop: 8,
              lineHeight: 1.4,
            }}
          >
            Aun descontando el problema de &quot;multiple testing&quot;
            (70 experimentos previos en EXPERIMENT_LOG), el DSR = 0.78
            mantiene la tesis sobre el umbral de significancia aceptable para
            ML cuantitativo. <b>No es un hallazgo por suerte.</b>
          </div>
        </div>
      </div>

      {/* Bottom footnote */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          textAlign: "center",
          fontSize: 14,
          fontFamily: "JetBrains Mono, monospace",
          color: TOKENS.colors.text.muted,
          letterSpacing: 2,
          opacity: interpOp,
        }}
      >
        PSR(SR) = Φ((SR · √(N-1)) / √(1 - γ₃·SR + 0.25·(γ₄-1)·SR²)) ·
        Bailey &amp; López de Prado (2012)
      </div>
    </AbsoluteFill>
  );
};

const SharpeBox: React.FC<{
  op: number;
  label: string;
  value: string;
  description: string;
  color: string;
  subValue?: string;
}> = ({ op, label, value, description, color, subValue }) => (
  <div
    style={{
      opacity: op,
      padding: "28px 22px",
      background: `${color}14`,
      border: `2px solid ${color}`,
      borderRadius: 12,
      textAlign: "center",
      minHeight: 240,
    }}
  >
    <div
      style={{
        fontSize: 14,
        letterSpacing: 4,
        color,
        fontFamily: "Inter, system-ui",
        fontWeight: 700,
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
    <div
      style={{
        fontSize: 92,
        fontFamily: "JetBrains Mono, monospace",
        fontWeight: 700,
        color,
        letterSpacing: -3,
        marginTop: 12,
      }}
    >
      {value}
    </div>
    {subValue && (
      <div
        style={{
          fontSize: 18,
          fontFamily: "Inter, system-ui",
          color,
          marginTop: 2,
          letterSpacing: 1,
        }}
      >
        {subValue}
      </div>
    )}
    <div
      style={{
        fontSize: 14,
        fontFamily: "Inter, system-ui",
        color: TOKENS.colors.text.secondary,
        marginTop: 14,
        lineHeight: 1.3,
      }}
    >
      {description}
    </div>
  </div>
);

const ArrowBlock: React.FC<{ op: number }> = ({ op }) => (
  <div
    style={{
      opacity: op,
      fontSize: 48,
      fontFamily: "JetBrains Mono, monospace",
      color: TOKENS.colors.accent.purple,
      textAlign: "center",
      fontWeight: 700,
    }}
  >
    →
  </div>
);
