import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T32 · Contribuciones · 40s (1200f)
 * 6 original contributions claimed by the thesis.
 */
const CONTRIBUTIONS = [
  {
    title: "Régimen como gate de ejecución",
    body: "Operacionalización del Hurst R/S como filtro binario (trade / skip) calibrado sobre 5 años previos y validado en 2025 OOS + 2026 YTD.",
  },
  {
    title: "HS efectivo con cap portafolio",
    body: "Fórmula min(HS_base, 3.5% / leverage) que desacopla la tolerancia por-trade del apalancamiento y capa la pérdida máxima sistémica.",
  },
  {
    title: "Stack mlops reproducible USD/COP",
    body: "37 DAGs Airflow + 25 servicios Docker con contratos SSOT, migraciones numeradas y auditoría via psql directo.",
  },
  {
    title: "Protocolo de corrección múltiple",
    body: "Aplicación de Benjamini-Hochberg FDR y DSR (Bailey-López de Prado) sobre k=70 experimentos previos registrados en EXPERIMENT_LOG.",
  },
  {
    title: "Decoupling de señal y ejecución",
    body: "Contratos UniversalSignalRecord + ExecutionStrategy permiten sustituir modelos sin reescribir lógica de ejecución (abstracción probada con 3 estrategias distintas).",
  },
  {
    title: "Corpus estructurado de noticias USD/COP",
    body: "Pipeline multi-fuente (Investing, Portafolio, GDELT) enriquecido con categorización + sentimiento + cross-reference para integración ML futura.",
  },
];

export const T32_Contribuciones: React.FC<ThesisProps> = ({
  variant: _variant,
}) => {
  const frame = useCurrentFrame();

  const titleOp = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const closerOp = interpolate(frame, [1060, 1140], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitFade = interpolate(frame, [1170, 1200], [1, 0], {
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
            color: TOKENS.colors.market.up,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          CAPÍTULO VI · APORTES ORIGINALES
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
          Seis contribuciones al estado del arte
        </div>
      </div>

      {/* Grid */}
      <div
        style={{
          position: "absolute",
          top: 200,
          left: 80,
          right: 80,
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 16,
        }}
      >
        {CONTRIBUTIONS.map((c, i) => {
          const appearAt = 90 + i * 140;
          const op = interpolate(frame, [appearAt, appearAt + 60], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const y = interpolate(frame, [appearAt, appearAt + 60], [30, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                opacity: op,
                transform: `translateY(${y}px)`,
                padding: "20px 22px",
                background: `linear-gradient(135deg, ${TOKENS.colors.market.up}18, ${TOKENS.colors.accent.cyan}10)`,
                border: `1px solid ${TOKENS.colors.market.up}55`,
                borderRadius: 10,
                minHeight: 240,
              }}
            >
              <div
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 8,
                  background: TOKENS.colors.market.up,
                  color: "#000",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 20,
                  fontFamily: "JetBrains Mono, monospace",
                  fontWeight: 700,
                  marginBottom: 14,
                }}
              >
                {i + 1}
              </div>
              <div
                style={{
                  fontSize: 22,
                  fontFamily: "Inter, system-ui",
                  fontWeight: 700,
                  color: "#fff",
                  letterSpacing: -0.3,
                  lineHeight: 1.2,
                  marginBottom: 10,
                }}
              >
                {c.title}
              </div>
              <div
                style={{
                  fontSize: 15,
                  fontFamily: "Inter, system-ui",
                  color: TOKENS.colors.text.secondary,
                  lineHeight: 1.45,
                }}
              >
                {c.body}
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 80,
          right: 80,
          padding: "14px 24px",
          background: `${TOKENS.colors.market.up}18`,
          border: `1px solid ${TOKENS.colors.market.up}`,
          borderRadius: 10,
          opacity: closerOp,
          fontSize: 20,
          fontFamily: "Inter, system-ui",
          color: "#fff",
          textAlign: "center",
        }}
      >
        <b style={{ color: TOKENS.colors.market.up }}>
          Cada contribución es falsable:
        </b>{" "}
        archivos específicos, datos auditables, configs frozen y resultados
        reproducibles.
      </div>
    </AbsoluteFill>
  );
};
