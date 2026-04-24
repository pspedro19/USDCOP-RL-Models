import React from "react";
import {
  AbsoluteFill,
  Sequence,
  staticFile,
  OffthreadVideo,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import { Typewriter } from "../../components/Typewriter";
import { MetricCounter } from "../../components/MetricCounter";
import { PITCH_METRICS } from "../../data/metrics";
import type { PitchProps } from "../../compositions/Pitch";

/**
 * P07 · SignalBridge + Airflow Deep Tour + API (35s · 1050 frames)
 *
 * Stages:
 *   0–240f    (0-8s)   : SignalBridge UI (dashboard + exchanges)
 *   240–900f  (8-30s)  : Airflow DEEP tour (login → DAGs → click → Code → scroll)
 *   900–1050f (30-35s) : "Conecta cualquier exchange" + API cards
 */
export const P07_SignalBridge: React.FC<PitchProps> = () => {
  const frame = useCurrentFrame();
  const m = PITCH_METRICS.infra;

  return (
    <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
      {/* Stage 1 (0-250f · 8.3s): SignalBridge clip full-bleed */}
      <Sequence from={0} durationInFrames={250} layout="none">
        <AbsoluteFill
          style={{
            opacity: interpolate(frame, [230, 248], [1, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <OffthreadVideo
            src={staticFile("captures/S08-signalbridge.webm")}
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: "brightness(0.95)",
            }}
          />
          <AbsoluteFill
            style={{
              background:
                "linear-gradient(180deg, rgba(0,0,0,0.55) 0%, transparent 18%, transparent 82%, rgba(0,0,0,0.55) 100%)",
              pointerEvents: "none",
            }}
          />
          <div
            style={{
              position: "absolute",
              top: 32,
              left: 48,
              display: "flex",
              flexDirection: "column",
              gap: 6,
              opacity: interpolate(frame, [0, 15], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <div
              style={{
                fontSize: 14,
                color: TOKENS.colors.market.up,
                letterSpacing: 5,
                textTransform: "uppercase",
                fontFamily: "Inter, system-ui",
                fontWeight: 700,
              }}
            >
              Caso 5 · Ejecución
            </div>
            <div
              style={{
                fontSize: 28,
                fontFamily: "Inter, system-ui",
                fontWeight: 700,
                background: `linear-gradient(135deg, ${TOKENS.colors.market.up}, ${TOKENS.colors.accent.cyan})`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              SignalBridge · OMS + Riesgo
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* Stage 2 (240-905f · 22.2s): AIRFLOW DEEP TOUR */}
      <Sequence from={240} durationInFrames={670} layout="none">
        <AbsoluteFill
          style={{
            opacity: interpolate(frame, [240, 262, 880, 905], [0, 1, 1, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <OffthreadVideo
            src={staticFile("captures/I04-airflow-dags.webm")}
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: "brightness(0.98)",
            }}
          />
          {/* Top gradient for label */}
          <AbsoluteFill
            style={{
              background:
                "linear-gradient(180deg, rgba(0,0,0,0.65) 0%, transparent 14%, transparent 78%, rgba(0,0,0,0.75) 100%)",
              pointerEvents: "none",
            }}
          />
          {/* Top label */}
          <div
            style={{
              position: "absolute",
              top: 32,
              left: 48,
              display: "flex",
              flexDirection: "column",
              gap: 6,
              opacity: interpolate(frame, [240, 270], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <div
              style={{
                fontSize: 14,
                color: TOKENS.colors.accent.cyan,
                letterSpacing: 5,
                textTransform: "uppercase",
                fontFamily: "Inter, system-ui",
                fontWeight: 700,
              }}
            >
              Orquestación · Airflow
            </div>
            <div
              style={{
                fontSize: 26,
                fontFamily: "Inter, system-ui",
                fontWeight: 700,
                background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.accent.purple})`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              37 DAGs · ejecución semanal reproducible
            </div>
          </div>

          {/* Bottom panel: metrics */}
          <div
            style={{
              position: "absolute",
              bottom: 36,
              left: "50%",
              transform: "translateX(-50%)",
              display: "flex",
              gap: 40,
              padding: "14px 30px",
              background: "rgba(0,0,0,0.82)",
              borderRadius: 12,
              border: `1px solid ${TOKENS.colors.accent.cyan}55`,
            }}
          >
            <MetricCounter
              target={m.airflow_dags}
              format="int"
              delay={30}
              label="DAGs Airflow"
              fontSize={40}
              color={TOKENS.colors.accent.cyan}
              labelSize={13}
            />
            <MetricCounter
              target={m.docker_services}
              format="int"
              delay={42}
              label="Servicios Docker"
              fontSize={40}
              color={TOKENS.colors.market.up}
              labelSize={13}
            />
            <MetricCounter
              target={m.db_migrations}
              format="int"
              delay={54}
              label="Migrations DB"
              fontSize={40}
              color={TOKENS.colors.accent.purple}
              labelSize={13}
            />
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* Stage 3 (900-1050f · 5s): API integration callout */}
      <Sequence from={900} layout="none">
        <AbsoluteFill
          style={{
            background: `radial-gradient(ellipse at center, ${TOKENS.colors.bg.primary}, ${TOKENS.colors.bg.deep})`,
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 22,
            padding: 50,
          }}
        >
          <div
            style={{
              fontSize: 18,
              color: TOKENS.colors.accent.cyan,
              letterSpacing: 5,
              textTransform: "uppercase",
              fontFamily: "Inter, system-ui",
              fontWeight: 700,
              opacity: interpolate(frame, [900, 915], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            Conecta cualquier exchange
          </div>

          <Typewriter
            text="MEXC · Binance · Kraken · Coinbase · + Custom API"
            speed={40}
            delay={0}
            fontSize={34}
            fontWeight={700}
            gradient={[TOKENS.colors.accent.cyan, TOKENS.colors.accent.purple]}
            cursor={false}
          />

          <div
            style={{
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
              justifyContent: "center",
              opacity: interpolate(frame, [945, 965], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            {[
              { label: "AES-256-GCM", sub: "Cifrado credentials" },
              { label: "CCXT", sub: "100+ exchanges" },
              { label: "9-check chain", sub: "Risk pre-trade" },
              { label: "Kill Switch", sub: "Pausa instantánea" },
            ].map((c, i) => (
              <div
                key={c.label}
                style={{
                  padding: "14px 20px",
                  background: TOKENS.colors.bg.card,
                  border: `1px solid ${TOKENS.colors.accent.cyan}40`,
                  borderRadius: 12,
                  textAlign: "center",
                  minWidth: 150,
                  opacity: interpolate(frame, [950 + i * 4, 968 + i * 4], [0, 1], {
                    extrapolateLeft: "clamp",
                    extrapolateRight: "clamp",
                  }),
                  transform: `translateY(${interpolate(
                    frame,
                    [950 + i * 4, 968 + i * 4],
                    [14, 0],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  )}px)`,
                }}
              >
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 800,
                    color: TOKENS.colors.accent.cyan,
                    fontFamily: "JetBrains Mono, monospace",
                    marginBottom: 2,
                  }}
                >
                  {c.label}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: TOKENS.colors.text.secondary,
                    letterSpacing: 1.5,
                    textTransform: "uppercase",
                  }}
                >
                  {c.sub}
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              fontSize: 13,
              color: TOKENS.colors.text.muted,
              fontFamily: "Inter, system-ui",
              marginTop: 4,
              letterSpacing: 2,
              opacity: interpolate(frame, [990, 1010], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            ¿Otra API? La arquitectura CCXT soporta integraciones custom.
          </div>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
