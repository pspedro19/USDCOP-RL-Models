import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { TOKENS } from "../../theme/tokens";
import type { ThesisProps } from "../../compositions/Thesis";

/**
 * T28 · PgAdmin (psql) · 30s (900f)
 * Terminal-style reveal of four SQL queries against the trading DB.
 */
type Query = {
  prompt: string;
  sql: string[];
  header: string;
  rows: string[];
  appearsAt: number;
};

const QUERIES: Query[] = [
  {
    prompt: "usdcop_trading=#",
    sql: [
      "SELECT tablename FROM pg_tables",
      "WHERE schemaname='public'",
      "AND tablename LIKE 'forecast_h5%';",
    ],
    header: "         tablename          ",
    rows: [
      " forecast_h5_predictions",
      " forecast_h5_signals",
      " forecast_h5_executions",
      " forecast_h5_subtrades",
      " forecast_h5_paper_trading",
      "(5 rows)",
    ],
    appearsAt: 30,
  },
  {
    prompt: "usdcop_trading=#",
    sql: [
      "SELECT signal_date, direction, confidence_tier,",
      "  hurst, regime, adjusted_leverage",
      "FROM forecast_h5_signals",
      "ORDER BY signal_date DESC LIMIT 3;",
    ],
    header:
      " signal_date | direction | confidence | hurst | regime      | lev ",
    rows: [
      " 2026-04-07  | SKIP      | NULL       | 0.41  | MEAN_REVERT | 0.00",
      " 2026-03-31  | SKIP      | NULL       | 0.38  | MEAN_REVERT | 0.00",
      " 2026-03-24  | SHORT     | HIGH       | 0.44  | INDETERM    | 0.60",
      "(3 rows)",
    ],
    appearsAt: 240,
  },
  {
    prompt: "usdcop_trading=#",
    sql: [
      "SELECT COUNT(*) AS n,",
      "  SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END) AS wins,",
      "  ROUND(AVG(pnl_pct)::numeric, 3) AS avg_pnl",
      "FROM forecast_h5_executions",
      "WHERE EXTRACT(YEAR FROM signal_date)=2025;",
    ],
    header: "  n | wins | avg_pnl ",
    rows: [" 34 |   28 |   0.768", "(1 row)"],
    appearsAt: 470,
  },
  {
    prompt: "usdcop_trading=#",
    sql: [
      "SELECT source_id, COUNT(*) AS articles",
      "FROM news_articles GROUP BY source_id;",
    ],
    header: "   source_id   | articles ",
    rows: [
      " investing     |       78",
      " portafolio    |      283",
      "(2 rows)",
    ],
    appearsAt: 690,
  },
];

export const T28_PgAdmin: React.FC<ThesisProps> = ({ variant: _variant }) => {
  const frame = useCurrentFrame();

  const fadeIn = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(frame, [870, 900], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const cursor = Math.floor(frame / 15) % 2 === 0 ? "▊" : " ";

  return (
    <AbsoluteFill
      style={{
        background: TOKENS.colors.bg.deep,
        opacity: fadeIn * fadeOut,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 28,
          left: 40,
          display: "flex",
          flexDirection: "column",
          gap: 4,
        }}
      >
        <div
          style={{
            fontSize: 13,
            color: TOKENS.colors.accent.cyan,
            letterSpacing: 5,
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          DATOS REALES · POSTGRESQL + TIMESCALEDB
        </div>
        <div
          style={{
            fontSize: 22,
            fontFamily: "JetBrains Mono, monospace",
            fontWeight: 700,
            color: "#fff",
            letterSpacing: -0.5,
          }}
        >
          psql · usdcop_trading
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          top: 120,
          left: 80,
          right: 80,
          bottom: 80,
          background: "#0b1020",
          border: `1px solid ${TOKENS.colors.accent.cyan}55`,
          borderRadius: 12,
          padding: 32,
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 20,
          lineHeight: 1.45,
          color: "#E5E7EB",
          overflow: "hidden",
          boxShadow: "0 20px 60px rgba(6,182,212,0.2)",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 16,
            display: "flex",
            gap: 8,
          }}
        >
          {["#ff5f57", "#febc2e", "#28c840"].map((c, i) => (
            <div
              key={i}
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: c,
              }}
            />
          ))}
        </div>

        <div style={{ marginTop: 12 }}>
          <span style={{ color: TOKENS.colors.text.secondary }}>
            $ docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading
          </span>
          <br />
          <span style={{ color: TOKENS.colors.text.muted }}>
            psql (15.4). Type &quot;help&quot; for help.
          </span>
        </div>

        {QUERIES.map((q, idx) => {
          const queryFrame = frame - q.appearsAt;
          if (queryFrame < 0) return null;

          const linesRevealed = Math.min(
            q.sql.length,
            Math.floor(queryFrame / 8)
          );
          const resultsAt = q.sql.length * 8 + 20;
          const showResults = queryFrame >= resultsAt;
          const onLine = linesRevealed < q.sql.length;
          const isLastQuery = idx === QUERIES.length - 1;

          return (
            <div key={idx} style={{ marginTop: 24 }}>
              {q.sql.slice(0, linesRevealed + (onLine ? 1 : 0)).map((line, i) => {
                const isCurrent = i === linesRevealed && onLine;
                return (
                  <div key={i}>
                    <span style={{ color: TOKENS.colors.accent.cyan }}>
                      {i === 0 ? q.prompt : "   ...>"}{" "}
                    </span>
                    <span style={{ color: "#E5E7EB" }}>
                      {line}
                      {isCurrent ? cursor : ""}
                    </span>
                  </div>
                );
              })}

              {showResults && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ color: TOKENS.colors.accent.purple }}>
                    {q.header}
                  </div>
                  <div style={{ color: TOKENS.colors.text.secondary }}>
                    {"-".repeat(q.header.length)}
                  </div>
                  {q.rows.map((r, i) => (
                    <div
                      key={i}
                      style={{
                        color: r.startsWith("(")
                          ? TOKENS.colors.text.muted
                          : "#E5E7EB",
                      }}
                    >
                      {r}
                    </div>
                  ))}
                  {isLastQuery && (
                    <div style={{ marginTop: 6 }}>
                      <span style={{ color: TOKENS.colors.accent.cyan }}>
                        {q.prompt}{" "}
                      </span>
                      <span>{cursor}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 24,
          left: "50%",
          transform: "translateX(-50%)",
          fontSize: 16,
          fontFamily: "JetBrains Mono, monospace",
          color: TOKENS.colors.text.secondary,
          letterSpacing: 2,
          opacity: interpolate(frame, [60, 120], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Cada número de esta tesis es auditable desde la base de datos.
      </div>
    </AbsoluteFill>
  );
};
