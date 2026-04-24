import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  staticFile,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { TOKENS } from "../../theme/tokens";

export interface DagCodeSceneProps {
  /** Starting frame inside the I04 webm (global tour timestamp × 30fps). */
  webmStartFrame: number;
  /** Layer label (e.g., "L5 · Régimen Gate") */
  layerLabel: string;
  /** DAG name (e.g., "forecast_h5_l5_vol_targeting") */
  dagName: string;
  /** Brief description shown at bottom */
  dagDescription: string;
  /** Accent color */
  accent?: string;
  /** Is this the MVP / highlighted DAG? */
  isHighlight?: boolean;
}

/**
 * Reusable scene that shows a specific DAG's code from the I04 tour webm.
 * Uses startFrom to jump directly to the right segment of the long tour.
 *
 * Each DAG segment in the webm is ~22 seconds long:
 *   0-12s   : login + DAG list
 *   12-34s  : core_l0_01 (segment 0)
 *   34-56s  : core_l0_03 (segment 1)
 *   56-78s  : forecast_h5_l3 (segment 2) — training MVP
 *   78-100s : forecast_h5_l5_signal (segment 3)
 *   100-122s: forecast_h5_l5_vol_targeting (segment 4) — regime gate MVP ⭐
 *   122-144s: forecast_h5_l7_multiday_executor (segment 5)
 *   144-166s: forecast_h5_l6_weekly_monitor (segment 6)
 *   166-188s: news_daily_pipeline (segment 7)
 *   188-210s: analysis_l8_daily_generation (segment 8)
 *   210-232s: core_watchdog (segment 9)
 */
export const DagCodeScene: React.FC<DagCodeSceneProps> = ({
  webmStartFrame,
  layerLabel,
  dagName,
  dagDescription,
  accent = TOKENS.colors.accent.cyan,
  isHighlight = false,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Fade in top/bottom labels
  const labelOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Fade out at the end
  const endOpacity = interpolate(
    frame,
    [durationInFrames - 15, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill style={{ background: "#fff", opacity: endOpacity }}>
      {/* Airflow capture full-bleed */}
      <AbsoluteFill>
        <OffthreadVideo
          src={staticFile("captures/I04-airflow-code-tour.webm")}
          muted
          startFrom={webmStartFrame}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </AbsoluteFill>

      {/* Top dark gradient for label readability */}
      <AbsoluteFill
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.62) 0%, transparent 16%, transparent 84%, rgba(0,0,0,0.75) 100%)",
          pointerEvents: "none",
        }}
      />

      {/* Top-left badge */}
      <div
        style={{
          position: "absolute",
          top: 28,
          left: 40,
          display: "flex",
          flexDirection: "column",
          gap: 4,
          opacity: labelOpacity,
        }}
      >
        <div
          style={{
            fontSize: 13,
            color: accent,
            letterSpacing: 5,
            textTransform: "uppercase",
            fontFamily: "Inter, system-ui",
            fontWeight: 700,
          }}
        >
          {layerLabel}
          {isHighlight && " ★"}
        </div>
        <div
          style={{
            fontSize: 22,
            fontFamily: "JetBrains Mono, monospace",
            fontWeight: 700,
            color: "#fff",
            textShadow: "0 2px 8px rgba(0,0,0,0.8)",
          }}
        >
          {dagName}
        </div>
      </div>

      {/* Bottom description */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: "50%",
          transform: "translateX(-50%)",
          padding: "12px 22px",
          background: "rgba(0,0,0,0.78)",
          borderRadius: 10,
          border: `1px solid ${accent}66`,
          fontSize: 18,
          color: "#fff",
          letterSpacing: 1.5,
          fontFamily: "Inter, system-ui",
          opacity: labelOpacity,
          textAlign: "center",
          maxWidth: 900,
        }}
      >
        {dagDescription}
      </div>
    </AbsoluteFill>
  );
};

/** Segment start frames (in I04 tour webm) for each DAG */
export const DAG_SEGMENTS = {
  login_list: 0,                            // 0-12s
  core_l0_01_ohlcv: 12 * 30,                // 12-34s · 360f
  core_l0_03_macro: 34 * 30,                // 34-56s · 1020f
  forecast_h5_l3: 56 * 30,                  // 56-78s · 1680f
  forecast_h5_l5_signal: 78 * 30,           // 78-100s · 2340f
  forecast_h5_l5_vol_targeting: 100 * 30,   // 100-122s · 3000f (regime gate ⭐)
  forecast_h5_l7: 122 * 30,                 // 122-144s · 3660f
  forecast_h5_l6: 144 * 30,                 // 144-166s · 4320f
  news_daily: 166 * 30,                     // 166-188s · 4980f
  analysis_l8: 188 * 30,                    // 188-210s · 5640f
  core_watchdog: 210 * 30,                  // 210-232s · 6300f
} as const;
