import React from "react";
import { AbsoluteFill } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T25_L6_Monitoring: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <DagCodeScene
      webmStartFrame={DAG_SEGMENTS.forecast_h5_l6}
      layerLabel="L6 · Monitoreo + Guardrails"
      dagName="forecast_h5_l6_weekly_monitor"
      dagDescription="Circuit breaker · rolling DA · 5 consecutive losses detection"
      accent={TOKENS.colors.semantic.warning}
    />
  </AbsoluteFill>
);
