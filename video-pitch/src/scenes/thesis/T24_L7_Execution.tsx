import React from "react";
import { AbsoluteFill } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T24_L7_Execution: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <DagCodeScene
      webmStartFrame={DAG_SEGMENTS.forecast_h5_l7}
      layerLabel="L7 · Ejecución Semanal"
      dagName="forecast_h5_l7_multiday_executor"
      dagDescription="TP/HS monitor intradía · Friday close · subtrades + re-entry"
      accent={TOKENS.colors.market.up}
    />
  </AbsoluteFill>
);
