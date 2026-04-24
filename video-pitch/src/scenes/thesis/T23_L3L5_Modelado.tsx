import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T23_L3L5_Modelado: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <Sequence from={0} durationInFrames={625} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.forecast_h5_l3}
        layerLabel="L3 · Entrenamiento"
        dagName="forecast_h5_l3_weekly_training"
        dagDescription="Ridge + BayesianRidge · expanding window 2020 → last Friday"
        accent={TOKENS.colors.accent.purple}
      />
    </Sequence>
    <Sequence from={625} durationInFrames={475} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.forecast_h5_l5_signal}
        layerLabel="L5 · Señal Semanal"
        dagName="forecast_h5_l5_weekly_signal"
        dagDescription="Ensemble mean + confidence scoring · 3-tier HIGH/MED/LOW"
        accent={TOKENS.colors.accent.purple}
      />
    </Sequence>
    <Sequence from={1100} durationInFrames={700} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.forecast_h5_l5_vol_targeting}
        layerLabel="L5 · Régimen Gate"
        dagName="forecast_h5_l5_vol_targeting"
        dagDescription="Hurst R/S · classify regime · effective HS · dynamic leverage"
        accent="#ef4444"
        isHighlight
      />
    </Sequence>
  </AbsoluteFill>
);
