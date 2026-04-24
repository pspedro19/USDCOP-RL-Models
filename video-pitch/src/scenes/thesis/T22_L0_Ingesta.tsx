import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T22_L0_Ingesta: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <Sequence from={0} durationInFrames={675} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.core_l0_01_ohlcv}
        layerLabel="L0 · Ingesta OHLCV"
        dagName="core_l0_01_ohlcv_backfill"
        dagDescription="Ingesta de 3 pares FX · TwelveData API + circuit breaker + UPSERT postgres"
        accent={TOKENS.colors.accent.cyan}
      />
    </Sequence>
    <Sequence from={675} durationInFrames={675} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.core_l0_03_macro}
        layerLabel="L0 · Ingesta Macro"
        dagName="core_l0_03_macro_backfill"
        dagDescription="40 variables macro · 7 adapters (FRED, BanRep, BCRP, DANE...)"
        accent={TOKENS.colors.accent.cyan}
      />
    </Sequence>
  </AbsoluteFill>
);
