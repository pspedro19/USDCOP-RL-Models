import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T26_L8_Intelligence: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <Sequence from={0} durationInFrames={675} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.news_daily}
        layerLabel="L8 · News Engine"
        dagName="news_daily_pipeline"
        dagDescription="5 adapters · enrichment 5-stage · sentiment + NER + clusters"
        accent={TOKENS.colors.accent.purple}
      />
    </Sequence>
    <Sequence from={675} durationInFrames={675} layout="none">
      <DagCodeScene
        webmStartFrame={DAG_SEGMENTS.analysis_l8}
        layerLabel="L8 · IA Semanal"
        dagName="analysis_l8_daily_generation"
        dagDescription="LLMClient · GPT-4o primary + Anthropic fallback · $1/día budget"
        accent={TOKENS.colors.accent.purple}
      />
    </Sequence>
  </AbsoluteFill>
);
