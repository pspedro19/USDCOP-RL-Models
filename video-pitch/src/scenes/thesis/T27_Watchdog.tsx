import React from "react";
import { AbsoluteFill } from "remotion";
import { TOKENS } from "../../theme/tokens";
import { DagCodeScene, DAG_SEGMENTS } from "./_DagCodeScene";
import type { ThesisProps } from "../../compositions/Thesis";

export const T27_Watchdog: React.FC<ThesisProps> = () => (
  <AbsoluteFill style={{ background: TOKENS.colors.bg.deep }}>
    <DagCodeScene
      webmStartFrame={DAG_SEGMENTS.core_watchdog}
      layerLabel="Watchdog · Auto-Recovery"
      dagName="core_watchdog"
      dagDescription="7 health checks · auto-heal stale data / forecasting / analysis / backups"
      accent={TOKENS.colors.accent.cyan}
    />
  </AbsoluteFill>
);
