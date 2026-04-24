import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../theme/tokens";

export const DisclaimerRoll: React.FC<{ delay?: number }> = ({ delay = 0 }) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - delay);
  const opacity = interpolate(local, [0, 20, 100, 140], [0, 1, 1, 0.95], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        maxWidth: 1200,
        textAlign: "center",
        fontSize: 20,
        lineHeight: 1.6,
        color: TOKENS.colors.text.muted,
        fontFamily: "Inter, system-ui",
        opacity,
        padding: "0 80px",
      }}
    >
      <div
        style={{
          fontSize: 14,
          letterSpacing: 4,
          textTransform: "uppercase",
          color: TOKENS.colors.semantic.warning,
          marginBottom: 10,
        }}
      >
        ⚠ Disclaimer
      </div>
      SignalBridge / Global Minds es una herramienta de análisis cuantitativo. Los
      resultados 2025-2026 corresponden a ejecuciones reales pero no garantizan
      rendimiento futuro. No constituye asesoría financiera. Trading conlleva
      riesgo de pérdida del capital invertido.
    </div>
  );
};
