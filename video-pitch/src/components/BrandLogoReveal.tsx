import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface BrandLogoRevealProps {
  primary?: string;   // "Global Minds"
  secondary?: string; // "SignalBridge"
  delay?: number;
}

export const BrandLogoReveal: React.FC<BrandLogoRevealProps> = ({
  primary = "Global Minds",
  secondary = "SignalBridge",
  delay = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);

  // Letter-by-letter physics for primary
  const lettersPrimary = primary.split("");
  const lettersSecondary = secondary.split("");

  const getLetterTransform = (i: number, start: number, staggerPerLetter = 4) => {
    const letterLocal = Math.max(0, local - start - i * staggerPerLetter);
    const appear = spring({
      frame: letterLocal,
      fps,
      config: { damping: 14, stiffness: 220, mass: 0.55 },
      durationInFrames: 18,
    });
    return { scale: appear, opacity: Math.min(1, appear * 1.3) };
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          fontSize: 128,
          fontWeight: 800,
          fontFamily: "Inter, system-ui",
          letterSpacing: -3,
          background: `linear-gradient(135deg, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.accent.purple})`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        {lettersPrimary.map((ch, i) => {
          const { scale, opacity } = getLetterTransform(i, 0, 3);
          return (
            <span
              key={i}
              style={{
                display: "inline-block",
                transform: `scale(${scale})`,
                opacity,
                whiteSpace: "pre",
              }}
            >
              {ch}
            </span>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          fontSize: 44,
          fontWeight: 300,
          fontFamily: "Inter, system-ui",
          letterSpacing: 8,
          color: TOKENS.colors.text.secondary,
          textTransform: "uppercase",
        }}
      >
        {lettersSecondary.map((ch, i) => {
          const { scale, opacity } = getLetterTransform(
            i,
            lettersPrimary.length * 3 + 10,
            3
          );
          return (
            <span
              key={i}
              style={{
                display: "inline-block",
                transform: `scale(${scale})`,
                opacity,
                whiteSpace: "pre",
              }}
            >
              {ch}
            </span>
          );
        })}
      </div>
    </div>
  );
};
