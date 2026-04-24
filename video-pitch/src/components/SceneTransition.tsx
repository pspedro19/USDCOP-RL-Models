import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface SceneTransitionProps {
  /** fade out in last N frames */
  fadeOutFrames?: number;
  /** fade in first N frames */
  fadeInFrames?: number;
  children: React.ReactNode;
  disabled?: boolean;
}

/** Fade-in + fade-out wrapper, scoped to parent Sequence duration */
export const SceneTransition: React.FC<SceneTransitionProps> = ({
  fadeOutFrames = 8,
  fadeInFrames = 8,
  children,
  disabled = false,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  if (disabled) {
    return <>{children}</>;
  }

  const fadeIn = interpolate(frame, [0, fadeInFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - fadeOutFrames, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic) }
  );
  const opacity = Math.min(fadeIn, fadeOut);

  return (
    <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>
  );
};

/** Visual whoosh effect: a cyan streak that sweeps across the screen.
 *  Pair with a whoosh SFX for max effect. */
export const WhooshOverlay: React.FC<{ delay?: number; durationInFrames?: number }> = ({
  delay = 0,
  durationInFrames = 18,
}) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - delay);
  const progress = interpolate(local, [0, durationInFrames], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const x = interpolate(progress, [0, 1], [-100, 120]);
  const opacity = interpolate(progress, [0, 0.2, 0.8, 1], [0, 0.85, 0.85, 0]);

  return (
    <AbsoluteFill style={{ pointerEvents: "none", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: `${x}%`,
          width: "30%",
          height: "100%",
          background: `linear-gradient(90deg, transparent, ${TOKENS.colors.accent.cyan}40, ${TOKENS.colors.accent.cyan}, ${TOKENS.colors.accent.cyan}40, transparent)`,
          filter: "blur(24px)",
          opacity,
          transform: "skewX(-12deg)",
        }}
      />
    </AbsoluteFill>
  );
};
