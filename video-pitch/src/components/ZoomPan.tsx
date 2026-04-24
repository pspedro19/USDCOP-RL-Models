import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  Easing,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export interface ZoomPanProps {
  /** path relative to public/ */
  src: string;
  /** starting transform */
  from?: { x: number; y: number; scale: number };
  /** ending transform */
  to?: { x: number; y: number; scale: number };
  /** optional override for total duration, else uses the composition duration */
  durationInFrames?: number;
  easing?: (t: number) => number;
  /** if true wrap in blurred glow backdrop */
  glow?: boolean;
}

export const ZoomPan: React.FC<ZoomPanProps> = ({
  src,
  from = { x: 0, y: 0, scale: 1 },
  to = { x: 0, y: 0, scale: 1.08 },
  durationInFrames,
  easing = Easing.inOut(Easing.cubic),
  glow = false,
}) => {
  const frame = useCurrentFrame();
  const cfg = useVideoConfig();
  const total = durationInFrames ?? cfg.durationInFrames;
  const progress = interpolate(frame, [0, total], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing,
  });
  const scale = interpolate(progress, [0, 1], [from.scale, to.scale]);
  const x = interpolate(progress, [0, 1], [from.x, to.x]);
  const y = interpolate(progress, [0, 1], [from.y, to.y]);

  return (
    <AbsoluteFill
      style={{
        overflow: "hidden",
        background: "#000",
      }}
    >
      {glow && (
        <AbsoluteFill
          style={{
            background: `radial-gradient(ellipse at center, rgba(6,182,212,0.15), transparent 60%)`,
            zIndex: 2,
            pointerEvents: "none",
          }}
        />
      )}
      <div
        style={{
          width: "100%",
          height: "100%",
          transform: `translate(${x}%, ${y}%) scale(${scale})`,
          transformOrigin: "center center",
          willChange: "transform",
        }}
      >
        <OffthreadVideo
          src={staticFile(src)}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
          muted
        />
      </div>
    </AbsoluteFill>
  );
};
