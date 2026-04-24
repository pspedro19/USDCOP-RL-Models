import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { TOKENS } from "../theme/tokens";

export interface TypewriterProps {
  text: string;
  /** chars per second */
  speed?: number;
  /** delay (frames) before starting */
  delay?: number;
  /** show blinking cursor */
  cursor?: boolean;
  /** styling */
  fontSize?: number;
  fontWeight?: number;
  color?: string;
  gradient?: [string, string];
  letterSpacing?: number;
  lineHeight?: number;
  align?: "left" | "center" | "right";
  maxWidth?: number;
  fontFamily?: string;
}

export const Typewriter: React.FC<TypewriterProps> = ({
  text,
  speed = 28,
  delay = 0,
  cursor = true,
  fontSize = 54,
  fontWeight = 600,
  color,
  gradient,
  letterSpacing = 0,
  lineHeight = 1.15,
  align = "center",
  maxWidth,
  fontFamily = "Inter, system-ui",
}) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - delay);
  const fps = 30;
  const framesPerChar = fps / speed;
  const chars = Math.floor(local / framesPerChar);
  const visible = text.slice(0, Math.min(chars, text.length));
  const done = chars >= text.length;
  const cursorOn = cursor && Math.floor(frame / 14) % 2 === 0;

  const textStyle: React.CSSProperties = gradient
    ? {
        background: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`,
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
      }
    : { color: color ?? TOKENS.colors.text.primary };

  return (
    <div
      style={{
        fontFamily,
        fontSize,
        fontWeight,
        letterSpacing,
        lineHeight,
        textAlign: align,
        maxWidth,
        display: "inline-flex",
        alignItems: "baseline",
        ...textStyle,
      }}
    >
      <span>{visible}</span>
      {cursor && !done && cursorOn && (
        <span
          style={{
            display: "inline-block",
            width: Math.round(fontSize * 0.06) + 2,
            height: fontSize * 0.95,
            marginLeft: 4,
            background: TOKENS.colors.accent.cyan,
            verticalAlign: "text-bottom",
          }}
        />
      )}
    </div>
  );
};
