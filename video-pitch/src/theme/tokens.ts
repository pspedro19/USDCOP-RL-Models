export const TOKENS = {
  colors: {
    bg: {
      primary: "#050816",
      secondary: "#0A0E27",
      deep: "#030712",
      card: "#0f172a",
    },
    accent: {
      cyan: "#06B6D4",
      purple: "#8B5CF6",
      indigo: "#6366f1",
    },
    market: {
      up: "#00D395",
      down: "#FF3B69",
      neutral: "#a1a1aa",
    },
    semantic: {
      success: "#22c55e",
      danger: "#ef4444",
      warning: "#f59e0b",
    },
    text: {
      primary: "#ffffff",
      secondary: "#a1a1aa",
      muted: "#52525b",
    },
  },
  typography: {
    display: { family: "Inter", weights: [800] },
    body: { family: "Inter", weights: [400, 600] },
    mono: { family: "JetBrains Mono", weights: [400, 600] },
  },
  timing: {
    fadeIn: 15,
    fadeOut: 12,
    stagger: 8,
    transition: 12,
  },
  safeZone: {
    landscape: { top: 60, bottom: 60, sides: 80 },
    vertical: { top: 150, bottom: 170, sides: 60 },
  },
  fontSize: {
    headline: 72,
    subhead: 48,
    body: 36,
    caption: 28,
  },
} as const;

export type Tokens = typeof TOKENS;
