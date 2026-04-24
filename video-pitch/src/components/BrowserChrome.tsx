import React from "react";
import { TOKENS } from "../theme/tokens";

export interface BrowserChromeProps {
  url?: string;
  children: React.ReactNode;
  width?: number;
  height?: number;
  radius?: number;
}

export const BrowserChrome: React.FC<BrowserChromeProps> = ({
  url = "https://dashboard.signalbridge.ai",
  children,
  width,
  height,
  radius = 14,
}) => {
  return (
    <div
      style={{
        width: width ?? "100%",
        height: height ?? "100%",
        background: TOKENS.colors.bg.secondary,
        borderRadius: radius,
        overflow: "hidden",
        border: `1px solid rgba(255,255,255,0.08)`,
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          height: 42,
          background: "#1a1a1a",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          alignItems: "center",
          padding: "0 16px",
          gap: 16,
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          <span style={{ width: 12, height: 12, borderRadius: 999, background: "#ff5f57" }} />
          <span style={{ width: 12, height: 12, borderRadius: 999, background: "#febc2e" }} />
          <span style={{ width: 12, height: 12, borderRadius: 999, background: "#28c840" }} />
        </div>
        <div
          style={{
            flex: 1,
            background: "#0a0a0a",
            borderRadius: 6,
            padding: "6px 12px",
            fontSize: 14,
            color: TOKENS.colors.text.secondary,
            fontFamily: "JetBrains Mono, monospace",
            letterSpacing: 0.5,
          }}
        >
          {url}
        </div>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>{children}</div>
    </div>
  );
};
