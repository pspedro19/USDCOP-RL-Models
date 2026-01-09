"use client";

import { useHotkeys } from "react-hotkeys-hook";
import { useCallback, useRef } from "react";
import { toast } from "react-hot-toast";

interface ShortcutAction {
  key: string;
  description: string;
  category: string;
  action: () => void;
  enabled?: boolean;
}

interface UseKeyboardShortcutsProps {
  onTimeframeChange?: (timeframe: string) => void;
  onDrawingTool?: (tool: string) => void;
  onTradingAction?: (action: string) => void;
  onChartAction?: (action: string) => void;
  onSystemAction?: (action: string) => void;
  enabled?: boolean;
}

export const useKeyboardShortcuts = ({
  onTimeframeChange,
  onDrawingTool,
  onTradingAction,
  onChartAction,
  onSystemAction,
  enabled = true,
}: UseKeyboardShortcutsProps = {}) => {
  const lastActionRef = useRef<string>("");

  const showShortcutToast = useCallback((action: string, description: string) => {
    if (lastActionRef.current !== action) {
      toast.success(`${description}`, {
        duration: 1500,
        position: "bottom-right",
        style: {
          background: "#1f2937",
          color: "#f9fafb",
          border: "1px solid #374151",
        },
      });
      lastActionRef.current = action;
      setTimeout(() => {
        lastActionRef.current = "";
      }, 1000);
    }
  }, []);

  // Timeframe shortcuts
  useHotkeys("1", () => {
    if (!enabled) return;
    onTimeframeChange?.("1m");
    showShortcutToast("1m", "Switched to 1 Minute");
  }, { enabled, preventDefault: true });

  useHotkeys("5", () => {
    if (!enabled) return;
    onTimeframeChange?.("5m");
    showShortcutToast("5m", "Switched to 5 Minutes");
  }, { enabled, preventDefault: true });

  useHotkeys("1,5", () => {
    if (!enabled) return;
    onTimeframeChange?.("15m");
    showShortcutToast("15m", "Switched to 15 Minutes");
  }, { enabled, preventDefault: true });

  useHotkeys("h", () => {
    if (!enabled) return;
    onTimeframeChange?.("1h");
    showShortcutToast("1h", "Switched to 1 Hour");
  }, { enabled, preventDefault: true });

  useHotkeys("4,h", () => {
    if (!enabled) return;
    onTimeframeChange?.("4h");
    showShortcutToast("4h", "Switched to 4 Hours");
  }, { enabled, preventDefault: true });

  useHotkeys("d", () => {
    if (!enabled) return;
    onTimeframeChange?.("1d");
    showShortcutToast("1d", "Switched to 1 Day");
  }, { enabled, preventDefault: true });

  useHotkeys("w", () => {
    if (!enabled) return;
    onTimeframeChange?.("1w");
    showShortcutToast("1w", "Switched to 1 Week");
  }, { enabled, preventDefault: true });

  // Drawing tools shortcuts
  useHotkeys("t", () => {
    if (!enabled) return;
    onDrawingTool?.("trendline");
    showShortcutToast("trendline", "Trend Line Tool");
  }, { enabled, preventDefault: true });

  useHotkeys("l", () => {
    if (!enabled) return;
    onDrawingTool?.("line");
    showShortcutToast("line", "Line Tool");
  }, { enabled, preventDefault: true });

  useHotkeys("f", () => {
    if (!enabled) return;
    onDrawingTool?.("fibonacci");
    showShortcutToast("fibonacci", "Fibonacci Tool");
  }, { enabled, preventDefault: true });

  useHotkeys("r", () => {
    if (!enabled) return;
    onDrawingTool?.("rectangle");
    showShortcutToast("rectangle", "Rectangle Tool");
  }, { enabled, preventDefault: true });

  useHotkeys("c", () => {
    if (!enabled) return;
    onDrawingTool?.("circle");
    showShortcutToast("circle", "Circle Tool");
  }, { enabled, preventDefault: true });

  // Trading shortcuts
  useHotkeys("b", () => {
    if (!enabled) return;
    onTradingAction?.("buy");
    showShortcutToast("buy", "Buy Order");
  }, { enabled, preventDefault: true });

  useHotkeys("s", () => {
    if (!enabled) return;
    onTradingAction?.("sell");
    showShortcutToast("sell", "Sell Order");
  }, { enabled, preventDefault: true });

  useHotkeys("x", () => {
    if (!enabled) return;
    onTradingAction?.("close");
    showShortcutToast("close", "Close Position");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+c", () => {
    if (!enabled) return;
    onTradingAction?.("cancel");
    showShortcutToast("cancel", "Cancel Orders");
  }, { enabled, preventDefault: true });

  // Chart actions
  useHotkeys("mod+s", () => {
    if (!enabled) return;
    onChartAction?.("save");
    showShortcutToast("save", "Chart Saved");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+z", () => {
    if (!enabled) return;
    onChartAction?.("undo");
    showShortcutToast("undo", "Undo");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+shift+z", () => {
    if (!enabled) return;
    onChartAction?.("redo");
    showShortcutToast("redo", "Redo");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+a", () => {
    if (!enabled) return;
    onChartAction?.("selectAll");
    showShortcutToast("selectAll", "Select All");
  }, { enabled, preventDefault: true });

  useHotkeys("delete", () => {
    if (!enabled) return;
    onChartAction?.("delete");
    showShortcutToast("delete", "Deleted");
  }, { enabled, preventDefault: true });

  useHotkeys("escape", () => {
    if (!enabled) return;
    onChartAction?.("escape");
    showShortcutToast("escape", "Tool Deselected");
  }, { enabled, preventDefault: true });

  // System shortcuts
  useHotkeys("mod+,", () => {
    if (!enabled) return;
    onSystemAction?.("preferences");
    showShortcutToast("preferences", "Preferences");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+/", () => {
    if (!enabled) return;
    onSystemAction?.("help");
    showShortcutToast("help", "Help");
  }, { enabled, preventDefault: true });

  useHotkeys("f11", () => {
    if (!enabled) return;
    onSystemAction?.("fullscreen");
    showShortcutToast("fullscreen", "Fullscreen");
  }, { enabled, preventDefault: true });

  useHotkeys("mod+shift+d", () => {
    if (!enabled) return;
    onSystemAction?.("darkMode");
    showShortcutToast("darkMode", "Dark Mode Toggle");
  }, { enabled, preventDefault: true });

  // Navigation shortcuts
  useHotkeys("space", () => {
    if (!enabled) return;
    onChartAction?.("pan");
    showShortcutToast("pan", "Pan Mode");
  }, { enabled, preventDefault: true });

  useHotkeys("+", () => {
    if (!enabled) return;
    onChartAction?.("zoomIn");
    showShortcutToast("zoomIn", "Zoom In");
  }, { enabled, preventDefault: true });

  useHotkeys("-", () => {
    if (!enabled) return;
    onChartAction?.("zoomOut");
    showShortcutToast("zoomOut", "Zoom Out");
  }, { enabled, preventDefault: true });

  useHotkeys("0", () => {
    if (!enabled) return;
    onChartAction?.("resetZoom");
    showShortcutToast("resetZoom", "Reset Zoom");
  }, { enabled, preventDefault: true });

  // Return shortcuts configuration for help system
  const shortcuts: ShortcutAction[] = [
    // Timeframes
    { key: "1", description: "Switch to 1 Minute", category: "Timeframes", action: () => onTimeframeChange?.("1m") },
    { key: "5", description: "Switch to 5 Minutes", category: "Timeframes", action: () => onTimeframeChange?.("5m") },
    { key: "1,5", description: "Switch to 15 Minutes", category: "Timeframes", action: () => onTimeframeChange?.("15m") },
    { key: "H", description: "Switch to 1 Hour", category: "Timeframes", action: () => onTimeframeChange?.("1h") },
    { key: "4,H", description: "Switch to 4 Hours", category: "Timeframes", action: () => onTimeframeChange?.("4h") },
    { key: "D", description: "Switch to 1 Day", category: "Timeframes", action: () => onTimeframeChange?.("1d") },
    { key: "W", description: "Switch to 1 Week", category: "Timeframes", action: () => onTimeframeChange?.("1w") },

    // Drawing Tools
    { key: "T", description: "Trend Line Tool", category: "Drawing Tools", action: () => onDrawingTool?.("trendline") },
    { key: "L", description: "Line Tool", category: "Drawing Tools", action: () => onDrawingTool?.("line") },
    { key: "F", description: "Fibonacci Tool", category: "Drawing Tools", action: () => onDrawingTool?.("fibonacci") },
    { key: "R", description: "Rectangle Tool", category: "Drawing Tools", action: () => onDrawingTool?.("rectangle") },
    { key: "C", description: "Circle Tool", category: "Drawing Tools", action: () => onDrawingTool?.("circle") },

    // Trading
    { key: "B", description: "Buy Order", category: "Trading", action: () => onTradingAction?.("buy") },
    { key: "S", description: "Sell Order", category: "Trading", action: () => onTradingAction?.("sell") },
    { key: "X", description: "Close Position", category: "Trading", action: () => onTradingAction?.("close") },
    { key: "Cmd+C", description: "Cancel Orders", category: "Trading", action: () => onTradingAction?.("cancel") },

    // Chart Actions
    { key: "Cmd+S", description: "Save Chart", category: "Chart", action: () => onChartAction?.("save") },
    { key: "Cmd+Z", description: "Undo", category: "Chart", action: () => onChartAction?.("undo") },
    { key: "Cmd+Shift+Z", description: "Redo", category: "Chart", action: () => onChartAction?.("redo") },
    { key: "Cmd+A", description: "Select All", category: "Chart", action: () => onChartAction?.("selectAll") },
    { key: "Delete", description: "Delete Selected", category: "Chart", action: () => onChartAction?.("delete") },
    { key: "Escape", description: "Deselect Tool", category: "Chart", action: () => onChartAction?.("escape") },

    // Navigation
    { key: "Space", description: "Pan Mode", category: "Navigation", action: () => onChartAction?.("pan") },
    { key: "+", description: "Zoom In", category: "Navigation", action: () => onChartAction?.("zoomIn") },
    { key: "-", description: "Zoom Out", category: "Navigation", action: () => onChartAction?.("zoomOut") },
    { key: "0", description: "Reset Zoom", category: "Navigation", action: () => onChartAction?.("resetZoom") },

    // System
    { key: "Cmd+K", description: "Command Palette", category: "System", action: () => onSystemAction?.("commandPalette") },
    { key: "Cmd+,", description: "Preferences", category: "System", action: () => onSystemAction?.("preferences") },
    { key: "Cmd+/", description: "Help", category: "System", action: () => onSystemAction?.("help") },
    { key: "F11", description: "Fullscreen", category: "System", action: () => onSystemAction?.("fullscreen") },
    { key: "Cmd+Shift+D", description: "Toggle Dark Mode", category: "System", action: () => onSystemAction?.("darkMode") },
  ];

  return { shortcuts };
};