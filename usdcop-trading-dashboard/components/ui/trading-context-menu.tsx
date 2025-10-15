"use client";

import * as React from "react";
import * as ContextMenuPrimitive from "@radix-ui/react-context-menu";
import {
  TrendingUp,
  TrendingDown,
  Copy,
  Share2,
  Download,
  Settings,
  Eye,
  EyeOff,
  Trash2,
  Edit3,
  Target,
  BarChart3,
  LineChart,
  Zap,
  Layers,
  Palette,
  AlertCircle,
  Bell,
  BellOff,
  Bookmark,
  BookmarkX,
  Lock,
  Unlock,
  RefreshCw,
  Camera
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  shortcut?: string;
  action: () => void;
  disabled?: boolean;
  danger?: boolean;
  submenu?: ContextMenuItem[];
  separator?: boolean;
}

interface TradingContextMenuProps {
  children: React.ReactNode;
  items?: ContextMenuItem[];
  onItemSelect?: (item: ContextMenuItem) => void;
  disabled?: boolean;
  className?: string;
}

const defaultMenuItems: ContextMenuItem[] = [
  // Trading Actions
  {
    id: "buy",
    label: "Place Buy Order",
    icon: TrendingUp,
    shortcut: "B",
    action: () => console.log("Buy order"),
  },
  {
    id: "sell",
    label: "Place Sell Order",
    icon: TrendingDown,
    shortcut: "S",
    action: () => console.log("Sell order"),
  },
  {
    id: "separator1",
    label: "",
    separator: true,
    action: () => {},
  },

  // Price Actions
  {
    id: "copyPrice",
    label: "Copy Price",
    icon: Copy,
    shortcut: "Ctrl+C",
    action: () => console.log("Copy price"),
  },
  {
    id: "setAlert",
    label: "Set Price Alert",
    icon: Bell,
    action: () => console.log("Set alert"),
  },
  {
    id: "addBookmark",
    label: "Add Bookmark",
    icon: Bookmark,
    action: () => console.log("Add bookmark"),
  },
  {
    id: "separator2",
    label: "",
    separator: true,
    action: () => {},
  },

  // Drawing Tools
  {
    id: "drawingTools",
    label: "Drawing Tools",
    icon: Edit3,
    action: () => console.log("Drawing tools"),
    submenu: [
      {
        id: "trendLine",
        label: "Trend Line",
        icon: LineChart,
        shortcut: "T",
        action: () => console.log("Trend line"),
      },
      {
        id: "fibonacci",
        label: "Fibonacci",
        icon: Target,
        shortcut: "F",
        action: () => console.log("Fibonacci"),
      },
      {
        id: "rectangle",
        label: "Rectangle",
        icon: Layers,
        shortcut: "R",
        action: () => console.log("Rectangle"),
      },
      {
        id: "separator3",
        label: "",
        separator: true,
        action: () => {},
      },
      {
        id: "clearAll",
        label: "Clear All Drawings",
        icon: Trash2,
        danger: true,
        action: () => console.log("Clear all"),
      },
    ],
  },

  // Indicators
  {
    id: "indicators",
    label: "Technical Indicators",
    icon: BarChart3,
    action: () => console.log("Indicators"),
    submenu: [
      {
        id: "movingAverage",
        label: "Moving Average",
        action: () => console.log("Moving Average"),
      },
      {
        id: "rsi",
        label: "RSI",
        action: () => console.log("RSI"),
      },
      {
        id: "macd",
        label: "MACD",
        action: () => console.log("MACD"),
      },
      {
        id: "bollingerBands",
        label: "Bollinger Bands",
        action: () => console.log("Bollinger Bands"),
      },
      {
        id: "separator4",
        label: "",
        separator: true,
        action: () => {},
      },
      {
        id: "removeIndicators",
        label: "Remove All Indicators",
        icon: EyeOff,
        danger: true,
        action: () => console.log("Remove indicators"),
      },
    ],
  },
  {
    id: "separator5",
    label: "",
    separator: true,
    action: () => {},
  },

  // Chart Options
  {
    id: "chartOptions",
    label: "Chart Options",
    icon: Settings,
    action: () => console.log("Chart options"),
    submenu: [
      {
        id: "candlestick",
        label: "Candlestick",
        action: () => console.log("Candlestick"),
      },
      {
        id: "line",
        label: "Line Chart",
        action: () => console.log("Line chart"),
      },
      {
        id: "area",
        label: "Area Chart",
        action: () => console.log("Area chart"),
      },
      {
        id: "separator6",
        label: "",
        separator: true,
        action: () => {},
      },
      {
        id: "colors",
        label: "Color Scheme",
        icon: Palette,
        action: () => console.log("Colors"),
      },
      {
        id: "gridlines",
        label: "Toggle Grid Lines",
        icon: Eye,
        action: () => console.log("Grid lines"),
      },
    ],
  },

  // Export Options
  {
    id: "export",
    label: "Export & Share",
    icon: Share2,
    action: () => console.log("Export"),
    submenu: [
      {
        id: "screenshot",
        label: "Take Screenshot",
        icon: Camera,
        shortcut: "Ctrl+Shift+S",
        action: () => console.log("Screenshot"),
      },
      {
        id: "exportPDF",
        label: "Export as PDF",
        icon: Download,
        action: () => console.log("Export PDF"),
      },
      {
        id: "exportPNG",
        label: "Export as PNG",
        icon: Download,
        action: () => console.log("Export PNG"),
      },
      {
        id: "separator7",
        label: "",
        separator: true,
        action: () => {},
      },
      {
        id: "shareLink",
        label: "Share Chart Link",
        icon: Share2,
        action: () => console.log("Share link"),
      },
    ],
  },
  {
    id: "separator8",
    label: "",
    separator: true,
    action: () => {},
  },

  // Advanced Options
  {
    id: "refresh",
    label: "Refresh Data",
    icon: RefreshCw,
    shortcut: "F5",
    action: () => console.log("Refresh"),
  },
  {
    id: "lockChart",
    label: "Lock Chart",
    icon: Lock,
    action: () => console.log("Lock chart"),
  },
];

const TradingContextMenu = React.forwardRef<
  React.ElementRef<typeof ContextMenuPrimitive.Root>,
  TradingContextMenuProps
>(({ children, items = defaultMenuItems, onItemSelect, disabled = false, className }, ref) => {
  const handleItemSelect = React.useCallback((item: ContextMenuItem) => {
    if (item.disabled) return;

    item.action();
    onItemSelect?.(item);
  }, [onItemSelect]);

  const renderMenuItem = React.useCallback((item: ContextMenuItem, depth = 0) => {
    if (item.separator) {
      return (
        <ContextMenuPrimitive.Separator
          key={item.id}
          className="h-px bg-gray-700 my-1"
        />
      );
    }

    if (item.submenu) {
      return (
        <ContextMenuPrimitive.Sub key={item.id}>
          <ContextMenuPrimitive.SubTrigger
            className={cn(
              "flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none",
              "focus:bg-gray-800 focus:text-white",
              "data-[state=open]:bg-gray-800 data-[state=open]:text-white",
              item.disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center space-x-2">
                {item.icon && <item.icon className="h-4 w-4" />}
                <span>{item.label}</span>
              </div>
              <svg className="h-3 w-3 ml-2" viewBox="0 0 15 15" fill="none">
                <path
                  d="m6 3 3 3-3 3"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </ContextMenuPrimitive.SubTrigger>
          <ContextMenuPrimitive.Portal>
            <ContextMenuPrimitive.SubContent
              className={cn(
                "z-50 min-w-[8rem] overflow-hidden rounded-md border bg-gray-900 border-gray-700 p-1 shadow-lg",
                "data-[state=open]:animate-in data-[state=closed]:animate-out",
                "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
                "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
                "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
                "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2"
              )}
            >
              {item.submenu.map(subItem => renderMenuItem(subItem, depth + 1))}
            </ContextMenuPrimitive.SubContent>
          </ContextMenuPrimitive.Portal>
        </ContextMenuPrimitive.Sub>
      );
    }

    return (
      <ContextMenuPrimitive.Item
        key={item.id}
        className={cn(
          "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none",
          "focus:bg-gray-800 focus:text-white transition-colors duration-150",
          item.disabled && "opacity-50 cursor-not-allowed",
          item.danger && "focus:bg-red-900 focus:text-red-100"
        )}
        disabled={item.disabled}
        onClick={() => handleItemSelect(item)}
      >
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-2">
            {item.icon && (
              <item.icon
                className={cn(
                  "h-4 w-4",
                  item.danger && "text-red-400"
                )}
              />
            )}
            <span className={item.danger ? "text-red-200" : "text-gray-200"}>
              {item.label}
            </span>
          </div>
          {item.shortcut && (
            <kbd className="ml-auto text-xs text-gray-400 font-mono">
              {item.shortcut}
            </kbd>
          )}
        </div>
      </ContextMenuPrimitive.Item>
    );
  }, [handleItemSelect]);

  if (disabled) {
    return <>{children}</>;
  }

  return (
    <ContextMenuPrimitive.Root>
      <ContextMenuPrimitive.Trigger asChild>
        {children}
      </ContextMenuPrimitive.Trigger>

      <ContextMenuPrimitive.Portal>
        <ContextMenuPrimitive.Content
          ref={ref}
          className={cn(
            "z-50 min-w-[12rem] overflow-hidden rounded-md border bg-gray-900/95 backdrop-blur-sm border-gray-700 p-1 shadow-xl",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
            className
          )}
          sideOffset={5}
        >
          {items.map(item => renderMenuItem(item))}
        </ContextMenuPrimitive.Content>
      </ContextMenuPrimitive.Portal>
    </ContextMenuPrimitive.Root>
  );
});

TradingContextMenu.displayName = "TradingContextMenu";

export { TradingContextMenu, type ContextMenuItem };