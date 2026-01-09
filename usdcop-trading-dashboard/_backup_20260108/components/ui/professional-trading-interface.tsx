"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  Command,
  HelpCircle,
  Settings,
  Maximize2,
  Minimize2,
  Monitor,
  Accessibility,
  Keyboard
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "react-hot-toast";

// Import our components
import { CommandPalette, type CommandItem } from "./command-palette";
import { SmartCrosshair, type CrosshairPosition, type PriceData } from "./smart-crosshair";
import { TradingContextMenu, type ContextMenuItem } from "./trading-context-menu";
import { HelpSystem } from "./help-system";

// Import our hooks
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useTouchGestures } from "@/hooks/useTouchGestures";
import { useWorkspaceManager } from "@/hooks/useWorkspaceManager";
import { useAccessibility } from "@/hooks/useAccessibility";

interface ProfessionalTradingInterfaceProps {
  children: React.ReactNode;
  chartData?: PriceData[];
  onTimeframeChange?: (timeframe: string) => void;
  onDrawingTool?: (tool: string) => void;
  onTradingAction?: (action: string) => void;
  onChartAction?: (action: string) => void;
  className?: string;
}

const ProfessionalTradingInterface = React.forwardRef<
  HTMLDivElement,
  ProfessionalTradingInterfaceProps
>(({
  children,
  chartData = [],
  onTimeframeChange,
  onDrawingTool,
  onTradingAction,
  onChartAction,
  className,
}, ref) => {
  // State management
  const [commandPaletteOpen, setCommandPaletteOpen] = React.useState(false);
  const [helpSystemOpen, setHelpSystemOpen] = React.useState(false);
  const [crosshairPosition, setCrosshairPosition] = React.useState<CrosshairPosition>({
    x: 0,
    y: 0,
    price: 0,
    time: 0,
    visible: false,
  });
  const [mousePosition, setMousePosition] = React.useState({ x: 0, y: 0 });
  const [isChartFocused, setIsChartFocused] = React.useState(false);

  // Refs
  const containerRef = React.useRef<HTMLDivElement>(null);
  const chartContainerRef = React.useRef<HTMLDivElement>(null);

  // Initialize hooks
  const accessibility = useAccessibility({
    onSettingsChange: (settings) => {
      console.log("Accessibility settings changed:", settings);
    },
  });

  const workspaceManager = useWorkspaceManager({
    onLayoutChange: (windows) => {
      console.log("Workspace layout changed:", windows);
    },
    onWorkspaceChange: (templateId) => {
      console.log("Workspace template changed:", templateId);
      toast.success("Workspace loaded");
    },
  });

  const touchGestures = useTouchGestures({
    onZoom: (scale, center) => {
      onChartAction?.(`zoom:${scale}:${center.x},${center.y}`);
    },
    onPan: (delta) => {
      onChartAction?.(`pan:${delta.x},${delta.y}`);
    },
    onLongPress: (position) => {
      console.log("Long press at:", position);
    },
    onDoubleClick: (position) => {
      onChartAction?.("resetZoom");
    },
    onThreeFingerTap: (position) => {
      setHelpSystemOpen(true);
    },
    containerRef: chartContainerRef,
  });

  const keyboardShortcuts = useKeyboardShortcuts({
    onTimeframeChange,
    onDrawingTool,
    onTradingAction,
    onChartAction,
    onSystemAction: (action) => {
      switch (action) {
        case "commandPalette":
          setCommandPaletteOpen(true);
          break;
        case "help":
          setHelpSystemOpen(true);
          break;
        case "fullscreen":
          workspaceManager.toggleFullscreen();
          break;
        case "preferences":
          console.log("Open preferences");
          break;
        default:
          console.log("System action:", action);
      }
    },
    enabled: isChartFocused,
  });

  // Custom command items for the command palette
  const customCommands: CommandItem[] = React.useMemo(() => [
    // Workspace commands
    {
      id: "workspace.save",
      title: "Save Workspace",
      subtitle: "Save current layout as template",
      category: "Workspace",
      icon: Monitor,
      action: () => {
        const name = prompt("Workspace name:");
        if (name) {
          workspaceManager.saveAsTemplate(name);
        }
      },
      keywords: ["save", "workspace", "template", "layout"],
    },
    {
      id: "workspace.tile",
      title: "Tile Windows",
      subtitle: "Arrange windows in a grid",
      category: "Workspace",
      icon: Maximize2,
      action: () => workspaceManager.arrangeWindows("tile"),
      keywords: ["tile", "arrange", "grid", "windows"],
    },
    {
      id: "workspace.cascade",
      title: "Cascade Windows",
      subtitle: "Arrange windows in a cascade",
      category: "Workspace",
      icon: Minimize2,
      action: () => workspaceManager.arrangeWindows("cascade"),
      keywords: ["cascade", "arrange", "windows"],
    },
    // Accessibility commands
    {
      id: "accessibility.highContrast",
      title: "Toggle High Contrast",
      subtitle: "Enable/disable high contrast mode",
      category: "Accessibility",
      icon: Accessibility,
      action: () => accessibility.updateSetting("highContrastMode", !accessibility.settings.highContrastMode),
      keywords: ["high", "contrast", "accessibility", "vision"],
    },
    {
      id: "accessibility.fontSize",
      title: "Increase Font Size",
      subtitle: "Make text larger for better readability",
      category: "Accessibility",
      icon: Accessibility,
      action: () => {
        const sizes = ["small", "medium", "large", "extra-large"] as const;
        const currentIndex = sizes.indexOf(accessibility.settings.fontSize);
        const nextIndex = (currentIndex + 1) % sizes.length;
        accessibility.updateSetting("fontSize", sizes[nextIndex]);
      },
      keywords: ["font", "size", "text", "larger", "accessibility"],
    },
    // Help commands
    {
      id: "help.shortcuts",
      title: "Keyboard Shortcuts",
      subtitle: "View all available shortcuts",
      category: "Help",
      icon: Keyboard,
      action: () => {
        setHelpSystemOpen(true);
        // Set to shortcuts tab after a brief delay
        setTimeout(() => {
          // This would need to be passed to HelpSystem component
        }, 100);
      },
      keywords: ["shortcuts", "keyboard", "help", "hotkeys"],
    },
  ], [workspaceManager, accessibility, setHelpSystemOpen]);

  // Custom context menu items
  const customContextMenuItems: ContextMenuItem[] = React.useMemo(() => [
    {
      id: "accessibility",
      label: "Accessibility",
      icon: Accessibility,
      action: () => console.log("Accessibility menu"),
      submenu: [
        {
          id: "highContrast",
          label: "High Contrast Mode",
          action: () => accessibility.updateSetting("highContrastMode", !accessibility.settings.highContrastMode),
        },
        {
          id: "screenReader",
          label: "Screen Reader Mode",
          action: () => accessibility.updateSetting("screenReaderMode", !accessibility.settings.screenReaderMode),
        },
        {
          id: "reducedMotion",
          label: "Reduced Motion",
          action: () => accessibility.updateSetting("reducedMotion", !accessibility.settings.reducedMotion),
        },
      ],
    },
    {
      id: "workspace",
      label: "Workspace",
      icon: Monitor,
      action: () => console.log("Workspace menu"),
      submenu: [
        {
          id: "saveWorkspace",
          label: "Save Workspace",
          action: () => {
            const name = prompt("Workspace name:");
            if (name) {
              workspaceManager.saveAsTemplate(name);
            }
          },
        },
        {
          id: "tileWindows",
          label: "Tile Windows",
          action: () => workspaceManager.arrangeWindows("tile"),
        },
        {
          id: "cascadeWindows",
          label: "Cascade Windows",
          action: () => workspaceManager.arrangeWindows("cascade"),
        },
        {
          id: "separator",
          label: "",
          separator: true,
          action: () => {},
        },
        {
          id: "fullscreen",
          label: "Toggle Fullscreen",
          action: () => workspaceManager.toggleFullscreen(),
        },
      ],
    },
  ], [accessibility, workspaceManager]);

  // Mouse movement tracking for crosshair
  const handleMouseMove = React.useCallback((event: React.MouseEvent) => {
    if (!chartContainerRef.current) return;

    const rect = chartContainerRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    setMousePosition({ x, y });

    // Simulate price calculation (this would come from your actual chart library)
    const mockPrice = 4000 + ((rect.height - y) / rect.height) * 500; // Mock USD/COP price
    const mockTime = Date.now(); // Mock timestamp

    setCrosshairPosition({
      x,
      y,
      price: mockPrice,
      time: mockTime,
      visible: true,
    });
  }, []);

  const handleMouseLeave = React.useCallback(() => {
    setCrosshairPosition(prev => ({ ...prev, visible: false }));
  }, []);

  // Focus management
  const handleFocus = React.useCallback(() => {
    setIsChartFocused(true);
  }, []);

  const handleBlur = React.useCallback(() => {
    setIsChartFocused(false);
  }, []);

  // Handle command execution
  const handleCommandExecute = React.useCallback((command: CommandItem) => {
    accessibility.queueAnnouncement(`Executed: ${command.title}`);
  }, [accessibility]);

  // Apply touch gestures
  const gestureBindings = touchGestures.gestures();

  return (
    <div
      ref={ref}
      className={cn(
        "relative w-full h-full bg-gray-900 text-white",
        "focus-within:outline-none",
        accessibility.settings.highContrastMode && "high-contrast",
        accessibility.settings.reducedMotion && "reduced-motion",
        className
      )}
      style={{
        fontSize: `var(--accessibility-font-size, 16px)`,
      }}
      onFocus={handleFocus}
      onBlur={handleBlur}
      tabIndex={0}
      role="application"
      aria-label="Professional Trading Interface"
    >
      {/* Skip Link for accessibility */}
      {accessibility.createSkipLink?.()}

      {/* Main interface container */}
      <div ref={containerRef} className="relative w-full h-full">
        {/* Chart container with touch gestures and context menu */}
        <TradingContextMenu
          items={customContextMenuItems}
          onItemSelect={(item) => {
            accessibility.queueAnnouncement(`Selected: ${item.label}`);
          }}
        >
          <div
            ref={chartContainerRef}
            {...gestureBindings}
            className="relative w-full h-full cursor-crosshair select-none"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            role="img"
            aria-label="Trading Chart"
            tabIndex={-1}
          >
            {/* Chart content */}
            <div className="w-full h-full">
              {children}
            </div>

            {/* Smart Crosshair */}
            <SmartCrosshair
              position={crosshairPosition}
              data={chartData}
              snapToPrice={true}
              showTooltip={true}
              showDistanceMeasurement={true}
              onPriceCopy={(price) => {
                accessibility.queueAnnouncement(`Price copied: ${price.toFixed(4)}`);
              }}
            />
          </div>
        </TradingContextMenu>

        {/* Floating action buttons for accessibility */}
        <div className="absolute top-4 right-4 flex flex-col space-y-2 z-40">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setCommandPaletteOpen(true)}
            className="p-3 bg-blue-500/20 hover:bg-blue-500/30 backdrop-blur-sm border border-blue-500/30 rounded-lg transition-colors"
            aria-label="Open Command Palette (Cmd+K)"
            title="Command Palette (Cmd+K)"
          >
            <Command className="h-5 w-5 text-blue-400" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setHelpSystemOpen(true)}
            className="p-3 bg-purple-500/20 hover:bg-purple-500/30 backdrop-blur-sm border border-purple-500/30 rounded-lg transition-colors"
            aria-label="Open Help System (Cmd+?)"
            title="Help & Tutorials (Cmd+?)"
          >
            <HelpCircle className="h-5 w-5 text-purple-400" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => accessibility.updateSetting("highContrastMode", !accessibility.settings.highContrastMode)}
            className={cn(
              "p-3 backdrop-blur-sm border rounded-lg transition-colors",
              accessibility.settings.highContrastMode
                ? "bg-yellow-500/30 border-yellow-500/50 text-yellow-400"
                : "bg-gray-500/20 hover:bg-gray-500/30 border-gray-500/30 text-gray-400"
            )}
            aria-label="Toggle High Contrast Mode"
            title="Toggle High Contrast"
          >
            <Accessibility className="h-5 w-5" />
          </motion.button>
        </div>

        {/* Workspace info overlay */}
        {workspaceManager.windows.length > 0 && (
          <div className="absolute bottom-4 left-4 z-40">
            <div className="px-3 py-2 bg-gray-800/80 backdrop-blur-sm border border-gray-600 rounded-lg text-sm text-gray-300">
              <div className="flex items-center space-x-2">
                <Monitor className="h-4 w-4" />
                <span>{workspaceManager.windows.length} windows open</span>
                {workspaceManager.currentTemplate && (
                  <span className="text-blue-400">• {workspaceManager.currentTemplate}</span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Status indicators */}
        <div className="absolute top-4 left-4 z-40">
          <div className="flex items-center space-x-2">
            {accessibility.settings.screenReaderMode && (
              <div className="px-2 py-1 bg-green-500/20 border border-green-500/30 rounded text-xs text-green-300">
                Screen Reader
              </div>
            )}
            {accessibility.settings.highContrastMode && (
              <div className="px-2 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded text-xs text-yellow-300">
                High Contrast
              </div>
            )}
            {accessibility.settings.reducedMotion && (
              <div className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 rounded text-xs text-blue-300">
                Reduced Motion
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Command Palette */}
      <CommandPalette
        commands={customCommands}
        onCommandExecute={handleCommandExecute}
      />

      {/* Help System */}
      <HelpSystem
        open={helpSystemOpen}
        onOpenChange={setHelpSystemOpen}
      />

      {/* Accessibility announcements region */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {/* Screen reader announcements will be inserted here */}
      </div>

      {/* Hidden heading for screen readers */}
      <h1 className="sr-only">Professional Trading Interface</h1>
    </div>
  );
});

ProfessionalTradingInterface.displayName = "ProfessionalTradingInterface";

export { ProfessionalTradingInterface };