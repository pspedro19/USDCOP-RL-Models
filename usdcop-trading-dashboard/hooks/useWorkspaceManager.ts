"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "react-hot-toast";

export interface WindowLayout {
  id: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  minimized: boolean;
  maximized: boolean;
  zIndex: number;
  component: string;
  props?: Record<string, any>;
}

export interface WorkspaceTemplate {
  id: string;
  name: string;
  description: string;
  windows: Omit<WindowLayout, "id">[];
  createdAt: Date;
  thumbnail?: string;
}

export interface MonitorInfo {
  id: string;
  isPrimary: boolean;
  width: number;
  height: number;
  availWidth: number;
  availHeight: number;
  colorDepth: number;
  orientation: string;
}

interface UseWorkspaceManagerProps {
  onLayoutChange?: (windows: WindowLayout[]) => void;
  onWorkspaceChange?: (templateId: string) => void;
  autoSave?: boolean;
  maxWindows?: number;
}

export const useWorkspaceManager = ({
  onLayoutChange,
  onWorkspaceChange,
  autoSave = true,
  maxWindows = 20,
}: UseWorkspaceManagerProps = {}) => {
  const [windows, setWindows] = useState<WindowLayout[]>([]);
  const [activeWindow, setActiveWindow] = useState<string | null>(null);
  const [workspaceTemplates, setWorkspaceTemplates] = useState<WorkspaceTemplate[]>([]);
  const [currentTemplate, setCurrentTemplate] = useState<string | null>(null);
  const [monitors, setMonitors] = useState<MonitorInfo[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const nextZIndex = useRef(1000);
  const dragState = useRef<{
    isDragging: boolean;
    windowId: string | null;
    startPosition: { x: number; y: number };
    windowStartPosition: { x: number; y: number };
  }>({
    isDragging: false,
    windowId: null,
    startPosition: { x: 0, y: 0 },
    windowStartPosition: { x: 0, y: 0 },
  });

  // Initialize monitors information
  useEffect(() => {
    const detectMonitors = () => {
      const screen = window.screen;
      const primaryMonitor: MonitorInfo = {
        id: "primary",
        isPrimary: true,
        width: screen.width,
        height: screen.height,
        availWidth: screen.availWidth,
        availHeight: screen.availHeight,
        colorDepth: screen.colorDepth,
        orientation: screen.orientation?.type || "landscape-primary",
      };

      setMonitors([primaryMonitor]);
    };

    detectMonitors();

    // Listen for screen changes
    if (screen.orientation) {
      screen.orientation.addEventListener("change", detectMonitors);
      return () => screen.orientation.removeEventListener("change", detectMonitors);
    }
  }, []);

  // Load saved workspace on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedWindows = localStorage.getItem("trading-workspace-windows");
      const savedTemplates = localStorage.getItem("trading-workspace-templates");

      if (savedWindows) {
        try {
          setWindows(JSON.parse(savedWindows));
        } catch (error) {
          console.error("Failed to load saved windows:", error);
        }
      }

      if (savedTemplates) {
        try {
          const templates = JSON.parse(savedTemplates);
          setWorkspaceTemplates(templates.map((t: any) => ({
            ...t,
            createdAt: new Date(t.createdAt),
          })));
        } catch (error) {
          console.error("Failed to load saved templates:", error);
        }
      }
    }
  }, []);

  // Auto-save workspace
  useEffect(() => {
    if (autoSave && windows.length > 0) {
      localStorage.setItem("trading-workspace-windows", JSON.stringify(windows));
    }
  }, [windows, autoSave]);

  // Save templates
  useEffect(() => {
    if (workspaceTemplates.length > 0) {
      localStorage.setItem("trading-workspace-templates", JSON.stringify(workspaceTemplates));
    }
  }, [workspaceTemplates]);

  // Create new window
  const createWindow = useCallback((
    component: string,
    props?: Record<string, any>,
    customLayout?: Partial<WindowLayout>
  ): string => {
    if (windows.length >= maxWindows) {
      toast.error(`Maximum of ${maxWindows} windows allowed`);
      return "";
    }

    const windowId = `window-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const primaryMonitor = monitors.find(m => m.isPrimary) || monitors[0];

    const defaultLayout: WindowLayout = {
      id: windowId,
      name: component,
      x: Math.random() * (primaryMonitor?.availWidth - 400) || 100,
      y: Math.random() * (primaryMonitor?.availHeight - 300) || 100,
      width: 400,
      height: 300,
      minimized: false,
      maximized: false,
      zIndex: nextZIndex.current++,
      component,
      props,
      ...customLayout,
    };

    setWindows(prev => [...prev, defaultLayout]);
    setActiveWindow(windowId);
    onLayoutChange?.([...windows, defaultLayout]);

    return windowId;
  }, [windows, maxWindows, monitors, onLayoutChange]);

  // Update window
  const updateWindow = useCallback((windowId: string, updates: Partial<WindowLayout>) => {
    setWindows(prev => prev.map(window =>
      window.id === windowId ? { ...window, ...updates } : window
    ));

    const updatedWindows = windows.map(window =>
      window.id === windowId ? { ...window, ...updates } : window
    );
    onLayoutChange?.(updatedWindows);
  }, [windows, onLayoutChange]);

  // Close window
  const closeWindow = useCallback((windowId: string) => {
    setWindows(prev => prev.filter(window => window.id !== windowId));
    if (activeWindow === windowId) {
      setActiveWindow(null);
    }

    const remainingWindows = windows.filter(window => window.id !== windowId);
    onLayoutChange?.(remainingWindows);

    toast.success("Window closed", { duration: 1500 });
  }, [windows, activeWindow, onLayoutChange]);

  // Minimize window
  const minimizeWindow = useCallback((windowId: string) => {
    updateWindow(windowId, { minimized: true });
  }, [updateWindow]);

  // Maximize window
  const maximizeWindow = useCallback((windowId: string) => {
    const window = windows.find(w => w.id === windowId);
    if (!window) return;

    const primaryMonitor = monitors.find(m => m.isPrimary) || monitors[0];

    if (window.maximized) {
      // Restore window
      updateWindow(windowId, {
        maximized: false,
        // Restore previous size/position if stored
        x: window.x || 100,
        y: window.y || 100,
        width: window.width || 400,
        height: window.height || 300,
      });
    } else {
      // Maximize window
      updateWindow(windowId, {
        maximized: true,
        x: 0,
        y: 0,
        width: primaryMonitor?.availWidth || window.innerWidth,
        height: primaryMonitor?.availHeight || window.innerHeight,
      });
    }
  }, [windows, monitors, updateWindow]);

  // Focus window (bring to front)
  const focusWindow = useCallback((windowId: string) => {
    setActiveWindow(windowId);
    updateWindow(windowId, {
      zIndex: nextZIndex.current++,
      minimized: false,
    });
  }, [updateWindow]);

  // Arrange windows in different layouts
  const arrangeWindows = useCallback((layout: "tile" | "cascade" | "stack") => {
    const visibleWindows = windows.filter(w => !w.minimized);
    const primaryMonitor = monitors.find(m => m.isPrimary) || monitors[0];

    if (!primaryMonitor || visibleWindows.length === 0) return;

    const availWidth = primaryMonitor.availWidth;
    const availHeight = primaryMonitor.availHeight;

    switch (layout) {
      case "tile": {
        const cols = Math.ceil(Math.sqrt(visibleWindows.length));
        const rows = Math.ceil(visibleWindows.length / cols);
        const windowWidth = availWidth / cols;
        const windowHeight = availHeight / rows;

        visibleWindows.forEach((window, index) => {
          const col = index % cols;
          const row = Math.floor(index / cols);

          updateWindow(window.id, {
            x: col * windowWidth,
            y: row * windowHeight,
            width: windowWidth,
            height: windowHeight,
            maximized: false,
          });
        });
        break;
      }

      case "cascade": {
        const offsetX = 30;
        const offsetY = 30;
        const baseWidth = Math.min(600, availWidth * 0.6);
        const baseHeight = Math.min(400, availHeight * 0.6);

        visibleWindows.forEach((window, index) => {
          updateWindow(window.id, {
            x: index * offsetX,
            y: index * offsetY,
            width: baseWidth,
            height: baseHeight,
            maximized: false,
          });
        });
        break;
      }

      case "stack": {
        visibleWindows.forEach((window, index) => {
          updateWindow(window.id, {
            x: availWidth / 2 - 300,
            y: availHeight / 2 - 200,
            width: 600,
            height: 400,
            maximized: false,
            zIndex: nextZIndex.current + index,
          });
        });
        nextZIndex.current += visibleWindows.length;
        break;
      }
    }

    toast.success(`Windows arranged in ${layout} layout`, { duration: 2000 });
  }, [windows, monitors, updateWindow]);

  // Save current layout as template
  const saveAsTemplate = useCallback((name: string, description?: string) => {
    if (!name.trim()) {
      toast.error("Template name is required");
      return;
    }

    const template: WorkspaceTemplate = {
      id: `template-${Date.now()}`,
      name: name.trim(),
      description: description || "",
      windows: windows.map(({ id, ...window }) => window),
      createdAt: new Date(),
    };

    setWorkspaceTemplates(prev => [...prev, template]);
    toast.success(`Template "${name}" saved`, { duration: 2000 });

    return template.id;
  }, [windows]);

  // Load workspace template
  const loadTemplate = useCallback((templateId: string) => {
    const template = workspaceTemplates.find(t => t.id === templateId);
    if (!template) {
      toast.error("Template not found");
      return;
    }

    // Clear current windows
    setWindows([]);

    // Create windows from template
    const newWindows: WindowLayout[] = template.windows.map((windowTemplate, index) => ({
      ...windowTemplate,
      id: `window-${Date.now()}-${index}`,
      zIndex: nextZIndex.current + index,
    }));

    nextZIndex.current += newWindows.length;

    setWindows(newWindows);
    setCurrentTemplate(templateId);
    onLayoutChange?.(newWindows);
    onWorkspaceChange?.(templateId);

    toast.success(`Template "${template.name}" loaded`, { duration: 2000 });
  }, [workspaceTemplates, onLayoutChange, onWorkspaceChange]);

  // Delete template
  const deleteTemplate = useCallback((templateId: string) => {
    setWorkspaceTemplates(prev => prev.filter(t => t.id !== templateId));
    if (currentTemplate === templateId) {
      setCurrentTemplate(null);
    }
    toast.success("Template deleted", { duration: 1500 });
  }, [currentTemplate]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      toast.error("Fullscreen not supported");
    }
  }, []);

  // Clear workspace
  const clearWorkspace = useCallback(() => {
    setWindows([]);
    setActiveWindow(null);
    setCurrentTemplate(null);
    onLayoutChange?.([]);
    toast.success("Workspace cleared", { duration: 1500 });
  }, [onLayoutChange]);

  return {
    // State
    windows,
    activeWindow,
    workspaceTemplates,
    currentTemplate,
    monitors,
    isFullscreen,

    // Window management
    createWindow,
    updateWindow,
    closeWindow,
    minimizeWindow,
    maximizeWindow,
    focusWindow,

    // Layout management
    arrangeWindows,
    clearWorkspace,

    // Template management
    saveAsTemplate,
    loadTemplate,
    deleteTemplate,

    // System
    toggleFullscreen,

    // Utilities
    getWindow: (id: string) => windows.find(w => w.id === id),
    getVisibleWindows: () => windows.filter(w => !w.minimized),
    getWindowCount: () => windows.length,
    canCreateWindow: () => windows.length < maxWindows,
  };
};