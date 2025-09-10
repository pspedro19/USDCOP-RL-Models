'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

export interface SidebarState {
  leftSidebarCollapsed: boolean;
  navigationSidebarOpen: boolean;
  isDesktop: boolean;
  isMobile: boolean;
  viewportWidth: number;
  autoHideEnabled: boolean;
}

export interface SidebarActions {
  toggleLeftSidebar: () => void;
  toggleNavigationSidebar: () => void;
  toggleBothSidebars: () => void;
  collapseLeftSidebar: () => void;
  expandLeftSidebar: () => void;
  openNavigationSidebar: () => void;
  closeNavigationSidebar: () => void;
  setAutoHide: (enabled: boolean) => void;
  resetToDefaults: () => void;
  forceSync: () => void;
}

export interface SidebarCalculations {
  leftSidebarWidth: number;
  navigationSidebarWidth: number;
  totalSidebarWidth: number;
  mainContentMarginLeft: string;
  isLeftSidebarVisible: boolean;
  isNavigationSidebarVisible: boolean;
}

// Constants for sidebar widths
const SIDEBAR_WIDTHS = {
  LEFT_EXPANDED: 320,     // 20rem = 320px
  LEFT_COLLAPSED: 64,     // 4rem = 64px  
  NAVIGATION: 320,        // 20rem = 320px
  NAVIGATION_MOBILE: 384, // 24rem = 384px (wider on mobile)
} as const;

// Breakpoints (matching Tailwind)
const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
} as const;

// Local storage keys
const STORAGE_KEYS = {
  LEFT_SIDEBAR_COLLAPSED: 'usdcop-left-sidebar-collapsed',
  NAVIGATION_SIDEBAR_OPEN: 'usdcop-navigation-sidebar-open',
  AUTO_HIDE_ENABLED: 'usdcop-auto-hide-enabled',
} as const;

export function useSidebarState(): SidebarState & SidebarActions & SidebarCalculations {
  // Core state
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(false);
  const [navigationSidebarOpen, setNavigationSidebarOpen] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [autoHideEnabled, setAutoHideEnabled] = useState(true);
  
  // Refs for managing state persistence
  const initialLoadRef = useRef(true);
  const resizeTimeoutRef = useRef<NodeJS.Timeout>();

  // Derived state
  const isMobile = viewportWidth < BREAKPOINTS.LG;
  const isDesktop = viewportWidth >= BREAKPOINTS.XL;

  // Load persisted state from localStorage
  const loadPersistedState = useCallback(() => {
    if (typeof window === 'undefined') return;

    try {
      const savedLeftCollapsed = localStorage.getItem(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED);
      const savedNavigationOpen = localStorage.getItem(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN);
      const savedAutoHide = localStorage.getItem(STORAGE_KEYS.AUTO_HIDE_ENABLED);

      if (savedLeftCollapsed !== null) {
        setLeftSidebarCollapsed(JSON.parse(savedLeftCollapsed));
      }
      
      // On desktop, restore navigation state; on mobile, always start closed
      if (savedNavigationOpen !== null && !isMobile) {
        setNavigationSidebarOpen(JSON.parse(savedNavigationOpen));
      }
      
      if (savedAutoHide !== null) {
        setAutoHideEnabled(JSON.parse(savedAutoHide));
      }
    } catch (error) {
      console.warn('Failed to load sidebar state from localStorage:', error);
    }
  }, [isMobile]);

  // Save state to localStorage
  const saveState = useCallback((key: string, value: boolean) => {
    if (typeof window === 'undefined') return;
    
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to save sidebar state to localStorage:', error);
    }
  }, []);

  // Viewport size detection with debouncing
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateViewportWidth = () => {
      setViewportWidth(window.innerWidth);
    };

    const debouncedResize = () => {
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
      
      resizeTimeoutRef.current = setTimeout(updateViewportWidth, 150);
    };

    // Set initial width
    updateViewportWidth();
    
    window.addEventListener('resize', debouncedResize);
    window.addEventListener('orientationchange', debouncedResize);

    return () => {
      window.removeEventListener('resize', debouncedResize);
      window.removeEventListener('orientationchange', debouncedResize);
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
    };
  }, []);

  // Auto-hide logic based on viewport changes
  useEffect(() => {
    if (!autoHideEnabled || initialLoadRef.current) return;

    // Auto-hide logic for different viewport sizes
    if (isMobile) {
      // On mobile: always collapse left sidebar, close navigation
      setLeftSidebarCollapsed(true);
      setNavigationSidebarOpen(false);
    } else if (viewportWidth >= BREAKPOINTS.MD && viewportWidth < BREAKPOINTS.XL) {
      // On tablet: keep left sidebar visible but collapsed
      setLeftSidebarCollapsed(true);
      setNavigationSidebarOpen(false);
    }
    // Desktop: preserve user preferences
    
  }, [isMobile, viewportWidth, autoHideEnabled]);

  // Load persisted state on mount and viewport changes
  useEffect(() => {
    if (initialLoadRef.current) {
      loadPersistedState();
      initialLoadRef.current = false;
    }
  }, [loadPersistedState]);

  // Actions
  const toggleLeftSidebar = useCallback(() => {
    const newState = !leftSidebarCollapsed;
    setLeftSidebarCollapsed(newState);
    saveState(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED, newState);
  }, [leftSidebarCollapsed, saveState]);

  const toggleNavigationSidebar = useCallback(() => {
    const newState = !navigationSidebarOpen;
    setNavigationSidebarOpen(newState);
    
    // Only save navigation state on desktop
    if (!isMobile) {
      saveState(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN, newState);
    }
  }, [navigationSidebarOpen, isMobile, saveState]);

  const toggleBothSidebars = useCallback(() => {
    const shouldExpand = leftSidebarCollapsed || !navigationSidebarOpen;
    
    if (shouldExpand) {
      setLeftSidebarCollapsed(false);
      setNavigationSidebarOpen(true);
      saveState(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED, false);
      if (!isMobile) {
        saveState(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN, true);
      }
    } else {
      setLeftSidebarCollapsed(true);
      setNavigationSidebarOpen(false);
      saveState(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED, true);
      if (!isMobile) {
        saveState(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN, false);
      }
    }
  }, [leftSidebarCollapsed, navigationSidebarOpen, isMobile, saveState]);

  const collapseLeftSidebar = useCallback(() => {
    if (!leftSidebarCollapsed) {
      setLeftSidebarCollapsed(true);
      saveState(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED, true);
    }
  }, [leftSidebarCollapsed, saveState]);

  const expandLeftSidebar = useCallback(() => {
    if (leftSidebarCollapsed) {
      setLeftSidebarCollapsed(false);
      saveState(STORAGE_KEYS.LEFT_SIDEBAR_COLLAPSED, false);
    }
  }, [leftSidebarCollapsed, saveState]);

  const openNavigationSidebar = useCallback(() => {
    if (!navigationSidebarOpen) {
      setNavigationSidebarOpen(true);
      if (!isMobile) {
        saveState(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN, true);
      }
    }
  }, [navigationSidebarOpen, isMobile, saveState]);

  const closeNavigationSidebar = useCallback(() => {
    if (navigationSidebarOpen) {
      setNavigationSidebarOpen(false);
      if (!isMobile) {
        saveState(STORAGE_KEYS.NAVIGATION_SIDEBAR_OPEN, false);
      }
    }
  }, [navigationSidebarOpen, isMobile, saveState]);

  const setAutoHideEnabledAction = useCallback((enabled: boolean) => {
    setAutoHideEnabled(enabled);
    saveState(STORAGE_KEYS.AUTO_HIDE_ENABLED, enabled);
  }, [saveState]);

  const resetToDefaults = useCallback(() => {
    setLeftSidebarCollapsed(false);
    setNavigationSidebarOpen(!isMobile);
    setAutoHideEnabled(true);
    
    // Clear localStorage
    Object.values(STORAGE_KEYS).forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (error) {
        console.warn(`Failed to remove ${key} from localStorage:`, error);
      }
    });
  }, [isMobile]);

  const forceSync = useCallback(() => {
    // Force re-evaluation of all derived state
    if (typeof window !== 'undefined') {
      setViewportWidth(window.innerWidth);
    }
    loadPersistedState();
  }, [loadPersistedState]);

  // Calculations for layout
  const isLeftSidebarVisible = !isMobile; // Left sidebar only visible on desktop/tablet
  const isNavigationSidebarVisible = isDesktop || navigationSidebarOpen;

  const leftSidebarWidth = isLeftSidebarVisible
    ? (leftSidebarCollapsed ? SIDEBAR_WIDTHS.LEFT_COLLAPSED : SIDEBAR_WIDTHS.LEFT_EXPANDED)
    : 0;

  const navigationSidebarWidth = isDesktop 
    ? SIDEBAR_WIDTHS.NAVIGATION
    : (navigationSidebarOpen ? SIDEBAR_WIDTHS.NAVIGATION_MOBILE : 0);

  const totalSidebarWidth = leftSidebarWidth + (isDesktop ? navigationSidebarWidth : 0);

  // Calculate main content margin based on visible sidebars and screen size
  const calculateMainContentMargin = (): string => {
    if (isMobile) {
      // Mobile: no margins, sidebars are overlays
      return 'ml-0';
    }
    
    if (viewportWidth >= BREAKPOINTS.XL) {
      // Desktop (xl): Both sidebars can be visible
      const leftWidth = leftSidebarCollapsed ? 64 : 320;
      const navWidth = 320;
      const totalWidth = leftWidth + navWidth;
      
      return `xl:ml-[${totalWidth}px]`;
    } else if (viewportWidth >= BREAKPOINTS.LG) {
      // Large (lg): Only left sidebar visible
      const leftWidth = leftSidebarCollapsed ? 64 : 320;
      
      return `lg:ml-[${leftWidth}px] xl:ml-[${leftWidth + 320}px]`;
    }
    
    return 'ml-0';
  };

  const mainContentMarginLeft = calculateMainContentMargin();

  return {
    // State
    leftSidebarCollapsed,
    navigationSidebarOpen,
    isDesktop,
    isMobile,
    viewportWidth,
    autoHideEnabled,
    
    // Actions
    toggleLeftSidebar,
    toggleNavigationSidebar,
    toggleBothSidebars,
    collapseLeftSidebar,
    expandLeftSidebar,
    openNavigationSidebar,
    closeNavigationSidebar,
    setAutoHide: setAutoHideEnabledAction,
    resetToDefaults,
    forceSync,
    
    // Calculations
    leftSidebarWidth,
    navigationSidebarWidth,
    totalSidebarWidth,
    mainContentMarginLeft,
    isLeftSidebarVisible,
    isNavigationSidebarVisible,
  };
}