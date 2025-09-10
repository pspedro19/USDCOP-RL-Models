'use client';

import { useMemo } from 'react';
import type { SidebarState } from './useSidebarState';

export interface ResponsiveLayoutConfig {
  mainContentWidth: string;
  mainContentMaxWidth: string;
  mainContentMarginLeft: string;
  mainContentPaddingLeft: string;
  mainContentPaddingRight: string;
  containerClasses: string;
  isFullWidth: boolean;
  availableWidth: number;
  sidebarOverlayLevel: number;
}

interface UseResponsiveLayoutProps extends SidebarState {
  leftSidebarWidth: number;
  navigationSidebarWidth: number;
  totalSidebarWidth: number;
  isLeftSidebarVisible: boolean;
  isNavigationSidebarVisible: boolean;
}

// Tailwind breakpoint values for calculations
const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

export function useResponsiveLayout({
  leftSidebarCollapsed,
  navigationSidebarOpen,
  isDesktop,
  isMobile,
  viewportWidth,
  leftSidebarWidth,
  navigationSidebarWidth,
  totalSidebarWidth,
  isLeftSidebarVisible,
  isNavigationSidebarVisible,
}: UseResponsiveLayoutProps): ResponsiveLayoutConfig {

  return useMemo(() => {
    // Calculate available width for main content
    const calculateAvailableWidth = (): number => {
      if (isMobile) {
        // On mobile, sidebars are overlays, so full width is available
        return viewportWidth;
      }
      
      if (viewportWidth >= BREAKPOINTS.XL) {
        // Desktop: subtract both sidebars if visible
        const leftWidth = isLeftSidebarVisible ? leftSidebarWidth : 0;
        const navWidth = isDesktop ? navigationSidebarWidth : 0;
        return viewportWidth - leftWidth - navWidth;
      } else if (viewportWidth >= BREAKPOINTS.LG) {
        // Large tablet: only left sidebar affects width
        const leftWidth = isLeftSidebarVisible ? leftSidebarWidth : 0;
        return viewportWidth - leftWidth;
      }
      
      // Smaller screens: full width available
      return viewportWidth;
    };

    // Calculate margin classes for different screen sizes
    const calculateMarginClasses = (): string => {
      if (isMobile) {
        return 'ml-0';
      }

      const margins: string[] = [];
      
      // Large screens (lg): Only left sidebar
      if (viewportWidth >= BREAKPOINTS.LG) {
        const leftWidth = leftSidebarCollapsed ? 64 : 320;
        margins.push(`lg:ml-[${leftWidth}px]`);
      }
      
      // Extra large screens (xl): Both sidebars
      if (viewportWidth >= BREAKPOINTS.XL) {
        const leftWidth = leftSidebarCollapsed ? 64 : 320;
        const navWidth = 320;
        const totalWidth = leftWidth + navWidth;
        margins.push(`xl:ml-[${totalWidth}px]`);
      }

      return margins.join(' ') || 'ml-0';
    };

    // Calculate dynamic container classes
    const calculateContainerClasses = (): string => {
      const classes = ['transition-all', 'duration-300', 'ease-in-out'];
      
      // Add responsive margins
      classes.push(calculateMarginClasses());
      
      // Add min-width constraints
      if (!isMobile && isLeftSidebarVisible) {
        classes.push('min-w-0'); // Prevent overflow
      }
      
      // Add max-width for ultra-wide screens
      if (viewportWidth >= BREAKPOINTS['2XL'] && !leftSidebarCollapsed && navigationSidebarOpen) {
        classes.push('max-w-none', '2xl:max-w-screen-xl', '2xl:mx-auto');
      }

      return classes.join(' ');
    };

    // Calculate padding adjustments
    const calculatePadding = (): { left: string; right: string } => {
      // Base padding
      let left = 'pl-3 sm:pl-4 lg:pl-6';
      let right = 'pr-3 sm:pr-4 lg:pr-6';
      
      // Adjust for mobile controls bar
      if (isMobile) {
        return { left, right: `${right} pb-32` };
      }
      
      // Adjust for very wide screens
      if (viewportWidth >= BREAKPOINTS['2XL'] && !leftSidebarCollapsed) {
        left = `${left} 2xl:pl-8`;
        right = `${right} 2xl:pr-8`;
      }
      
      return { left, right };
    };

    // Calculate overlay z-index levels
    const calculateOverlayLevel = (): number => {
      if (isMobile && navigationSidebarOpen) {
        return 40; // High z-index for mobile navigation overlay
      }
      
      if (viewportWidth < BREAKPOINTS.LG && isLeftSidebarVisible) {
        return 35; // Medium z-index for tablet sidebar overlay
      }
      
      return 10; // Normal z-index for integrated sidebars
    };

    const availableWidth = calculateAvailableWidth();
    const isFullWidth = isMobile || availableWidth >= viewportWidth * 0.9;
    const containerClasses = calculateContainerClasses();
    const padding = calculatePadding();
    const sidebarOverlayLevel = calculateOverlayLevel();

    // Calculate optimal max-width
    const calculateMaxWidth = (): string => {
      if (isFullWidth || isMobile) {
        return 'max-w-none';
      }
      
      // For very wide screens, limit content width for readability
      if (availableWidth > 1400) {
        return 'max-w-screen-xl';
      } else if (availableWidth > 1200) {
        return 'max-w-screen-lg';
      }
      
      return 'max-w-none';
    };

    return {
      mainContentWidth: isFullWidth ? 'w-full' : `w-[${availableWidth}px]`,
      mainContentMaxWidth: calculateMaxWidth(),
      mainContentMarginLeft: calculateMarginClasses(),
      mainContentPaddingLeft: padding.left,
      mainContentPaddingRight: padding.right,
      containerClasses,
      isFullWidth,
      availableWidth,
      sidebarOverlayLevel,
    };
  }, [
    leftSidebarCollapsed,
    navigationSidebarOpen,
    isDesktop,
    isMobile,
    viewportWidth,
    leftSidebarWidth,
    navigationSidebarWidth,
    totalSidebarWidth,
    isLeftSidebarVisible,
    isNavigationSidebarVisible,
  ]);
}

// Additional utility hook for responsive breakpoint detection
export function useBreakpoints(viewportWidth: number) {
  return useMemo(() => ({
    isSm: viewportWidth >= BREAKPOINTS.SM,
    isMd: viewportWidth >= BREAKPOINTS.MD,
    isLg: viewportWidth >= BREAKPOINTS.LG,
    isXl: viewportWidth >= BREAKPOINTS.XL,
    is2Xl: viewportWidth >= BREAKPOINTS['2XL'],
    breakpoint: 
      viewportWidth >= BREAKPOINTS['2XL'] ? '2xl' :
      viewportWidth >= BREAKPOINTS.XL ? 'xl' :
      viewportWidth >= BREAKPOINTS.LG ? 'lg' :
      viewportWidth >= BREAKPOINTS.MD ? 'md' :
      viewportWidth >= BREAKPOINTS.SM ? 'sm' : 'xs',
  }), [viewportWidth]);
}