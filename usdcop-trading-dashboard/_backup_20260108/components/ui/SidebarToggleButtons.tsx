'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, 
  X, 
  ChevronLeft, 
  ChevronRight, 
  Maximize2, 
  Minimize2,
  Settings,
  Eye,
  EyeOff,
  Monitor,
  Smartphone
} from 'lucide-react';
import type { SidebarState, SidebarActions } from '@/hooks/useSidebarState';

interface SidebarToggleButtonsProps extends SidebarState, SidebarActions {
  className?: string;
  showLabels?: boolean;
  variant?: 'floating' | 'integrated' | 'minimal';
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export function SidebarToggleButtons({
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
  setAutoHide,
  resetToDefaults,
  
  // Props
  className = '',
  showLabels = false,
  variant = 'floating',
  position = 'top-left'
}: SidebarToggleButtonsProps) {

  const getPositionClasses = () => {
    switch (position) {
      case 'top-left':
        return 'top-4 left-4';
      case 'top-right':
        return 'top-4 right-4';
      case 'bottom-left':
        return 'bottom-4 left-4';
      case 'bottom-right':
        return 'bottom-4 right-4';
      default:
        return 'top-4 left-4';
    }
  };

  const getVariantClasses = () => {
    switch (variant) {
      case 'floating':
        return 'fixed z-50 glass-surface-primary backdrop-blur-md border border-slate-700/50 rounded-2xl shadow-glass-lg p-2';
      case 'integrated':
        return 'glass-surface-secondary rounded-xl border border-slate-700/30 p-2';
      case 'minimal':
        return 'flex gap-2';
      default:
        return 'fixed z-50 glass-surface-primary backdrop-blur-md border border-slate-700/50 rounded-2xl shadow-glass-lg p-2';
    }
  };

  const buttonBaseClasses = 'glass-button-primary backdrop-blur-sm border-cyan-500/30 hover:border-cyan-400/60 rounded-xl text-cyan-400 hover:text-cyan-100 shadow-glass-sm hover:shadow-glass-md hover:shadow-cyan-500/20 transition-all duration-300 group touch-manipulation flex items-center justify-center min-h-[40px] min-w-[40px]';

  // Navigation toggle button (mobile/desktop aware)
  const NavigationToggleButton = () => (
    <motion.button
      onClick={toggleNavigationSidebar}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={buttonBaseClasses}
      title={navigationSidebarOpen ? "Close navigation menu" : "Open navigation menu"}
      aria-label={navigationSidebarOpen ? "Close navigation menu" : "Open navigation menu"}
      aria-expanded={navigationSidebarOpen}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={navigationSidebarOpen ? 'close' : 'menu'}
          initial={{ opacity: 0, rotate: -90 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: 90 }}
          transition={{ duration: 0.2 }}
        >
          {navigationSidebarOpen ? 
            <X size={20} className="drop-shadow-sm" /> : 
            <Menu size={20} className="drop-shadow-sm" />
          }
        </motion.div>
      </AnimatePresence>
      {showLabels && (
        <span className="ml-2 text-sm font-medium">
          {navigationSidebarOpen ? 'Close' : 'Menu'}
        </span>
      )}
    </motion.button>
  );

  // Left sidebar toggle button (desktop only)
  const LeftSidebarToggleButton = () => (
    <motion.button
      onClick={toggleLeftSidebar}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={buttonBaseClasses}
      title={leftSidebarCollapsed ? "Expand controls panel" : "Collapse controls panel"}
      aria-label={leftSidebarCollapsed ? "Expand controls panel" : "Collapse controls panel"}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={leftSidebarCollapsed ? 'collapsed' : 'expanded'}
          initial={{ opacity: 0, rotate: leftSidebarCollapsed ? -90 : 90 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: leftSidebarCollapsed ? 90 : -90 }}
          transition={{ duration: 0.2 }}
        >
          {leftSidebarCollapsed ? 
            <ChevronRight size={20} className="drop-shadow-sm" /> : 
            <ChevronLeft size={20} className="drop-shadow-sm" />
          }
        </motion.div>
      </AnimatePresence>
      {showLabels && (
        <span className="ml-2 text-sm font-medium">
          {leftSidebarCollapsed ? 'Expand' : 'Collapse'}
        </span>
      )}
    </motion.button>
  );

  // Both sidebars toggle button (smart toggle)
  const BothSidebarsToggleButton = () => {
    const shouldExpand = leftSidebarCollapsed || !navigationSidebarOpen;
    
    return (
      <motion.button
        onClick={toggleBothSidebars}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className={`${buttonBaseClasses} ${shouldExpand ? '' : 'bg-cyan-500/10'}`}
        title={shouldExpand ? "Expand all panels" : "Collapse all panels"}
        aria-label={shouldExpand ? "Expand all panels" : "Collapse all panels"}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={shouldExpand ? 'expand' : 'collapse'}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
          >
            {shouldExpand ? 
              <Maximize2 size={20} className="drop-shadow-sm" /> : 
              <Minimize2 size={20} className="drop-shadow-sm" />
            }
          </motion.div>
        </AnimatePresence>
        {showLabels && (
          <span className="ml-2 text-sm font-medium">
            {shouldExpand ? 'Expand All' : 'Collapse All'}
          </span>
        )}
      </motion.button>
    );
  };

  // Auto-hide toggle button
  const AutoHideToggleButton = () => (
    <motion.button
      onClick={() => setAutoHide(!autoHideEnabled)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`${buttonBaseClasses} ${autoHideEnabled ? 'bg-purple-500/10' : ''}`}
      title={autoHideEnabled ? "Disable auto-hide" : "Enable auto-hide"}
      aria-label={autoHideEnabled ? "Disable auto-hide" : "Enable auto-hide"}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={autoHideEnabled ? 'enabled' : 'disabled'}
          initial={{ opacity: 0, rotate: -180 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: 180 }}
          transition={{ duration: 0.2 }}
        >
          {autoHideEnabled ? 
            <Eye size={20} className="drop-shadow-sm" /> : 
            <EyeOff size={20} className="drop-shadow-sm" />
          }
        </motion.div>
      </AnimatePresence>
      {showLabels && (
        <span className="ml-2 text-sm font-medium">
          {autoHideEnabled ? 'Auto Hide' : 'Manual'}
        </span>
      )}
    </motion.button>
  );

  // Device indicator (informational)
  const DeviceIndicator = () => (
    <div className="flex items-center gap-1 px-2 py-1 bg-slate-800/50 rounded-lg border border-slate-700/30">
      <motion.div
        animate={{ rotate: [0, 360] }}
        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
      >
        {isMobile ? 
          <Smartphone size={16} className="text-purple-400" /> : 
          <Monitor size={16} className="text-cyan-400" />
        }
      </motion.div>
      {showLabels && (
        <span className="text-xs font-mono text-slate-400">
          {viewportWidth}px
        </span>
      )}
    </div>
  );

  return (
    <div className={`${getVariantClasses()} ${getPositionClasses()} ${className}`}>
      <div className="flex items-center gap-2">
        
        {/* Navigation toggle - Always visible */}
        <NavigationToggleButton />
        
        {/* Left sidebar toggle - Desktop only */}
        {!isMobile && <LeftSidebarToggleButton />}
        
        {/* Separator */}
        {!isMobile && (
          <div className="w-px h-6 bg-slate-700/50" />
        )}
        
        {/* Smart toggle - Desktop only */}
        {!isMobile && <BothSidebarsToggleButton />}
        
        {/* Auto-hide toggle - Advanced feature */}
        {!isMobile && variant === 'floating' && (
          <AutoHideToggleButton />
        )}
        
        {/* Device indicator - Debug/info */}
        {variant === 'floating' && showLabels && (
          <DeviceIndicator />
        )}
        
        {/* Reset button - Advanced feature */}
        {variant === 'floating' && (
          <motion.button
            onClick={resetToDefaults}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`${buttonBaseClasses} text-orange-400 hover:text-orange-100 border-orange-500/30 hover:border-orange-400/60`}
            title="Reset to defaults"
            aria-label="Reset sidebar settings to defaults"
          >
            <Settings size={20} className="drop-shadow-sm" />
            {showLabels && (
              <span className="ml-2 text-sm font-medium">Reset</span>
            )}
          </motion.button>
        )}
      </div>
      
      {/* Status indicator */}
      <AnimatePresence>
        {autoHideEnabled && variant === 'floating' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 pt-2 border-t border-slate-700/30"
          >
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                Smart Mode
              </span>
              <span className="font-mono">
                {isMobile ? 'Mobile' : isDesktop ? 'Desktop' : 'Tablet'}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}