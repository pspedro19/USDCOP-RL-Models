'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Monitor, 
  Smartphone, 
  Tablet, 
  Eye, 
  EyeOff,
  Settings,
  Info,
  Check,
  X
} from 'lucide-react';
import { useSidebarState } from '@/hooks/useSidebarState';
import { useResponsiveLayout, useBreakpoints } from '@/hooks/useResponsiveLayout';
import { SidebarToggleButtons } from '@/components/ui/SidebarToggleButtons';

/**
 * Componente de ejemplo que demuestra todas las funcionalidades
 * del sistema de sidebar inteligente y dinámico
 */
export function SidebarSystemExample() {
  const sidebarState = useSidebarState();
  const responsiveLayout = useResponsiveLayout(sidebarState);
  const breakpoints = useBreakpoints(sidebarState.viewportWidth);

  const {
    leftSidebarCollapsed,
    navigationSidebarOpen,
    isDesktop,
    isMobile,
    viewportWidth,
    autoHideEnabled,
    leftSidebarWidth,
    navigationSidebarWidth,
    totalSidebarWidth,
    isLeftSidebarVisible,
    isNavigationSidebarVisible
  } = sidebarState;

  const statusItems = [
    {
      label: 'Viewport Width',
      value: `${viewportWidth}px`,
      icon: viewportWidth >= 1280 ? Monitor : viewportWidth >= 1024 ? Tablet : Smartphone,
      color: 'text-cyan-400'
    },
    {
      label: 'Device Type',
      value: isMobile ? 'Mobile' : isDesktop ? 'Desktop' : 'Tablet',
      icon: isMobile ? Smartphone : isDesktop ? Monitor : Tablet,
      color: isMobile ? 'text-purple-400' : isDesktop ? 'text-green-400' : 'text-yellow-400'
    },
    {
      label: 'Auto-Hide',
      value: autoHideEnabled ? 'Enabled' : 'Disabled',
      icon: autoHideEnabled ? Eye : EyeOff,
      color: autoHideEnabled ? 'text-green-400' : 'text-red-400'
    },
    {
      label: 'Left Sidebar',
      value: leftSidebarCollapsed ? 'Collapsed' : 'Expanded',
      icon: leftSidebarCollapsed ? X : Check,
      color: leftSidebarCollapsed ? 'text-red-400' : 'text-green-400'
    },
    {
      label: 'Navigation Sidebar',
      value: navigationSidebarOpen ? 'Open' : 'Closed',
      icon: navigationSidebarOpen ? Check : X,
      color: navigationSidebarOpen ? 'text-green-400' : 'text-red-400'
    }
  ];

  const layoutInfo = [
    {
      label: 'Left Sidebar Width',
      value: `${leftSidebarWidth}px`,
      visible: isLeftSidebarVisible
    },
    {
      label: 'Navigation Width',
      value: `${navigationSidebarWidth}px`,
      visible: isNavigationSidebarVisible
    },
    {
      label: 'Total Sidebar Width',
      value: `${totalSidebarWidth}px`,
      visible: true
    },
    {
      label: 'Available Content Width',
      value: `${responsiveLayout.availableWidth}px`,
      visible: true
    },
    {
      label: 'Is Full Width',
      value: responsiveLayout.isFullWidth ? 'Yes' : 'No',
      visible: true
    }
  ];

  return (
    <div className="p-6 space-y-8 bg-slate-900/50 backdrop-blur-sm rounded-xl border border-slate-700/50">
      {/* Header */}
      <div className="text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center gap-3 mb-4"
        >
          <div className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl">
            <Settings className="w-8 h-8 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Sidebar System Demo
            </h2>
            <p className="text-slate-400 text-sm">
              Intelligent & Dynamic Sidebar Management
            </p>
          </div>
        </motion.div>
      </div>

      {/* Toggle Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-slate-800/30 rounded-xl p-6 border border-slate-700/30"
      >
        <h3 className="text-lg font-semibold text-cyan-300 mb-4 flex items-center gap-2">
          <Info className="w-5 h-5" />
          Control Panel
        </h3>
        
        <div className="flex flex-wrap gap-4 justify-center">
          {/* Different variants of toggle buttons */}
          <SidebarToggleButtons
            {...sidebarState}
            variant="integrated"
            showLabels={true}
          />
        </div>
        
        <div className="mt-4 text-xs text-slate-500 text-center">
          Try the buttons above to test the sidebar system functionality
        </div>
      </motion.div>

      {/* Status Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
      >
        {statusItems.map((item, index) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30"
          >
            <div className="flex items-center gap-3 mb-2">
              <item.icon className={`w-5 h-5 ${item.color}`} />
              <h4 className="text-sm font-medium text-slate-300">{item.label}</h4>
            </div>
            <p className={`text-lg font-bold ${item.color}`}>{item.value}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Layout Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-slate-800/30 rounded-xl p-6 border border-slate-700/30"
      >
        <h3 className="text-lg font-semibold text-cyan-300 mb-4">
          Layout Calculations
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {layoutInfo.map((item, index) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.05 }}
              className={`p-3 rounded-lg border ${
                item.visible 
                  ? 'bg-slate-700/30 border-slate-600/50 text-slate-300' 
                  : 'bg-slate-800/20 border-slate-700/30 text-slate-500'
              }`}
            >
              <div className="text-xs text-slate-400 mb-1">{item.label}</div>
              <div className="font-mono font-semibold">{item.value}</div>
              {!item.visible && (
                <div className="text-xs text-slate-500 mt-1">Not visible</div>
              )}
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Breakpoint Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-slate-800/30 rounded-xl p-6 border border-slate-700/30"
      >
        <h3 className="text-lg font-semibold text-cyan-300 mb-4">
          Responsive Breakpoints
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {Object.entries({
            'XS': viewportWidth < 640,
            'SM': breakpoints.isSm && !breakpoints.isMd,
            'MD': breakpoints.isMd && !breakpoints.isLg,
            'LG': breakpoints.isLg && !breakpoints.isXl,
            'XL': breakpoints.isXl && !breakpoints.is2Xl,
            '2XL': breakpoints.is2Xl
          }).map(([name, active]) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 }}
              className={`p-3 rounded-lg text-center font-mono text-sm transition-all ${
                active 
                  ? 'bg-cyan-500/20 border-2 border-cyan-400/50 text-cyan-300 shadow-glow-cyan' 
                  : 'bg-slate-700/30 border border-slate-600/50 text-slate-400'
              }`}
            >
              {name}
              {active && (
                <motion.div
                  className="w-2 h-2 bg-cyan-400 rounded-full mx-auto mt-1"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
            </motion.div>
          ))}
        </div>
        
        <div className="mt-4 text-xs text-slate-500 text-center">
          Current breakpoint: <span className="text-cyan-400 font-semibold">{breakpoints.breakpoint.toUpperCase()}</span>
        </div>
      </motion.div>

      {/* CSS Classes Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="bg-slate-800/30 rounded-xl p-6 border border-slate-700/30"
      >
        <h3 className="text-lg font-semibold text-cyan-300 mb-4">
          Generated CSS Classes
        </h3>
        
        <div className="space-y-3">
          <div>
            <label className="text-sm text-slate-400 mb-1 block">Container Classes:</label>
            <code className="block p-3 bg-slate-900/50 rounded-lg text-cyan-300 font-mono text-sm break-all">
              {responsiveLayout.containerClasses}
            </code>
          </div>
          
          <div>
            <label className="text-sm text-slate-400 mb-1 block">Main Content Margin:</label>
            <code className="block p-3 bg-slate-900/50 rounded-lg text-cyan-300 font-mono text-sm">
              {responsiveLayout.mainContentMarginLeft}
            </code>
          </div>
          
          <div>
            <label className="text-sm text-slate-400 mb-1 block">Padding Classes:</label>
            <code className="block p-3 bg-slate-900/50 rounded-lg text-cyan-300 font-mono text-sm break-all">
              {responsiveLayout.mainContentPaddingLeft} {responsiveLayout.mainContentPaddingRight}
            </code>
          </div>
        </div>
      </motion.div>

      {/* Instructions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-xl p-6 border border-cyan-500/20"
      >
        <h3 className="text-lg font-semibold text-cyan-300 mb-4">
          How to Test
        </h3>
        
        <div className="space-y-2 text-sm text-slate-300">
          <p>• <strong>Resize your browser window</strong> to see responsive behavior</p>
          <p>• <strong>Use the toggle buttons</strong> to control sidebar states</p>
          <p>• <strong>Try keyboard shortcuts:</strong> Ctrl+B (left sidebar), Escape (close navigation)</p>
          <p>• <strong>Check localStorage</strong> to see persistence in action</p>
          <p>• <strong>Open DevTools</strong> to inspect responsive breakpoints</p>
        </div>
      </motion.div>
    </div>
  );
}