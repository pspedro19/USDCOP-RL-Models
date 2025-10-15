'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useNavigation, useNavigationPerformance } from '@/lib/contexts/NavigationContext';
import { 
  LineChart, 
  Signal, 
  TrendingUp, 
  Database, 
  Shield, 
  Brain,
  Activity,
  BarChart3,
  GitBranch,
  Cpu,
  Target,
  PieChart,
  Zap,
  Key,
  ChevronRight,
  Sparkles,
  Clock,
  Gauge,
  ArrowLeft,
  Settings,
  Maximize2,
  Minimize2,
  AlertTriangle,
  Bell,
  Thermometer
} from 'lucide-react';

const views = [
  // Trading views - Solo 2 vistas principales
  { id: 'dashboard-home', name: 'Dashboard Home', icon: Activity, category: 'Trading', description: 'Professional USDCOP trading chart with full features', priority: 'high' },
  { id: 'professional-terminal', name: 'Professional Terminal', icon: LineChart, category: 'Trading', description: 'Advanced professional trading terminal', priority: 'high' },
];

const categoryConfig = {
  Trading: { color: 'from-cyan-400 to-blue-400', bgColor: 'from-cyan-500/10 to-blue-500/10' },
  Risk: { color: 'from-red-400 to-orange-400', bgColor: 'from-red-500/10 to-orange-500/10' },
  Pipeline: { color: 'from-green-400 to-emerald-400', bgColor: 'from-green-500/10 to-emerald-500/10' },
  System: { color: 'from-purple-400 to-pink-400', bgColor: 'from-purple-500/10 to-pink-500/10' }
};

const priorityConfig = {
  high: { indicator: 'bg-red-400', label: 'High Priority' },
  medium: { indicator: 'bg-yellow-400', label: 'Medium Priority' },
  low: { indicator: 'bg-green-400', label: 'Low Priority' }
};

export function EnhancedNavigationSidebar() {
  const { state, changeView, goBack, toggleSidebarExpansion } = useNavigation();
  const performance = useNavigationPerformance();
  
  const categories = ['Trading', 'Risk', 'Pipeline', 'System'];
  const canGoBack = state.viewHistory.length > 1;
  
  // Get frequently used views
  const frequentViews = views
    .filter(view => state.viewStates[view.id]?.lastVisited)
    .sort((a, b) => (state.viewStates[b.id]?.lastVisited || 0) - (state.viewStates[a.id]?.lastVisited || 0))
    .slice(0, 3);

  return (
    <div className={`${state.sidebarExpanded ? 'w-80' : 'w-16'} bg-slate-900/95 backdrop-blur-xl border-r border-slate-700/50 h-full flex flex-col transition-all duration-300 relative overflow-hidden`}>
      {/* Sidebar Glow Effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 to-purple-500/5" />
      <div className="absolute top-0 bottom-0 left-0 w-[1px] bg-gradient-to-b from-transparent via-blue-500/30 to-transparent" />
      
      {/* Header */}
      <div className="p-4 border-b border-slate-700/50 relative z-10">
        <motion.div 
          className="flex items-center justify-between cursor-pointer"
          onClick={toggleSidebarExpansion}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <AnimatePresence>
            {state.sidebarExpanded && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center gap-3"
              >
                <div className="relative">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/25">
                    <Sparkles className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-r from-emerald-400 to-green-500 rounded-full border border-slate-900 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-white font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">USD/COP Navigation</h2>
                  <p className="text-xs text-slate-400">16 Professional Views</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <motion.div
            animate={{ rotate: state.sidebarExpanded ? 0 : 180 }}
            transition={{ duration: 0.3 }}
          >
            {state.sidebarExpanded ? <Minimize2 className="w-5 h-5 text-slate-400" /> : <Maximize2 className="w-5 h-5 text-slate-400" />}
          </motion.div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <AnimatePresence>
        {state.sidebarExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-4 border-b border-slate-700/50 space-y-2"
          >
            {/* Back Button */}
            <motion.button
              onClick={goBack}
              disabled={!canGoBack}
              whileHover={canGoBack ? { scale: 1.02, x: 5 } : {}}
              whileTap={canGoBack ? { scale: 0.98 } : {}}
              className={`w-full p-3 rounded-lg flex items-center gap-3 transition-all ${
                canGoBack 
                  ? 'bg-slate-800/50 text-slate-300 hover:bg-slate-800/70 hover:text-cyan-300' 
                  : 'bg-slate-800/30 text-slate-500 cursor-not-allowed'
              }`}
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">Go Back</span>
              {canGoBack && state.viewHistory[1] && (
                <span className="text-xs text-slate-500 ml-auto">
                  {views.find(v => v.id === state.viewHistory[1])?.name || 'Previous'}
                </span>
              )}
            </motion.button>

            {/* Performance Indicator */}
            {performance.totalTransitions > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 bg-slate-800/30 rounded-lg border border-slate-700/30"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-medium text-slate-300">Navigation Performance</span>
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Avg Transition</span>
                    <span className="text-purple-400 font-mono">{performance.averageTransitionTime.toFixed(0)}ms</span>
                  </div>
                  <div className="w-full bg-slate-700/50 rounded-full h-1">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-pink-400 h-1 rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min(100, (performance.averageTransitionTime / 1000) * 100)}%` }}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Frequent Views */}
      <AnimatePresence>
        {state.sidebarExpanded && frequentViews.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4 border-b border-slate-700/50"
          >
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              Frequently Used
            </h3>
            <div className="space-y-1">
              {frequentViews.map((view) => (
                <motion.button
                  key={`frequent-${view.id}`}
                  onClick={() => changeView(view.id)}
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 group ${
                    state.activeView === view.id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-300'
                      : 'hover:bg-slate-800/50 text-slate-300 hover:text-cyan-300'
                  }`}
                >
                  <view.icon size={16} className={`${
                    state.activeView === view.id ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'
                  } transition-colors duration-200`} />
                  <span className="text-sm font-medium truncate">{view.name}</span>
                  <Clock className="w-3 h-3 text-slate-500 ml-auto" />
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Navigation */}
      <div className="flex-1 overflow-y-auto py-4 px-3">
        {state.sidebarExpanded ? (
          // Expanded view with categories
          categories.map((category) => (
            <div key={category} className="mb-6">
              <div className={`bg-gradient-to-r ${categoryConfig[category as keyof typeof categoryConfig].bgColor} rounded-lg p-3 mb-3`}>
                <h3 className={`text-sm font-bold bg-gradient-to-r ${categoryConfig[category as keyof typeof categoryConfig].color} bg-clip-text text-transparent uppercase tracking-wide`}>
                  {category}
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  {views.filter(view => view.category === category).length} views available
                </p>
              </div>
              
              <div className="space-y-1">
                {views.filter(view => view.category === category).map((view, index) => (
                  <motion.button
                    key={view.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => changeView(view.id)}
                    disabled={state.isTransitioning}
                    whileHover={{ scale: 1.02, x: 5 }}
                    whileTap={{ scale: 0.98 }}
                    className={`w-full p-3 rounded-xl flex items-center gap-3 transition-all duration-200 group disabled:opacity-50 relative ${
                      state.activeView === view.id 
                        ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-300 border border-cyan-500/30' 
                        : 'hover:bg-slate-800/50 text-slate-300 hover:text-cyan-300'
                    }`}
                  >
                    {/* Priority indicator */}
                    <div className={`w-2 h-2 rounded-full ${priorityConfig[view.priority as keyof typeof priorityConfig].indicator} ${
                      view.priority === 'high' ? 'animate-pulse' : ''
                    }`} />
                    
                    <view.icon size={18} className={`${
                      state.activeView === view.id ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'
                    } transition-colors duration-200`} />
                    
                    <div className="text-left flex-1">
                      <div className="text-sm font-medium">{view.name}</div>
                      <div className="text-xs text-slate-500 group-hover:text-slate-400">{view.description}</div>
                    </div>
                    
                    {state.activeView === view.id && (
                      <motion.div
                        layoutId="activeViewIndicator"
                        className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"
                      />
                    )}
                    
                    {state.isTransitioning && state.activeView === view.id && (
                      <div className="w-4 h-4 border-2 border-cyan-400/20 border-t-cyan-400 rounded-full animate-spin" />
                    )}
                    
                    {/* Last visited indicator */}
                    {state.viewStates[view.id]?.lastVisited && (
                      <div className="absolute top-1 right-1 w-1 h-1 bg-green-400 rounded-full" />
                    )}
                  </motion.button>
                ))}
              </div>
            </div>
          ))
        ) : (
          // Collapsed view with icons only
          <div className="space-y-2">
            {views.map((view) => (
              <motion.button
                key={view.id}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => changeView(view.id)}
                className={`w-full p-3 rounded-lg flex justify-center relative group ${
                  state.activeView === view.id ? 'bg-cyan-500/20 border border-cyan-500/50' : 'hover:bg-slate-800/50'
                }`}
                title={view.name}
              >
                <view.icon size={20} className={`${
                  state.activeView === view.id ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'
                } transition-colors duration-200`} />
                
                {/* Priority indicator for collapsed view */}
                <div className={`absolute top-1 right-1 w-2 h-2 rounded-full ${priorityConfig[view.priority as keyof typeof priorityConfig].indicator} ${
                  view.priority === 'high' ? 'animate-pulse' : ''
                }`} />
                
                {/* Tooltip for collapsed view */}
                <div className="absolute left-full ml-2 px-2 py-1 bg-slate-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                  {view.name}
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <AnimatePresence>
        {state.sidebarExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4 border-t border-slate-700/50 relative z-10"
          >
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-medium">System Status</span>
              <motion.div 
                className="w-2 h-2 bg-green-400 rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Active View</span>
                <span className="text-cyan-400 font-mono">
                  {views.find(v => v.id === state.activeView)?.name || 'Unknown'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Data Source</span>
                <span className={`font-mono ${
                  state.dataSource === 'l0' ? 'text-blue-400' :
                  state.dataSource === 'l1' ? 'text-green-400' : 'text-purple-400'
                }`}>
                  {state.dataSource.toUpperCase()}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}