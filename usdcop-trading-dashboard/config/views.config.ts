/**
 * Views Configuration - Dynamic Navigation System
 * ==============================================
 *
 * Configuración centralizada de todas las vistas del sistema
 * 100% configurable - NO hardcoded en componentes
 *
 * Benefits:
 * - Single source of truth para navegación
 * - Fácil agregar/remover vistas
 * - Configuración por roles (futuro)
 * - Puede extenderse para cargar desde API/database
 */

import {
  Activity,
  LineChart,
  Signal,
  TrendingUp,
  Zap,
  Shield,
  AlertTriangle,
  Database,
  BarChart3,
  GitBranch,
  Brain,
  Cpu,
  Target
} from 'lucide-react';

export interface ViewConfig {
  id: string;
  name: string;
  icon: any; // LucideIcon
  category: 'Trading' | 'Risk' | 'Pipeline' | 'System';
  description: string;
  priority: 'high' | 'medium' | 'low';
  enabled?: boolean;
  requiresAuth?: boolean;
  requiredRole?: string[];
}

export interface CategoryConfig {
  name: string;
  color: string;
  bgColor: string;
  description: string;
}

/**
 * All available views in the system
 * Easy to modify, extend, or load from API
 */
export const VIEWS: ViewConfig[] = [
  // ===== TRADING VIEWS (5) =====
  {
    id: 'dashboard-home',
    name: 'Dashboard Home',
    icon: Activity,
    category: 'Trading',
    description: 'Professional USDCOP trading chart with full features',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'professional-terminal',
    name: 'Professional Terminal',
    icon: LineChart,
    category: 'Trading',
    description: 'Advanced professional trading terminal',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'live-terminal',
    name: 'Live Trading',
    icon: Signal,
    category: 'Trading',
    description: 'Real-time trading terminal with live data',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'executive-overview',
    name: 'Executive Overview',
    icon: TrendingUp,
    category: 'Trading',
    description: 'Executive trading dashboard overview',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'trading-signals',
    name: 'Trading Signals',
    icon: Zap,
    category: 'Trading',
    description: 'AI-powered trading signals',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },

  // ===== RISK MANAGEMENT (2) =====
  {
    id: 'risk-monitor',
    name: 'Risk Monitor',
    icon: Shield,
    category: 'Risk',
    description: 'Real-time risk monitoring and alerts',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'risk-alerts',
    name: 'Risk Alerts',
    icon: AlertTriangle,
    category: 'Risk',
    description: 'Risk alert management center',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },

  // ===== DATA PIPELINE L0-L5 (5) =====
  {
    id: 'l0-raw-data',
    name: 'L0 - Raw Data',
    icon: Database,
    category: 'Pipeline',
    description: 'Raw USDCOP market data visualization',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'l1-features',
    name: 'L1 - Features',
    icon: BarChart3,
    category: 'Pipeline',
    description: 'Feature statistics and analysis',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'l3-correlations',
    name: 'L3 - Correlations',
    icon: GitBranch,
    category: 'Pipeline',
    description: 'Correlation matrix and analysis',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'l4-rl-ready',
    name: 'L4 - RL Ready',
    icon: Brain,
    category: 'Pipeline',
    description: 'RL-ready data preparation dashboard',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'l5-model',
    name: 'L5 - Model',
    icon: Cpu,
    category: 'Pipeline',
    description: 'Model performance and metrics',
    priority: 'medium',
    enabled: true,
    requiresAuth: true
  },

  // ===== SYSTEM (1) =====
  {
    id: 'backtest-results',
    name: 'Backtest Results',
    icon: Target,
    category: 'System',
    description: 'Comprehensive backtest analysis and L6 results',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  }
];

/**
 * Category configuration
 */
export const CATEGORIES: Record<string, CategoryConfig> = {
  Trading: {
    name: 'Trading',
    color: 'from-cyan-400 to-blue-400',
    bgColor: 'from-cyan-500/10 to-blue-500/10',
    description: 'Real-time trading terminals and analysis'
  },
  Risk: {
    name: 'Risk',
    color: 'from-red-400 to-orange-400',
    bgColor: 'from-red-500/10 to-orange-500/10',
    description: 'Risk management and monitoring'
  },
  Pipeline: {
    name: 'Pipeline',
    color: 'from-green-400 to-emerald-400',
    bgColor: 'from-green-500/10 to-emerald-500/10',
    description: 'Data processing pipeline L0-L5'
  },
  System: {
    name: 'System',
    color: 'from-purple-400 to-pink-400',
    bgColor: 'from-purple-500/10 to-pink-500/10',
    description: 'System analysis and backtesting'
  }
};

/**
 * Priority configuration
 */
export const PRIORITY_CONFIG = {
  high: {
    indicator: 'bg-red-400',
    label: 'High Priority',
    order: 1
  },
  medium: {
    indicator: 'bg-yellow-400',
    label: 'Medium Priority',
    order: 2
  },
  low: {
    indicator: 'bg-green-400',
    label: 'Low Priority',
    order: 3
  }
};

/**
 * Helper functions
 */

/**
 * Get all enabled views
 */
export function getEnabledViews(): ViewConfig[] {
  return VIEWS.filter(view => view.enabled !== false);
}

/**
 * Get views by category
 */
export function getViewsByCategory(category: string): ViewConfig[] {
  return VIEWS.filter(view => view.category === category && view.enabled !== false);
}

/**
 * Get view by ID
 */
export function getViewById(id: string): ViewConfig | undefined {
  return VIEWS.find(view => view.id === id);
}

/**
 * Get all categories with view counts
 */
export function getCategoriesWithCounts(): Array<{ category: string; count: number; config: CategoryConfig }> {
  const categories = Object.keys(CATEGORIES);
  return categories.map(category => ({
    category,
    count: getViewsByCategory(category).length,
    config: CATEGORIES[category]
  }));
}

/**
 * Get high priority views (for quick access)
 */
export function getHighPriorityViews(limit: number = 5): ViewConfig[] {
  return VIEWS
    .filter(view => view.priority === 'high' && view.enabled !== false)
    .slice(0, limit);
}

/**
 * Future: Load views from API
 * This function can be extended to fetch views from backend
 */
export async function loadViewsFromAPI(): Promise<ViewConfig[]> {
  // TODO: Implement API call to backend
  // const response = await fetch('/api/views/config');
  // return await response.json();

  // For now, return static config
  return VIEWS;
}

/**
 * Export total count
 */
export const TOTAL_VIEWS = VIEWS.filter(v => v.enabled !== false).length;
