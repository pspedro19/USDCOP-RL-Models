/**
 * Views Configuration - Professional Trading Dashboard
 * =====================================================
 *
 * Configuración de las 4 vistas principales para usuarios finales
 * 100% datos reales desde el backend
 */

import {
  Activity,
  Signal,
  TrendingUp,
  Zap,
  Target,
  BarChart2
} from 'lucide-react';

export interface ViewConfig {
  id: string;
  name: string;
  icon: any; // LucideIcon
  category: 'Trading' | 'Analysis';
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
 * Professional Trading Views - 4 vistas principales
 */
export const VIEWS: ViewConfig[] = [
  // ===== TRADING (3 vistas) =====
  {
    id: 'live-terminal',
    name: 'Live Trading',
    icon: Signal,
    category: 'Trading',
    description: 'Real-time chart with live market data',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'trading-signals',
    name: 'Signals & Trades',
    icon: Zap,
    category: 'Trading',
    description: 'Multi-model signals and trade history',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },
  {
    id: 'executive-overview',
    name: 'Executive Dashboard',
    icon: TrendingUp,
    category: 'Trading',
    description: 'Portfolio KPIs and performance metrics',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  },

  // ===== ANALYSIS (1 vista) =====
  {
    id: 'backtest-results',
    name: 'Performance & Backtest',
    icon: Target,
    category: 'Analysis',
    description: 'Equity curves and backtest analysis',
    priority: 'high',
    enabled: true,
    requiresAuth: true
  }
];

/**
 * Category configuration - Solo 2 categorías profesionales
 */
export const CATEGORIES: Record<string, CategoryConfig> = {
  Trading: {
    name: 'Trading',
    color: 'from-cyan-400 to-blue-400',
    bgColor: 'from-cyan-500/10 to-blue-500/10',
    description: 'Real-time trading and monitoring'
  },
  Analysis: {
    name: 'Analysis',
    color: 'from-purple-400 to-pink-400',
    bgColor: 'from-purple-500/10 to-pink-500/10',
    description: 'Performance analysis and backtesting'
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
