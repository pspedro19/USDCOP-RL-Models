/**
 * NavigationContext - Context for managing navigation state and performance
 * This file was missing and causing compilation errors
 */

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export interface NavigationItem {
  id: string;
  label: string;
  icon?: React.ComponentType<any>;
  active?: boolean;
  section?: string;
}

export interface NavigationSection {
  section: string;
  items: NavigationItem[];
  color?: string;
  priority?: 'always-visible' | 'critical' | 'secondary';
}

export interface NavigationContextType {
  activeView: string;
  setActiveView: (view: string) => void;
  sidebarExpanded: boolean;
  setSidebarExpanded: (expanded: boolean) => void;
  navigationItems: NavigationSection[];
  setNavigationItems: (items: NavigationSection[]) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const NavigationContext = createContext<NavigationContextType | undefined>(undefined);

export interface NavigationProviderProps {
  children: ReactNode;
  initialItems?: NavigationSection[];
}

export const NavigationProvider: React.FC<NavigationProviderProps> = ({
  children,
  initialItems = []
}) => {
  const [activeView, setActiveView] = useState('terminal');
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [navigationItems, setNavigationItems] = useState<NavigationSection[]>(initialItems);
  const [isLoading, setIsLoading] = useState(false);

  const contextValue: NavigationContextType = {
    activeView,
    setActiveView,
    sidebarExpanded,
    setSidebarExpanded,
    navigationItems,
    setNavigationItems,
    isLoading,
    setIsLoading,
  };

  return (
    <NavigationContext.Provider value={contextValue}>
      {children}
    </NavigationContext.Provider>
  );
};

export const useNavigation = (): NavigationContextType => {
  const context = useContext(NavigationContext);
  if (context === undefined) {
    throw new Error('useNavigation must be used within a NavigationProvider');
  }
  return context;
};

// Hook for navigation performance metrics
export const useNavigationPerformance = () => {
  const [metrics, setMetrics] = useState({
    navigationCount: 0,
    averageNavigationTime: 0,
    lastNavigationTime: Date.now(),
  });

  const trackNavigation = useCallback((startTime: number) => {
    const endTime = Date.now();
    const navigationTime = endTime - startTime;

    setMetrics(prev => ({
      navigationCount: prev.navigationCount + 1,
      averageNavigationTime: (prev.averageNavigationTime + navigationTime) / 2,
      lastNavigationTime: endTime,
    }));
  }, []);

  return {
    metrics,
    trackNavigation,
  };
};

export default NavigationContext;