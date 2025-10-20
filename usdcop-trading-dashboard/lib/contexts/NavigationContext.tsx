/**
 * NavigationContext - Context for managing navigation state and performance
 * Enhanced to support EnhancedNavigationSidebar requirements
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

export interface ViewState {
  lastVisited: number;
  visitCount: number;
}

export interface NavigationState {
  activeView: string;
  viewHistory: string[];
  sidebarExpanded: boolean;
  isTransitioning: boolean;
  viewStates: Record<string, ViewState>;
  dataSource?: string;
}

export interface NavigationContextValue {
  state: NavigationState;
  changeView: (viewId: string) => void;
  goBack: () => void;
  toggleSidebarExpansion: () => void;
}

const NavigationContext = createContext<NavigationContextValue | undefined>(undefined);

export interface NavigationProviderProps {
  children: ReactNode;
  initialView?: string;
  onViewChange?: (viewId: string) => void;
}

export const NavigationProvider: React.FC<NavigationProviderProps> = ({
  children,
  initialView = 'dashboard-home',
  onViewChange
}) => {
  const [state, setState] = useState<NavigationState>({
    activeView: initialView,
    viewHistory: [initialView],
    sidebarExpanded: true,
    isTransitioning: false,
    viewStates: {
      [initialView]: {
        lastVisited: Date.now(),
        visitCount: 1
      }
    },
    dataSource: 'l0'
  });

  const changeView = useCallback((viewId: string) => {
    if (state.isTransitioning || state.activeView === viewId) return;

    setState(prev => ({
      ...prev,
      isTransitioning: true
    }));

    // Simulate transition delay for smooth UX
    setTimeout(() => {
      setState(prev => {
        const newViewStates = {
          ...prev.viewStates,
          [viewId]: {
            lastVisited: Date.now(),
            visitCount: (prev.viewStates[viewId]?.visitCount || 0) + 1
          }
        };

        return {
          ...prev,
          activeView: viewId,
          viewHistory: [viewId, ...prev.viewHistory.filter(v => v !== viewId)].slice(0, 10),
          isTransitioning: false,
          viewStates: newViewStates
        };
      });

      // Call optional callback
      if (onViewChange) {
        onViewChange(viewId);
      }
    }, 150);
  }, [state.isTransitioning, state.activeView, onViewChange]);

  const goBack = useCallback(() => {
    if (state.viewHistory.length > 1) {
      const previousView = state.viewHistory[1];
      changeView(previousView);
    }
  }, [state.viewHistory, changeView]);

  const toggleSidebarExpansion = useCallback(() => {
    setState(prev => ({
      ...prev,
      sidebarExpanded: !prev.sidebarExpanded
    }));
  }, []);

  const contextValue: NavigationContextValue = {
    state,
    changeView,
    goBack,
    toggleSidebarExpansion
  };

  return (
    <NavigationContext.Provider value={contextValue}>
      {children}
    </NavigationContext.Provider>
  );
};

export const useNavigation = () => {
  const context = useContext(NavigationContext);
  if (context === undefined) {
    throw new Error('useNavigation must be used within a NavigationProvider');
  }
  return context;
};

// Hook for navigation performance metrics
export const useNavigationPerformance = () => {
  const [totalTransitions, setTotalTransitions] = useState(0);
  const [averageTransitionTime, setAverageTransitionTime] = useState(0);
  const [transitionTimes, setTransitionTimes] = useState<number[]>([]);

  const recordTransition = useCallback((duration: number) => {
    setTotalTransitions(prev => prev + 1);
    setTransitionTimes(prev => {
      const newTimes = [...prev, duration].slice(-20); // Keep last 20
      const avg = newTimes.reduce((sum, time) => sum + time, 0) / newTimes.length;
      setAverageTransitionTime(avg);
      return newTimes;
    });
  }, []);

  return {
    totalTransitions,
    averageTransitionTime,
    recordTransition
  };
};

export default NavigationContext;