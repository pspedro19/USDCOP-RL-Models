'use client';

import React from 'react';

// Import ONLY the 4 professional trading views
import LiveTradingTerminal from './views/LiveTradingTerminal';
import TradingSignalsMultiModel from './views/TradingSignalsMultiModel';
import ExecutiveOverview from './views/ExecutiveOverview';
import L6BacktestResults from './views/L6BacktestResults';

interface ViewRendererProps {
  activeView: string;
}

const ViewRenderer: React.FC<ViewRendererProps> = ({ activeView }) => {
  // Professional Trading Views - 4 vistas principales
  const viewComponents: Record<string, React.ComponentType> = {
    'live-terminal': LiveTradingTerminal,          // Live Trading - Chart + Real-time metrics
    'trading-signals': TradingSignalsMultiModel,   // Signals & Trades - Multi-model signals + Trade history
    'executive-overview': ExecutiveOverview,       // Executive Dashboard - Portfolio KPIs
    'backtest-results': L6BacktestResults,         // Performance & Backtest - Equity curves
  };

  // Get the component for the active view
  const ViewComponent = viewComponents[activeView];

  // Default to Live Trading if view not found
  if (!ViewComponent) {
    return <LiveTradingTerminal />;
  }

  return <ViewComponent />;
};

export default ViewRenderer;