'use client';

import React from 'react';

// Import all view components
import TradingTerminalView from './views/TradingTerminalView';
import EnhancedTradingTerminal from './views/EnhancedTradingTerminal';
import RealTimeChart from './views/RealTimeChart';
import TradingSignals from './views/TradingSignals';
import BacktestResults from './views/BacktestResults';
import RLModelHealth from './views/RLModelHealth';
import RiskManagement from './views/RiskManagement';
import RealTimeRiskMonitor from './views/RealTimeRiskMonitor';
import PortfolioExposureAnalysis from './views/PortfolioExposureAnalysis';
import RiskAlertsCenter from './views/RiskAlertsCenter';
import L0RawDataDashboard from './views/L0RawDataDashboard';
import L1FeatureStats from './views/L1FeatureStats';
import DataPipelineQuality from './views/DataPipelineQuality';
import L3Correlations from './views/L3Correlations';
import L4RLReadyData from './views/L4RLReadyData';
import L5ModelDashboard from './views/L5ModelDashboard';
import L6BacktestResults from './views/L6BacktestResults';
import ExecutiveOverview from './views/ExecutiveOverview';
import LiveTradingTerminal from './views/LiveTradingTerminal';
import AuditCompliance from './views/AuditCompliance';

interface ViewRendererProps {
  activeView: string;
}

const ViewRenderer: React.FC<ViewRendererProps> = ({ activeView }) => {
  // Map of view IDs to components
  const viewComponents: Record<string, React.ComponentType> = {
    // TRADING MODULE
    'terminal': EnhancedTradingTerminal,
    'terminal-advanced': TradingTerminalView,
    'live-terminal': LiveTradingTerminal,
    'realtime': RealTimeChart,
    'signals': TradingSignals,
    'backtest': BacktestResults,
    'ml-analytics': RLModelHealth,
    
    // RISK MANAGEMENT
    'portfolio': RiskManagement,
    'realtime-risk': RealTimeRiskMonitor,
    'exposure': PortfolioExposureAnalysis,
    'alerts': RiskAlertsCenter,
    
    // DATA PIPELINE AVANZADO
    'l0': L0RawDataDashboard,
    'l1': L1FeatureStats,
    'l2': DataPipelineQuality,
    'l3': L3Correlations,
    'l4': L4RLReadyData,
    'l5': L5ModelDashboard,
    'l6': L6BacktestResults,
    
    // EXECUTIVE & COMPLIANCE
    'executive': ExecutiveOverview,
    'audit': AuditCompliance,
  };

  // Get the component for the active view
  const ViewComponent = viewComponents[activeView];

  // If no component found, show default enhanced trading terminal
  if (!ViewComponent) {
    return <EnhancedTradingTerminal />;
  }

  return <ViewComponent />;
};

export default ViewRenderer;