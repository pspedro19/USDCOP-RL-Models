'use client';

import React from 'react';

// Import all view components
import TradingTerminalView from './views/TradingTerminalView';
import EnhancedTradingTerminal from './views/EnhancedTradingTerminal';
import ProfessionalTradingTerminalSimplified from './views/ProfessionalTradingTerminalSimplified';
import RealTimeChart from './views/RealTimeChart';
import TradingSignals from './views/TradingSignals';
import BacktestResults from './views/BacktestResults';
import RLModelHealth from './views/RLModelHealth';
import RiskManagement from './views/RiskManagement';
// import RealTimeRiskMonitor from './views/RealTimeRiskMonitor'; // REMOVED - Not needed
// import PortfolioExposureAnalysis from './views/PortfolioExposureAnalysis'; // REMOVED - Not needed
// import RiskAlertsCenter from './views/RiskAlertsCenter'; // REMOVED - Not needed
import L0RawDataDashboard from './views/L0RawDataDashboard';
import L1FeatureStats from './views/L1FeatureStats';
import L2PreparedData from './views/L2PreparedData';
import DataPipelineQuality from './views/DataPipelineQuality';
import L3Correlations from './views/L3Correlations';
import L4RLReadyData from './views/L4RLReadyData';
import L5ModelDashboard from './views/L5ModelDashboard';
import L6BacktestResults from './views/L6BacktestResults';
import ExecutiveOverview from './views/ExecutiveOverview';
import LiveTradingTerminal from './views/LiveTradingTerminal';
import UltimateVisualDashboard from './views/UltimateVisualDashboard';
import AuditCompliance from './views/AuditCompliance';
import UnifiedTradingTerminal from './views/UnifiedTradingTerminal';
import PipelineStatus from './views/PipelineStatusV2';
import TradingSignalsMultiModel from './views/TradingSignalsMultiModel';

interface ViewRendererProps {
  activeView: string;
}

const ViewRenderer: React.FC<ViewRendererProps> = ({ activeView }) => {
  // Map of view IDs to components - CLEANED: Removed duplicates
  const viewComponents: Record<string, React.ComponentType> = {
    // Trading Views (3 total) - REMOVED dashboard-home
    'live-terminal': LiveTradingTerminal,
    'executive-overview': ExecutiveOverview,
    'trading-signals': TradingSignalsMultiModel,  // UPDATED: Multi-model version

    // Risk Management (2 total) - REMOVED per user request
    // 'risk-monitor': RealTimeRiskMonitor,
    // 'risk-alerts': RiskAlertsCenter,

    // Data Pipeline L0-L6 (7 total)
    'pipeline-status': PipelineStatus,
    'l0-raw-data': L0RawDataDashboard,
    'l1-features': L1FeatureStats,
    'l2-prepare': L2PreparedData,  // Dedicated L2 component with real data
    'l3-correlations': L3Correlations,
    'l4-rl-ready': L4RLReadyData,
    'l5-model': L5ModelDashboard,

    // Analysis & Backtest (1 total) - Merged L6 into backtest-results
    'backtest-results': L6BacktestResults,
  };

  // Get the component for the active view
  const ViewComponent = viewComponents[activeView];

  // If no component found, show default UnifiedTradingTerminal
  if (!ViewComponent) {
    return <UnifiedTradingTerminal />;
  }

  return <ViewComponent />;
};

export default ViewRenderer;