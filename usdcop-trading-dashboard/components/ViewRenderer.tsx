'use client';

import React from 'react';

// Import all view components
import TradingTerminalView from './views/TradingTerminalView';
import EnhancedTradingTerminal from './views/EnhancedTradingTerminal';
import ProfessionalTradingTerminal from './views/ProfessionalTradingTerminal';
import ProfessionalTradingTerminalSimplified from './views/ProfessionalTradingTerminalSimplified';
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
import UltimateVisualDashboard from './views/UltimateVisualDashboard';
import AuditCompliance from './views/AuditCompliance';
import UnifiedTradingTerminal from './views/UnifiedTradingTerminal';

interface ViewRendererProps {
  activeView: string;
}

const ViewRenderer: React.FC<ViewRendererProps> = ({ activeView }) => {
  // Map of view IDs to components - SOLO 2 VISTAS
  const viewComponents: Record<string, React.ComponentType> = {
    'dashboard-home': UnifiedTradingTerminal,
    'professional-terminal': UnifiedTradingTerminal,
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