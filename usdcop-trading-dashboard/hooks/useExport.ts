/**
 * Professional Export Hooks for Dashboard Components
 * 
 * Provides seamless integration of export functionality with React components
 */

import { useCallback, useRef } from 'react';
import { toast } from 'react-hot-toast';
import {
  ProfessionalExportManager,
  exportConfigs,
  ChartData,
  TableData,
  MetricData
} from '@/lib/services/export-manager';

export const useBacktestExport = () => {
  const chartRefs = useRef<{ [key: string]: HTMLElement }>({});
  
  const setChartRef = useCallback((key: string, element: HTMLElement | null) => {
    if (element) {
      chartRefs.current[key] = element;
    }
  }, []);
  
  const exportToPDF = useCallback(async (backtestData: any) => {
    const loadingToast = toast.loading('Generating backtest report...');
    
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.backtestReport);
      
      const charts: ChartData[] = Object.entries(chartRefs.current).map(([key, element]) => ({
        element,
        title: getChartTitle(key),
        width: 170,
        height: 120
      }));
      
      await exporter.exportBacktestResults(backtestData);
      
      toast.success('Backtest report exported successfully!', { id: loadingToast });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export backtest report', { id: loadingToast });
    }
  }, []);
  
  const exportToCSV = useCallback((backtestData: any) => {
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.backtestReport);
      
      // Export trades data
      if (backtestData.trades && backtestData.trades.length > 0) {
        exporter.exportToCSV(backtestData.trades, 'backtest_trades');
        toast.success('Trades data exported to CSV!');
      } else {
        toast.error('No trades data available for export');
      }
    } catch (error) {
      console.error('CSV export failed:', error);
      toast.error('Failed to export CSV');
    }
  }, []);
  
  const exportToExcel = useCallback((backtestData: any) => {
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.backtestReport);
      
      const sheets = [
        { name: 'Summary', data: [backtestData.summary] },
        { name: 'Monthly Returns', data: backtestData.monthlyReturns || [] },
        { name: 'Trades', data: backtestData.trades || [] },
        { name: 'Drawdowns', data: backtestData.drawdowns || [] }
      ].filter(sheet => sheet.data.length > 0);
      
      exporter.exportToExcel(sheets);
      toast.success('Backtest data exported to Excel!');
    } catch (error) {
      console.error('Excel export failed:', error);
      toast.error('Failed to export Excel');
    }
  }, []);
  
  return {
    setChartRef,
    exportToPDF,
    exportToCSV,
    exportToExcel
  };
};

export const useRiskExport = () => {
  const chartRefs = useRef<{ [key: string]: HTMLElement }>({});
  
  const setChartRef = useCallback((key: string, element: HTMLElement | null) => {
    if (element) {
      chartRefs.current[key] = element;
    }
  }, []);
  
  const exportToPDF = useCallback(async (riskData: any) => {
    const loadingToast = toast.loading('Generating risk management report...');
    
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.riskReport);
      await exporter.exportRiskReport(riskData);
      
      toast.success('Risk management report exported successfully!', { id: loadingToast });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export risk report', { id: loadingToast });
    }
  }, []);
  
  const exportToCSV = useCallback((riskData: any) => {
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.riskReport);
      
      if (riskData.positions && riskData.positions.length > 0) {
        exporter.exportToCSV(riskData.positions, 'risk_positions');
        toast.success('Risk positions exported to CSV!');
      } else {
        toast.error('No positions data available for export');
      }
    } catch (error) {
      console.error('CSV export failed:', error);
      toast.error('Failed to export CSV');
    }
  }, []);
  
  return {
    setChartRef,
    exportToPDF,
    exportToCSV
  };
};

export const useModelExport = () => {
  const chartRefs = useRef<{ [key: string]: HTMLElement }>({});
  
  const setChartRef = useCallback((key: string, element: HTMLElement | null) => {
    if (element) {
      chartRefs.current[key] = element;
    }
  }, []);
  
  const exportToPDF = useCallback(async (modelData: any) => {
    const loadingToast = toast.loading('Generating model performance report...');
    
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.modelReport);
      await exporter.exportModelReport(modelData);
      
      toast.success('Model performance report exported successfully!', { id: loadingToast });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export model report', { id: loadingToast });
    }
  }, []);
  
  const exportToCSV = useCallback((modelData: any) => {
    try {
      const exporter = new ProfessionalExportManager(exportConfigs.modelReport);
      
      if (modelData.predictions && modelData.predictions.length > 0) {
        exporter.exportToCSV(modelData.predictions, 'model_predictions');
        toast.success('Model predictions exported to CSV!');
      } else {
        toast.error('No predictions data available for export');
      }
    } catch (error) {
      console.error('CSV export failed:', error);
      toast.error('Failed to export CSV');
    }
  }, []);
  
  return {
    setChartRef,
    exportToPDF,
    exportToCSV
  };
};

export const useCorrelationExport = () => {
  const chartRefs = useRef<{ [key: string]: HTMLElement }>({});
  
  const setChartRef = useCallback((key: string, element: HTMLElement | null) => {
    if (element) {
      chartRefs.current[key] = element;
    }
  }, []);
  
  const exportToPDF = useCallback(async (correlationData: any) => {
    const loadingToast = toast.loading('Generating correlation analysis report...');
    
    try {
      const exporter = new ProfessionalExportManager({
        title: 'L3 Correlation Analysis Report',
        subtitle: 'Feature Correlation & Multicollinearity Analysis',
        author: 'Analytics Pipeline System',
        company: 'Professional Trading Platform',
        watermark: 'CONFIDENTIAL'
      });
      
      const metrics: MetricData[] = [
        { label: 'Features Analyzed', value: correlationData.featureCount || 0 },
        { label: 'High Correlations', value: correlationData.highCorrelations || 0 },
        { label: 'Multicollinearity Issues', value: correlationData.multicollinearityIssues || 0 },
        { label: 'Avg VIF Score', value: correlationData.avgVif?.toFixed(2) || 'N/A' }
      ];
      
      const tables: TableData[] = [
        {
          title: 'High Correlation Pairs',
          headers: ['Feature 1', 'Feature 2', 'Correlation', 'Significance'],
          rows: correlationData.highCorrelationPairs?.map((pair: any) => [
            pair.feature1,
            pair.feature2,
            pair.correlation.toFixed(3),
            pair.pValue < 0.05 ? 'Significant' : 'Not Significant'
          ]) || []
        },
        {
          title: 'VIF Analysis',
          headers: ['Feature', 'VIF Score', 'Status'],
          rows: correlationData.vifAnalysis?.map((item: any) => [
            item.feature,
            item.vif.toFixed(2),
            item.vif > 5 ? 'High Multicollinearity' : 'OK'
          ]) || []
        }
      ];
      
      const charts: ChartData[] = Object.entries(chartRefs.current).map(([key, element]) => ({
        element,
        title: getChartTitle(key),
        width: 170,
        height: 120
      }));
      
      await exporter.exportToPDF(metrics, tables, charts);
      
      toast.success('Correlation analysis report exported successfully!', { id: loadingToast });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export correlation report', { id: loadingToast });
    }
  }, []);
  
  const exportToCSV = useCallback((correlationData: any) => {
    try {
      const exporter = new ProfessionalExportManager({
        title: 'L3 Correlation Analysis',
        company: 'Professional Trading Platform'
      });
      
      if (correlationData.correlationMatrix && correlationData.correlationMatrix.length > 0) {
        exporter.exportToCSV(correlationData.correlationMatrix, 'correlation_matrix');
        toast.success('Correlation matrix exported to CSV!');
      } else {
        toast.error('No correlation data available for export');
      }
    } catch (error) {
      console.error('CSV export failed:', error);
      toast.error('Failed to export CSV');
    }
  }, []);
  
  return {
    setChartRef,
    exportToPDF,
    exportToCSV
  };
};

export const usePipelineExport = () => {
  const exportToPDF = useCallback(async (pipelineData: any) => {
    const loadingToast = toast.loading('Generating pipeline health report...');
    
    try {
      const exporter = new ProfessionalExportManager({
        title: 'Pipeline Health Report',
        subtitle: 'Data Pipeline Status & Performance Analysis',
        author: 'Pipeline Monitoring System',
        company: 'Professional Trading Platform',
        watermark: 'CONFIDENTIAL'
      });
      
      const metrics: MetricData[] = [
        { label: 'Healthy Layers', value: pipelineData.healthyLayers || 0 },
        { label: 'Warning Layers', value: pipelineData.warningLayers || 0 },
        { label: 'Error Layers', value: pipelineData.errorLayers || 0 },
        { label: 'Total Layers', value: pipelineData.totalLayers || 0 },
        { label: 'Overall SLA Compliance', value: `${pipelineData.slaCompliance?.toFixed(1)}%` || 'N/A' },
        { label: 'Avg Processing Time', value: `${pipelineData.avgProcessingTime?.toFixed(1)}s` || 'N/A' }
      ];
      
      const tables: TableData[] = [
        {
          title: 'Layer Status Summary',
          headers: ['Layer', 'Status', 'Last Update', 'Data Age (hours)', 'Error Message'],
          rows: pipelineData.layers?.map((layer: any) => [
            layer.layer,
            layer.status,
            layer.last_update === 'Never' ? 'Never' : new Date(layer.last_update).toLocaleString(),
            layer.data_freshness_hours.toFixed(1),
            layer.error_message || 'None'
          ]) || []
        }
      ];
      
      await exporter.exportToPDF(metrics, tables);
      
      toast.success('Pipeline health report exported successfully!', { id: loadingToast });
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export pipeline report', { id: loadingToast });
    }
  }, []);
  
  const exportToCSV = useCallback((pipelineData: any) => {
    try {
      const exporter = new ProfessionalExportManager({
        title: 'Pipeline Health Data',
        company: 'Professional Trading Platform'
      });
      
      if (pipelineData.layers && pipelineData.layers.length > 0) {
        exporter.exportToCSV(pipelineData.layers, 'pipeline_health');
        toast.success('Pipeline health data exported to CSV!');
      } else {
        toast.error('No pipeline data available for export');
      }
    } catch (error) {
      console.error('CSV export failed:', error);
      toast.error('Failed to export CSV');
    }
  }, []);
  
  return {
    exportToPDF,
    exportToCSV
  };
};

// Utility function to get chart titles
function getChartTitle(key: string): string {
  const titles: { [key: string]: string } = {
    'performance-chart': 'Performance Over Time',
    'returns-chart': 'Monthly Returns',
    'drawdown-chart': 'Drawdown Analysis',
    'correlation-heatmap': 'Correlation Heatmap',
    'risk-metrics': 'Risk Metrics',
    'var-chart': 'Value at Risk',
    'model-accuracy': 'Model Accuracy Trend',
    'feature-importance': 'Feature Importance',
    'drift-analysis': 'Model Drift Analysis',
    'pipeline-health': 'Pipeline Health Status'
  };
  
  return titles[key] || key.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}