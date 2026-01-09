import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, Activity } from 'lucide-react';

const MetricTable = ({ metric, data, isExpanded, onToggle }) => {
  const columns = data.columns;
  const rows = data.data;

  const getCellColor = (value, colName, higherIsBetter) => {
    if (colName === 'model') return '';

    const numValue = parseFloat(value);
    if (isNaN(numValue)) return '';

    // For Sharpe Ratio
    if (metric === 'sharpe_ratio') {
      if (numValue >= 2) return '#10b981'; // green
      if (numValue >= 1) return '#34d399'; // light green
      if (numValue >= 0) return '#fbbf24'; // yellow
      return '#ef4444'; // red
    }

    // For Profit Factor
    if (metric === 'profit_factor') {
      if (numValue >= 1.3) return '#10b981';
      if (numValue >= 1.0) return '#34d399';
      return '#ef4444';
    }

    // For Direction Accuracy
    if (metric === 'direction_accuracy') {
      if (numValue >= 55) return '#10b981';
      if (numValue >= 50) return '#34d399';
      if (numValue >= 45) return '#fbbf24';
      return '#ef4444';
    }

    // For Max Drawdown (lower is better)
    if (metric === 'max_drawdown') {
      if (numValue <= 50) return '#10b981';
      if (numValue <= 80) return '#fbbf24';
      return '#ef4444';
    }

    // For Total Return
    if (metric === 'total_return') {
      if (numValue > 10) return '#10b981';
      if (numValue > 0) return '#34d399';
      return '#ef4444';
    }

    return '';
  };

  const formatValue = (value, colName) => {
    if (colName === 'model') return value.toUpperCase().replace('_', ' ');
    const num = parseFloat(value);
    if (isNaN(num)) return value;
    if (metric === 'total_return' && Math.abs(num) >= 100) return num.toFixed(0);
    if (metric === 'direction_accuracy' || metric === 'max_drawdown') return num.toFixed(1);
    return num.toFixed(2);
  };

  const getIcon = () => {
    switch(metric) {
      case 'direction_accuracy': return <Activity size={16} />;
      case 'sharpe_ratio': return <TrendingUp size={16} />;
      case 'profit_factor': return <TrendingUp size={16} />;
      case 'max_drawdown': return <TrendingDown size={16} />;
      case 'total_return': return <TrendingUp size={16} />;
      default: return <Activity size={16} />;
    }
  };

  return (
    <div className="metric-table-container">
      <div className="metric-table-header" onClick={onToggle}>
        <div className="metric-table-title">
          {getIcon()}
          <span>{data.title}</span>
        </div>
        <div className="metric-table-toggle">
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {isExpanded && (
        <div className="metric-table-content">
          <p className="metric-description">{data.description}</p>
          <div className="table-scroll">
            <table className="metrics-pivot-table">
              <thead>
                <tr>
                  <th>Modelo</th>
                  {columns.map(col => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={idx} className={idx === 0 ? 'best-row' : ''}>
                    <td className="model-cell">{formatValue(row.model, 'model')}</td>
                    {columns.map(col => (
                      <td
                        key={col}
                        style={{ color: getCellColor(row[col], col, data.higher_is_better) }}
                      >
                        {formatValue(row[col], col)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

const BacktestMetricsTables = () => {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedMetrics, setExpandedMetrics] = useState({
    direction_accuracy: true,
    sharpe_ratio: true,
    profit_factor: false,
    max_drawdown: false,
    total_return: false
  });

  useEffect(() => {
    fetch('/backtest_metrics_summary.json')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load metrics');
        return res.json();
      })
      .then(data => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const toggleMetric = (metric) => {
    setExpandedMetrics(prev => ({
      ...prev,
      [metric]: !prev[metric]
    }));
  };

  const expandAll = () => {
    setExpandedMetrics({
      direction_accuracy: true,
      sharpe_ratio: true,
      profit_factor: true,
      max_drawdown: true,
      total_return: true
    });
  };

  const collapseAll = () => {
    setExpandedMetrics({
      direction_accuracy: false,
      sharpe_ratio: false,
      profit_factor: false,
      max_drawdown: false,
      total_return: false
    });
  };

  if (loading) return <div className="metrics-loading">Cargando métricas...</div>;
  if (error) return <div className="metrics-error">Error: {error}</div>;
  if (!metricsData) return null;

  const metricOrder = ['direction_accuracy', 'sharpe_ratio', 'profit_factor', 'max_drawdown', 'total_return'];

  return (
    <div className="backtest-metrics-tables">
      <div className="metrics-header">
        <h3>Métricas Walk-Forward por Modelo y Horizonte</h3>
        <div className="metrics-actions">
          <button onClick={expandAll}>Expandir Todo</button>
          <button onClick={collapseAll}>Colapsar Todo</button>
        </div>
      </div>
      <div className="metrics-info">
        <span>Run: {metricsData.run_id}</span>
        <span>Ventanas: {metricsData.walk_forward_windows}</span>
      </div>
      <div className="metrics-tables-list">
        {metricOrder.map(metric => (
          <MetricTable
            key={metric}
            metric={metric}
            data={metricsData.metrics[metric]}
            isExpanded={expandedMetrics[metric]}
            onToggle={() => toggleMetric(metric)}
          />
        ))}
      </div>
    </div>
  );
};

export default BacktestMetricsTables;
