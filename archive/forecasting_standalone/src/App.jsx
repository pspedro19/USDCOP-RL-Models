import React, { useState, useEffect, useMemo } from 'react';
import { loadData, filterData, getUniqueValues } from './utils/dataProcessor';
import MetricCard from './components/MetricCard';
import ImageViewer from './components/ImageViewer';
import MetricsRankingPanel from './components/MetricsRankingPanel';
import BacktestMetricsTables from './components/BacktestMetricsTables';
import { LayoutDashboard, Calendar, BarChart2, Filter, Clock, Database, HardDrive, Image } from 'lucide-react';

// Main Dashboard Component
function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dataSource, setDataSource] = useState('loading');
  const [error, setError] = useState(null);

  // State
  const [viewType, setViewType] = useState('forward_forecast');
  const [inferenceWeek, setInferenceWeek] = useState('');
  const [selectedModel, setSelectedModel] = useState('ALL');
  const [horizon, setHorizon] = useState('ALL');
  const [imageType, setImageType] = useState('simple');

  const resultsBaseUrl = useMemo(() => {
    return './';
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const csvUrl = '/bi_dashboard_unified.csv';
        const loadedData = await loadData(csvUrl);
        setData(loadedData);
        setDataSource('csv_local');

        const weeks = getUniqueValues(loadedData, 'inference_week');
        if (weeks.length > 0) setInferenceWeek(weeks[weeks.length - 1]);
      } catch (err) {
        console.error('Failed to load data:', err);
        setError('No se pudieron cargar los datos. Verifica que el archivo CSV este disponible.');
        setDataSource('none');
      }

      setLoading(false);
    };

    fetchData();
  }, []);

  // Filter Logic
  const filteredData = useMemo(() => {
    let res = filterData(data, { viewType, inferenceWeek });

    // Model Filter
    if (selectedModel !== 'ALL') {
      res = res.filter(row => row.model_name === selectedModel);
    }

    // Horizon Filter (Only apply if not ALL)
    if (horizon !== 'ALL') {
      res = res.filter(row => String(row.horizon_days) === String(horizon));
    }

    return res;
  }, [data, viewType, inferenceWeek, selectedModel, horizon]);

  const weeks = useMemo(() => getUniqueValues(data, 'inference_week'), [data]);
  const models = useMemo(() => getUniqueValues(data, 'model_name'), [data]);
  const availableHorizons = useMemo(() => {
    const filteredData = data.filter(row => row.view_type === viewType);
    return getUniqueValues(filteredData, 'horizon_days');
  }, [data, viewType]);

  // Ensemble model variants with display labels
  const ensembleVariants = [
    { value: 'ENSEMBLE_BEST_OF_BREED', label: 'Best of Breed', imageKey: 'best_of_breed' },
    { value: 'ENSEMBLE_TOP_3', label: 'Top 3 Average', imageKey: 'top_3' },
    { value: 'ENSEMBLE_TOP_6_MEAN', label: 'Top 6 Average', imageKey: 'top_6_mean' }
  ];

  // Get display label for model
  const getModelDisplayLabel = (modelName) => {
    if (modelName === 'CONSENSUS') return 'Consensus (All Models Average)';
    if (modelName === 'ENSEMBLE') return 'Ensemble (Weighted)';
    const ensembleVar = ensembleVariants.find(v => v.value === modelName);
    if (ensembleVar) return ensembleVar.label;
    return modelName;
  };

  // Check if model is an ensemble variant
  const isEnsembleVariant = (modelName) => {
    return ensembleVariants.some(v => v.value === modelName);
  };

  // Get ensemble image key for the selected variant
  const getEnsembleImageKey = (modelName) => {
    const variant = ensembleVariants.find(v => v.value === modelName);
    return variant ? variant.imageKey : null;
  };

  const formatNum = (val, decimals = 2) => {
    if (val === null || val === undefined || isNaN(Number(val))) return 'N/A';
    return Number(val).toFixed(decimals);
  };

  // Helper function to get image prefix based on imageType
  const getImagePrefix = () => {
    switch (imageType) {
      case 'complete':
        return 'complete_forecast_';
      case 'fan':
        return 'fan_chart_';
      case 'simple':
      default:
        return 'forward_forecast_';
    }
  };

  // Helper function to get image type label for captions
  const getImageTypeLabel = () => {
    switch (imageType) {
      case 'complete':
        return 'Complete Forecast';
      case 'fan':
        return 'Fan Chart';
      case 'simple':
      default:
        return 'Simple Forecast';
    }
  };

  const renderContent = () => {
    let imageSrc = null;
    let imageCaption = "";
    let metrics = [];

    // Case 1: Consolidated Forecast (All Models) - Use consensus image
    if (viewType === 'forward_forecast' && selectedModel === 'ALL') {
      imageSrc = `forward_forecast_consensus_all_models.png`;
      imageCaption = "Consensus Forecast (All Models + Average)";

      if (filteredData.length > 0) {
        const row = filteredData[0];
        metrics = [
          { title: "Avg Direction Accuracy", value: `${formatNum(row.model_avg_direction_accuracy)}%` },
          { title: "Avg RMSE", value: formatNum(row.model_avg_rmse, 4) }
        ];
      }
    }
    // Case 2: Ensemble Variant Selected
    else if (isEnsembleVariant(selectedModel)) {
      const ensembleImageKey = getEnsembleImageKey(selectedModel);
      const displayLabel = getModelDisplayLabel(selectedModel);

      if (data.length > 0) {
        const sampleImg = data.find(d => d.image_forecast && d.image_forecast.includes('forward_forecast'))?.image_forecast;
        if (sampleImg) {
          const dirPart = sampleImg.substring(0, sampleImg.lastIndexOf('forward_forecast_'));
          imageSrc = `${dirPart}forward_forecast_ensemble_${ensembleImageKey}.png`;
        } else {
          imageSrc = `forward_forecast_ensemble_${ensembleImageKey}.png`;
        }
      } else {
        imageSrc = `forward_forecast_ensemble_${ensembleImageKey}.png`;
      }
      imageCaption = `${displayLabel} Ensemble Forecast`;

      metrics = [
        { title: "Ensemble Method", value: displayLabel },
        { title: "Type", value: ensembleImageKey === 'best_of_breed' ? 'Best model per horizon' :
                                ensembleImageKey === 'top_3' ? 'Average of Top 3 models' :
                                'Average of Top 6 models' }
      ];
    }
    // Case 3: Specific Model Selected
    else {
      if (filteredData.length === 1) {
        const row = filteredData[0];

        if (viewType === 'backtest') {
          imageSrc = row.image_backtest;
          imageCaption = `${row.model_name} Backtest (H=${row.horizon_days})`;
          metrics = [
            { title: "Direction Accuracy", value: `${formatNum(row.direction_accuracy)}%` },
            { title: "RMSE", value: formatNum(row.rmse, 4) },
            { title: "MAE", value: formatNum(row.mae, 4) },
            { title: "R2 Score", value: formatNum(row.r2, 4) }
          ];
        } else {
          const baseImage = row.image_forecast;
          if (baseImage && baseImage.includes('forward_forecast_')) {
            imageSrc = baseImage.replace('forward_forecast_', getImagePrefix());
          } else {
            imageSrc = baseImage;
          }
          imageCaption = `${row.model_name} ${getImageTypeLabel()} (H=${row.horizon_days})`;
          metrics = [
            { title: "Predicted Price", value: `$${formatNum(row.predicted_price, 0)}` },
            { title: "Signal", value: row.signal, color: row.signal === 'BUY' ? 'var(--success)' : row.signal === 'SELL' ? 'var(--danger)' : 'var(--text-secondary)' },
            { title: "Change %", value: `${formatNum(row.price_change_pct)}%` }
          ];
        }
      }
      else if (filteredData.length > 1) {
        const row = filteredData[0];

        if (viewType === 'backtest') {
          imageSrc = row.image_backtest;
          imageCaption = `${row.model_name} (Overview - H=${row.horizon_days})`;
        } else {
          const baseImage = row.image_forecast;
          if (baseImage && baseImage.includes('forward_forecast_')) {
            imageSrc = baseImage.replace('forward_forecast_', getImagePrefix());
          } else {
            imageSrc = baseImage;
          }
          imageCaption = `${row.model_name} ${getImageTypeLabel()} (Overview - H=${row.horizon_days})`;
        }

        metrics = [
          { title: "Combined Avg DA", value: `${formatNum(row.model_avg_direction_accuracy)}%` },
          { title: "Combined Avg RMSE", value: formatNum(row.model_avg_rmse, 4) }
        ];
      } else {
        return <div className="loading">No data found for this selection.</div>;
      }
    }

    return (
      <div className={`content-wrapper ${viewType === 'backtest' ? 'backtest-view' : ''}`}>
        {/* Model Rankings Panel - Only shown in Backtest view */}
        {viewType === 'backtest' && (
          <MetricsRankingPanel
            data={data}
            selectedHorizon={horizon}
            selectedModel={selectedModel}
          />
        )}

        {/* 1. Visualization Area (Takes Max Space) */}
        <div className="visual-section">
          {imageSrc ? (
            <ImageViewer
              src={imageSrc}
              baseUrl={resultsBaseUrl}
              caption={imageCaption}
            />
          ) : (
            <div className="loading">Select a Model and Horizon to view chart</div>
          )}
        </div>

        {/* 2. Metrics Bar (Bottom) */}
        <div className="metrics-bar">
          {metrics.map((m, idx) => (
            <div key={idx} className="metric-card">
              <div className="metric-title">{m.title}</div>
              <div className="metric-value" style={{ color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>

        {/* 3. Metrics Tables - Only in Backtest view */}
        {viewType === 'backtest' && (
          <BacktestMetricsTables />
        )}
      </div>
    );
  };

  if (loading) return <div className="loading">Cargando datos del Dashboard...</div>;

  if (error) return (
    <div className="loading" style={{ color: '#ff6b6b' }}>
      <p>{error}</p>
    </div>
  );

  return (
    <div className="dashboard-layout">
      {/* SIDEBAR CONTROLS */}
      <aside className="sidebar">
        <h1>USD/COP AI</h1>
        <p className="subtitle">Analytics Console</p>

        {/* Data Source Indicator */}
        <div className="data-source-indicator" style={{
          padding: '8px 12px',
          margin: '10px 0',
          borderRadius: '6px',
          fontSize: '0.75em',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          backgroundColor: 'rgba(108, 117, 125, 0.2)',
          color: '#6c757d'
        }}>
          <HardDrive size={14} />
          <span>CSV Local</span>
        </div>

        <div className="control-group">
          <label className="control-label"><LayoutDashboard size={14} style={{ display: 'inline', marginRight: 8 }} />View Mode</label>
          <select value={viewType} onChange={(e) => {
            setViewType(e.target.value);
            setHorizon('ALL');
          }}>
            <option value="forward_forecast">Forward Forecast</option>
            <option value="backtest">Backtest Analysis</option>
          </select>
        </div>

        {viewType === 'forward_forecast' && (
          <div className="control-group">
            <label className="control-label"><Calendar size={14} style={{ display: 'inline', marginRight: 8 }} />Inference Week</label>
            <select value={inferenceWeek} onChange={(e) => setInferenceWeek(e.target.value)}>
              {weeks.map(w => (
                <option key={w} value={w}>Week {w}</option>
              ))}
            </select>
          </div>
        )}

        <div className="control-group">
          <label className="control-label"><Filter size={14} style={{ display: 'inline', marginRight: 8 }} />Model</label>
          <select value={selectedModel} onChange={(e) => {
            setSelectedModel(e.target.value);
            if (e.target.value === 'ALL') setHorizon('ALL');
          }}>
            <option value="ALL">All Models (Consensus)</option>
            {models.filter(m => !m.includes('ENSEMBLE') && m !== 'CONSENSUS').map(m => (
              <option key={m} value={m}>{getModelDisplayLabel(m)}</option>
            ))}
            <option disabled>─── Ensembles ───</option>
            {ensembleVariants.map(v => (
              <option key={v.value} value={v.value}>{v.label}</option>
            ))}
            {models.includes('CONSENSUS') && (
              <option value="CONSENSUS">{getModelDisplayLabel('CONSENSUS')}</option>
            )}
          </select>
        </div>

        {/* Horizon Filter - Only for Backtest view when a model is selected */}
        {selectedModel !== 'ALL' && viewType === 'backtest' && (
          <div className="control-group">
            <label className="control-label"><Clock size={14} style={{ display: 'inline', marginRight: 8 }} />Horizon</label>
            <select value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              <option value="ALL">Overview (Avg)</option>
              {availableHorizons.map(h => (
                <option key={h} value={h}>H={h} Days</option>
              ))}
            </select>
          </div>
        )}
      </aside>

      <main className="main-content">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
