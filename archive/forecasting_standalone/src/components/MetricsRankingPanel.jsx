import React, { useMemo } from 'react';

/**
 * MetricsRankingPanel - Panel de Rankings de Modelos con Métricas de Trading
 *
 * Muestra un ranking de modelos basado en métricas calculadas con
 * Walk-Forward Validation (estándar doctoral para trading sistemático).
 *
 * Métricas mostradas:
 * - Direction Accuracy (DA): Precisión direccional
 * - Sharpe Ratio: Retorno ajustado por riesgo (anualizado)
 * - Profit Factor: Ganancia bruta / Pérdida bruta
 * - Max Drawdown: Caída máxima del equity
 */
const MetricsRankingPanel = ({ data, selectedHorizon, selectedModel }) => {
  const rankings = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Filtrar solo filas de backtest con métricas de trading
    const filtered = data.filter(row =>
      row.view_type === 'backtest' &&
      (selectedHorizon === 'ALL' || row.horizon_days === parseInt(selectedHorizon))
    );

    // Agrupar por modelo y calcular promedios
    const modelMetrics = {};
    filtered.forEach(row => {
      const modelId = row.model_id;
      if (!modelMetrics[modelId]) {
        modelMetrics[modelId] = {
          model_id: modelId,
          da_values: [],
          sharpe_values: [],
          pf_values: [],
          mdd_values: [],
          return_values: []
        };
      }
      const m = modelMetrics[modelId];

      // Usar direction_accuracy del backtest, o wf_direction_accuracy si existe
      const da = row.wf_direction_accuracy || row.direction_accuracy;
      if (da != null && !isNaN(da)) m.da_values.push(parseFloat(da));
      if (row.sharpe != null && !isNaN(row.sharpe)) m.sharpe_values.push(parseFloat(row.sharpe));
      if (row.profit_factor != null && !isNaN(row.profit_factor)) m.pf_values.push(parseFloat(row.profit_factor));
      if (row.max_drawdown != null && !isNaN(row.max_drawdown)) m.mdd_values.push(parseFloat(row.max_drawdown));
      if (row.total_return != null && !isNaN(row.total_return)) m.return_values.push(parseFloat(row.total_return));
    });

    // Calcular promedios
    const avg = arr => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

    return Object.values(modelMetrics)
      .map(m => ({
        model_id: m.model_id,
        da: avg(m.da_values),
        sharpe: avg(m.sharpe_values),
        pf: avg(m.pf_values),
        mdd: avg(m.mdd_values),
        totalReturn: avg(m.return_values)
      }))
      .filter(m => m.da !== null) // Solo modelos con datos
      .sort((a, b) => (b.da || 0) - (a.da || 0)); // Ordenar por DA descendente
  }, [data, selectedHorizon]);

  // Formatear métricas
  const formatMetric = (val, decimals = 2, suffix = '') => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    return val.toFixed(decimals) + suffix;
  };

  // Formatear DA como porcentaje (puede venir como 0.65 o 65)
  const formatDA = (da) => {
    if (da === null || da === undefined || isNaN(da)) return 'N/A';
    // Si es menor a 1, asumir que es decimal (0.65 = 65%)
    const pct = da < 1 ? da * 100 : da;
    return pct.toFixed(1) + '%';
  };

  // Badges para Sharpe
  const getSharpeBadge = (sharpe) => {
    if (sharpe === null || isNaN(sharpe)) return { icon: '', color: '#888' };
    if (sharpe >= 2) return { icon: '★★', color: '#4ecca3' };    // Excelente
    if (sharpe >= 1) return { icon: '★', color: '#4ecca3' };     // Bueno
    if (sharpe >= 0) return { icon: '○', color: '#f0ad4e' };     // Neutral
    return { icon: '✗', color: '#ff6b6b' };                       // Negativo
  };

  // Badges para Profit Factor
  const getPFBadge = (pf) => {
    if (pf === null || isNaN(pf)) return { icon: '', color: '#888' };
    if (pf >= 1.5) return { icon: '★★', color: '#4ecca3' };
    if (pf >= 1.0) return { icon: '★', color: '#4ecca3' };
    return { icon: '✗', color: '#ff6b6b' };
  };

  // Color para Max Drawdown
  const getMDDColor = (mdd) => {
    if (mdd === null || isNaN(mdd)) return '#888';
    if (mdd < 0.1) return '#4ecca3';   // < 10% excelente
    if (mdd < 0.2) return '#f0ad4e';   // < 20% aceptable
    return '#ff6b6b';                   // > 20% riesgoso
  };

  // Formatear nombre del modelo
  const formatModelName = (modelId) => {
    return modelId
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  // Si no hay datos con métricas de trading, mostrar mensaje
  const hasTradeMetrics = rankings.some(m => m.sharpe !== null || m.pf !== null);

  if (rankings.length === 0) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px',
        border: '1px solid #2d3748'
      }}>
        <h3 style={{ color: '#4ecca3', marginBottom: '12px', fontSize: '16px' }}>
          Model Rankings
        </h3>
        <p style={{ color: '#888', fontSize: '14px' }}>
          No hay datos de backtest disponibles para el horizonte seleccionado.
        </p>
      </div>
    );
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
      borderRadius: '12px',
      padding: '20px',
      marginBottom: '20px',
      border: '1px solid #2d3748',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ color: '#4ecca3', margin: 0, fontSize: '16px' }}>
          Model Rankings {selectedHorizon !== 'ALL' ? `(H=${selectedHorizon})` : '(All Horizons)'}
        </h3>
        {hasTradeMetrics && (
          <span style={{
            fontSize: '11px',
            color: '#888',
            background: '#2d3748',
            padding: '4px 8px',
            borderRadius: '4px'
          }}>
            Walk-Forward Validated
          </span>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '13px',
          color: '#e0e0e0'
        }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #4ecca3' }}>
              <th style={{ padding: '10px 8px', textAlign: 'left', fontWeight: '600' }}>#</th>
              <th style={{ padding: '10px 8px', textAlign: 'left', fontWeight: '600' }}>Model</th>
              <th style={{ padding: '10px 8px', textAlign: 'right', fontWeight: '600' }}>DA</th>
              {hasTradeMetrics && (
                <>
                  <th style={{ padding: '10px 8px', textAlign: 'right', fontWeight: '600' }}>Sharpe</th>
                  <th style={{ padding: '10px 8px', textAlign: 'right', fontWeight: '600' }}>PF</th>
                  <th style={{ padding: '10px 8px', textAlign: 'right', fontWeight: '600' }}>MaxDD</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {rankings.map((m, idx) => {
              const isSelected = selectedModel && m.model_id.toLowerCase() === selectedModel.toLowerCase();
              const isFirst = idx === 0;
              const sharpeBadge = getSharpeBadge(m.sharpe);
              const pfBadge = getPFBadge(m.pf);

              return (
                <tr
                  key={m.model_id}
                  style={{
                    borderBottom: '1px solid #2d3748',
                    background: isSelected
                      ? 'rgba(78, 204, 163, 0.2)'
                      : isFirst
                        ? 'rgba(78, 204, 163, 0.08)'
                        : 'transparent',
                    transition: 'background 0.2s'
                  }}
                >
                  <td style={{
                    padding: '10px 8px',
                    color: isFirst ? '#4ecca3' : '#888',
                    fontWeight: isFirst ? 'bold' : 'normal'
                  }}>
                    {idx + 1}
                  </td>
                  <td style={{
                    padding: '10px 8px',
                    fontWeight: isFirst || isSelected ? 'bold' : 'normal',
                    color: isSelected ? '#4ecca3' : '#e0e0e0'
                  }}>
                    {formatModelName(m.model_id)}
                  </td>
                  <td style={{
                    padding: '10px 8px',
                    textAlign: 'right',
                    fontWeight: 'bold',
                    color: m.da && (m.da > 0.55 || m.da > 55) ? '#4ecca3' : '#f0ad4e'
                  }}>
                    {formatDA(m.da)}
                  </td>
                  {hasTradeMetrics && (
                    <>
                      <td style={{ padding: '10px 8px', textAlign: 'right' }}>
                        <span style={{ color: sharpeBadge.color, marginRight: '4px' }}>
                          {sharpeBadge.icon}
                        </span>
                        {formatMetric(m.sharpe)}
                      </td>
                      <td style={{ padding: '10px 8px', textAlign: 'right' }}>
                        <span style={{ color: pfBadge.color, marginRight: '4px' }}>
                          {pfBadge.icon}
                        </span>
                        {formatMetric(m.pf)}
                      </td>
                      <td style={{
                        padding: '10px 8px',
                        textAlign: 'right',
                        color: getMDDColor(m.mdd)
                      }}>
                        {formatMetric(m.mdd ? m.mdd * 100 : null, 1, '%')}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {hasTradeMetrics && (
        <div style={{
          marginTop: '12px',
          padding: '10px',
          background: '#2d3748',
          borderRadius: '6px',
          fontSize: '11px',
          color: '#888'
        }}>
          <div style={{ marginBottom: '4px' }}>
            <strong style={{ color: '#aaa' }}>Sharpe:</strong>{' '}
            <span style={{ color: '#4ecca3' }}>★★</span> ≥2.0 (Excelente) |{' '}
            <span style={{ color: '#4ecca3' }}>★</span> ≥1.0 (Bueno) |{' '}
            <span style={{ color: '#f0ad4e' }}>○</span> ≥0 (Neutral) |{' '}
            <span style={{ color: '#ff6b6b' }}>✗</span> &lt;0
          </div>
          <div>
            <strong style={{ color: '#aaa' }}>PF:</strong>{' '}
            <span style={{ color: '#4ecca3' }}>★★</span> ≥1.5 |{' '}
            <span style={{ color: '#4ecca3' }}>★</span> ≥1.0 |{' '}
            <span style={{ color: '#ff6b6b' }}>✗</span> &lt;1.0 (Pérdida)
            <span style={{ marginLeft: '16px' }}>
              <strong style={{ color: '#aaa' }}>MaxDD:</strong>{' '}
              <span style={{ color: '#4ecca3' }}>&lt;10%</span> |{' '}
              <span style={{ color: '#f0ad4e' }}>&lt;20%</span> |{' '}
              <span style={{ color: '#ff6b6b' }}>&gt;20%</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsRankingPanel;
