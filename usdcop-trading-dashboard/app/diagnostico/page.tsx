'use client';

import { useEffect, useState } from 'react';

export default function DiagnosticoPage() {
  const [apiData, setApiData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [timestamp, setTimestamp] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setTimestamp(new Date().toLocaleString('es-CO'));

      try {
        const response = await fetch('/api/analytics/market-conditions?symbol=USDCOP&days=30&_=' + Date.now());
        const data = await response.json();
        setApiData(data);
      } catch (error) {
        console.error('Error:', error);
        setApiData({ error: String(error) });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div style={{
      fontFamily: 'monospace',
      padding: '40px',
      backgroundColor: '#000',
      color: '#0f0',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '32px', marginBottom: '20px', color: '#0ff' }}>
        🔍 DIAGNÓSTICO - CACHE TEST
      </h1>

      <div style={{ fontSize: '18px', marginBottom: '40px' }}>
        <p>⏰ Timestamp: <strong>{timestamp}</strong></p>
        <p>🔄 Cache Bypass: <strong>?_={Date.now()}</strong></p>
      </div>

      {loading ? (
        <p style={{ fontSize: '24px' }}>⏳ Cargando datos del API...</p>
      ) : (
        <div>
          <h2 style={{ fontSize: '24px', marginBottom: '20px', color: '#ff0' }}>
            📊 VALORES QUE TU NAVEGADOR RECIBIÓ DEL API:
          </h2>

          {apiData?.conditions ? (
            <div style={{ backgroundColor: '#111', padding: '20px', borderRadius: '8px' }}>
              {apiData.conditions.map((condition: any, index: number) => (
                <div
                  key={index}
                  style={{
                    marginBottom: '15px',
                    padding: '15px',
                    backgroundColor: condition.indicator.includes('VIX') ||
                                     condition.indicator.includes('Oil') ||
                                     condition.indicator.includes('Fed')
                      ? '#1a3a1a' : '#222',
                    borderLeft: '4px solid #0f0'
                  }}
                >
                  <div style={{ fontSize: '20px', color: '#0ff' }}>
                    {condition.indicator}
                  </div>
                  <div style={{ fontSize: '32px', fontWeight: 'bold', margin: '10px 0' }}>
                    {condition.value}
                  </div>
                  <div style={{ fontSize: '14px', color: '#999' }}>
                    Change: {condition.change} | Status: {condition.status}
                  </div>
                  <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                    {condition.description}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <pre style={{ color: '#f00' }}>
              {JSON.stringify(apiData, null, 2)}
            </pre>
          )}

          <div style={{
            marginTop: '40px',
            padding: '20px',
            backgroundColor: '#1a1a00',
            border: '2px solid #ff0',
            borderRadius: '8px'
          }}>
            <h3 style={{ color: '#ff0', marginBottom: '15px' }}>
              ✅ VALORES CORRECTOS (Sin hardcodeo):
            </h3>
            <ul style={{ fontSize: '18px', lineHeight: '2' }}>
              <li>VIX Index: ~10-40 (calculado de volatilidad real)</li>
              <li>Oil Price: 0.0 (sin API externa configurada)</li>
              <li>Fed Policy: 0.0 (sin API externa configurada)</li>
              <li>Credit Spreads: ~4-10 (volatilidad × 3)</li>
              <li>EM Sentiment: ~10-30 (calculado de momentum)</li>
            </ul>
          </div>

          <div style={{
            marginTop: '40px',
            padding: '20px',
            backgroundColor: '#1a0000',
            border: '2px solid #f00',
            borderRadius: '8px'
          }}>
            <h3 style={{ color: '#f00', marginBottom: '15px' }}>
              ❌ VALORES VIEJOS (Hardcodeados - YA ELIMINADOS):
            </h3>
            <ul style={{ fontSize: '18px', lineHeight: '2' }}>
              <li>VIX Index: 18.5 (VIEJO - ELIMINADO)</li>
              <li>Oil Price: 84.7 (VIEJO - ELIMINADO)</li>
              <li>Fed Policy: 5.25 (VIEJO - ELIMINADO)</li>
              <li>Credit Spreads: 145 (VIEJO - ELIMINADO)</li>
              <li>EM Sentiment: 42.1 (VIEJO - ELIMINADO)</li>
            </ul>
            <p style={{ marginTop: '20px', fontSize: '16px', color: '#f88' }}>
              Si ves estos valores arriba ⬆️, tu navegador tiene CACHE VIEJO.
            </p>
          </div>

          <div style={{
            marginTop: '40px',
            padding: '20px',
            backgroundColor: '#001a1a',
            border: '2px solid #0ff',
            borderRadius: '8px'
          }}>
            <h3 style={{ color: '#0ff', marginBottom: '15px' }}>
              🔧 CÓMO ELIMINAR EL CACHE:
            </h3>
            <ol style={{ fontSize: '16px', lineHeight: '2' }}>
              <li>Presiona <strong>F12</strong> para abrir DevTools</li>
              <li>Haz clic derecho en el botón de refresh (🔄)</li>
              <li>Selecciona: <strong>"Empty Cache and Hard Reload"</strong></li>
              <li>O ve a: <strong>Application → Clear Storage → Clear site data</strong></li>
            </ol>
          </div>

          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '30px',
              padding: '15px 30px',
              fontSize: '18px',
              backgroundColor: '#0f0',
              color: '#000',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            🔄 RECARGAR DIAGNÓSTICO
          </button>
        </div>
      )}
    </div>
  );
}
