import { useState, useEffect } from 'react';
import { useRealTimePrice } from '@/hooks/useRealTimePrice';
import { MarketDataService } from '@/lib/services/market-data-service';

interface MarketHealthStatus {
  status: 'healthy' | 'degraded' | 'down'
  latency: number
  lastCheck: string
  services: Record<string, boolean>
}

export const useRealTimeMarketData = () => {
  const realTimePrice = useRealTimePrice('USDCOP');
  const [marketHealth, setMarketHealth] = useState<MarketHealthStatus | null>(null);
  const [additionalData, setAdditionalData] = useState({
    volume: 0,
    high24h: 0,
    low24h: 0,
    spread: 0,
    vwapError: 0,
    pegRate: 0,
    orderFlow: {
      buy: 0,
      sell: 0,
      imbalance: 0
    },
    technicals: {
      rsi: 0,
      macd: 0,
      bollinger: {
        upper: 0,
        middle: 0,
        lower: 0
      },
      ema20: 0,
      ema50: 0,
      ema200: 0
    }
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await MarketDataService.checkAPIHealth();
        setMarketHealth(health);
      } catch (error) {
        console.error('Failed to check market health:', error);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchAdditionalData = async () => {
      try {
        const stats = await MarketDataService.getSymbolStats('USDCOP');
        if (stats) {
          setAdditionalData(prev => ({
            ...prev,
            volume: stats.volume_24h || prev.volume,
            high24h: stats.high_24h || prev.high24h,
            low24h: stats.low_24h || prev.low24h,
            spread: stats.spread || prev.spread
          }));
        }

        const candlestickData = await MarketDataService.getCandlestickData(
          'USDCOP',
          '5m',
          undefined,
          undefined,
          1,
          true
        );

        if (candlestickData?.data && candlestickData.data.length > 0) {
          const latestCandle = candlestickData.data[candlestickData.data.length - 1];
          const indicators = latestCandle.indicators;

          if (indicators) {
            setAdditionalData(prev => ({
              ...prev,
              technicals: {
                rsi: indicators.rsi || 0,
                macd: indicators.macd || 0,
                bollinger: {
                  upper: indicators.bb_upper || 0,
                  middle: indicators.bb_middle || 0,
                  lower: indicators.bb_lower || 0
                },
                ema20: indicators.ema_20 || 0,
                ema50: indicators.ema_50 || 0,
                ema200: indicators.ema_200 || 0
              }
            }));
          }
        }

        const rlMetricsResponse = await fetch(`/api/analytics/rl-metrics?symbol=USDCOP&days=30`);

        if (rlMetricsResponse.ok) {
          const rlMetrics = await rlMetricsResponse.json();
          setAdditionalData(prev => ({
            ...prev,
            vwapError: rlMetrics.metrics?.vwapError || 0,
            pegRate: rlMetrics.metrics?.pegRate || 0
          }));
        }

        const orderFlowResponse = await fetch(`/api/analytics/order-flow?symbol=USDCOP&window=60`);

        if (orderFlowResponse.ok) {
          const orderFlowData = await orderFlowResponse.json();
          if (orderFlowData.data_available) {
            setAdditionalData(prev => ({
              ...prev,
              orderFlow: {
                buy: orderFlowData.order_flow.buy_percent || 0,
                sell: orderFlowData.order_flow.sell_percent || 0,
                imbalance: orderFlowData.order_flow.imbalance || 0
              }
            }));
          }
        }
      } catch (error) {
        console.error('Failed to fetch additional market data:', error);
      }
    };

    fetchAdditionalData();
    const interval = setInterval(fetchAdditionalData, 10000);

    return () => clearInterval(interval);
  }, []);

  return {
    price: realTimePrice.currentPrice?.price || 0,
    change: realTimePrice.priceChange || 0,
    changePercent: realTimePrice.priceChangePercent || 0,
    volume: additionalData.volume || realTimePrice.currentPrice?.volume || 0,
    high24h: additionalData.high24h,
    low24h: additionalData.low24h,
    spread: additionalData.spread,
    vwapError: additionalData.vwapError,
    pegRate: additionalData.pegRate,
    timestamp: realTimePrice.currentPrice ? new Date(realTimePrice.currentPrice.timestamp) : new Date(),
    orderFlow: additionalData.orderFlow,
    technicals: additionalData.technicals,
    isConnected: realTimePrice.isConnected,
    source: realTimePrice.currentPrice?.source || 'unknown',
    marketHealth
  };
};
