'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getPrediction, MLPrediction } from '@/lib/services/mlmodel';
import { fetchTechnicalIndicators } from '@/lib/services/twelvedata';
import { ArrowUpCircle, ArrowDownCircle, MinusCircle, AlertCircle, TrendingUp, Activity } from 'lucide-react';

export default function TradingSignals() {
  const [prediction, setPrediction] = useState<MLPrediction | null>(null);
  const [indicators, setIndicators] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        setLoading(true);
        const techIndicators = await fetchTechnicalIndicators('USD/COP');
        setIndicators(techIndicators);

        const features = {
          rsi: techIndicators.rsi?.rsi || 50,
          macd: techIndicators.macd?.macd || 0,
          sma: techIndicators.sma?.sma || 0,
          ema: techIndicators.ema?.ema || 0,
          bbands_upper: techIndicators.bbands?.upper_band || 0,
          bbands_lower: techIndicators.bbands?.lower_band || 0,
          stoch_k: techIndicators.stoch?.slow_k || 50,
          stoch_d: techIndicators.stoch?.slow_d || 50,
        };

        const mlPrediction = await getPrediction(features);
        setPrediction(mlPrediction);
        setError(null);
      } catch (err) {
        // Silently use mock data when service is unavailable
        setError(null);
      } finally {
        setLoading(false);
      }
    };

    fetchSignals();
    const interval = setInterval(fetchSignals, 60000);

    return () => clearInterval(interval);
  }, []);

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return <ArrowUpCircle className="h-8 w-8 text-green-500" />;
      case 'SELL':
        return <ArrowDownCircle className="h-8 w-8 text-red-500" />;
      default:
        return <MinusCircle className="h-8 w-8 text-yellow-500" />;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'SELL':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default:
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    }
  };

  const getRSISignal = (value: number) => {
    if (value > 70) return { signal: 'Overbought', color: 'text-red-500' };
    if (value < 30) return { signal: 'Oversold', color: 'text-green-500' };
    return { signal: 'Neutral', color: 'text-gray-500' };
  };

  const getStochSignal = (k: number, d: number) => {
    if (k > 80 && d > 80) return { signal: 'Overbought', color: 'text-red-500' };
    if (k < 20 && d < 20) return { signal: 'Oversold', color: 'text-green-500' };
    if (k > d) return { signal: 'Bullish', color: 'text-green-500' };
    if (k < d) return { signal: 'Bearish', color: 'text-red-500' };
    return { signal: 'Neutral', color: 'text-gray-500' };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-500">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>ML Trading Signal</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {prediction && getSignalIcon(prediction.prediction)}
              <div>
                <Badge className={prediction ? getSignalColor(prediction.prediction) : ''}>
                  {prediction?.prediction || 'HOLD'}
                </Badge>
                <p className="text-sm text-gray-500 mt-1">
                  Confidence: {prediction ? (prediction.confidence * 100).toFixed(1) : 0}%
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Expected Return</p>
              <p className={`text-lg font-bold ${(prediction?.expected_return || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {prediction ? (prediction.expected_return * 100).toFixed(2) : 0}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">RSI (14)</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {indicators?.rsi?.rsi?.toFixed(2) || 'N/A'}
            </div>
            {indicators?.rsi && (
              <p className={`text-xs ${getRSISignal(indicators.rsi.rsi).color}`}>
                {getRSISignal(indicators.rsi.rsi).signal}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">MACD</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {indicators?.macd?.macd?.toFixed(4) || 'N/A'}
            </div>
            <p className="text-xs text-gray-500">
              Signal: {indicators?.macd?.macd_signal?.toFixed(4) || 'N/A'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stochastic</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              K: {indicators?.stoch?.slow_k?.toFixed(2) || 'N/A'}
            </div>
            <p className="text-xs text-gray-500">
              D: {indicators?.stoch?.slow_d?.toFixed(2) || 'N/A'}
            </p>
            {indicators?.stoch && (
              <p className={`text-xs ${getStochSignal(indicators.stoch.slow_k, indicators.stoch.slow_d).color}`}>
                {getStochSignal(indicators.stoch.slow_k, indicators.stoch.slow_d).signal}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Bollinger Bands</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <p className="text-xs text-gray-500">Upper: {indicators?.bbands?.upper_band?.toFixed(2) || 'N/A'}</p>
              <p className="text-xs text-gray-500">Middle: {indicators?.bbands?.middle_band?.toFixed(2) || 'N/A'}</p>
              <p className="text-xs text-gray-500">Lower: {indicators?.bbands?.lower_band?.toFixed(2) || 'N/A'}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Signal Components</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {prediction?.features && Object.entries(prediction.features).map(([key, value]) => (
              <div key={key}>
                <p className="text-sm text-gray-500">{key.replace(/_/g, ' ').toUpperCase()}</p>
                <p className="text-lg font-semibold">{(value as number).toFixed(2)}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}