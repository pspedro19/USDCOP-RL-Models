'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Activity,
  Target,
  Clock,
  Zap,
  Shield,
  Bell,
  RefreshCw,
  Database,
  Brain,
  AlertCircle
} from 'lucide-react';
import SignalAlerts from './SignalAlerts';

interface TradingSignal {
  id: string;
  timestamp: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string[];
  riskScore: number;
  expectedReturn: number;
  timeHorizon: string;
  modelSource: string;
  latency: number;
}

interface SignalPerformance {
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  totalSignals: number;
  activeSignals: number;
}

export default function TradingSignals() {
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [performance, setPerformance] = useState<SignalPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [minConfidence, setMinConfidence] = useState(70);
  const [showAlerts, setShowAlerts] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [dataSource, setDataSource] = useState<string>('');

  useEffect(() => {
    loadSignals();
    const interval = setInterval(loadSignals, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadSignals = async () => {
    try {
      setError(null);

      // Try multi-strategy signals endpoint first (real data)
      const response = await fetch('/api/trading/signals/multi-strategy');

      if (response.ok) {
        const data = await response.json();
        const apiSignals = data.signals || data.data?.signals || [];

        // Transform API response to component format
        const transformedSignals: TradingSignal[] = apiSignals.map((s: any, idx: number) => ({
          id: s.signal_id || s.id || `sig_${Date.now()}_${idx}`,
          timestamp: s.timestamp || new Date().toISOString(),
          type: (s.signal || s.type || 'HOLD').toUpperCase() as 'BUY' | 'SELL' | 'HOLD',
          confidence: (s.confidence ?? s.probability ?? 0.7) * 100,
          price: s.price ?? s.current_price ?? 0,
          stopLoss: s.stop_loss ?? s.stopLoss,
          takeProfit: s.take_profit ?? s.takeProfit,
          reasoning: s.reasoning || s.factors || ['Signal from ML model'],
          riskScore: s.risk_score ?? s.riskScore ?? 5,
          expectedReturn: s.expected_return ?? s.expectedReturn ?? 0,
          timeHorizon: s.time_horizon ?? s.timeHorizon ?? '15-30 min',
          modelSource: s.model_id ?? s.model ?? s.strategy ?? 'ML_Model',
          latency: s.latency ?? s.processing_time ?? 50
        }));

        setSignals(transformedSignals);
        setDataSource(data.source || 'Multi-Strategy API');
        setLastUpdate(data.timestamp || new Date().toISOString());

        // Calculate performance from signals if not provided
        if (data.performance) {
          setPerformance(data.performance);
        } else if (transformedSignals.length > 0) {
          // Fetch performance from dedicated endpoint
          try {
            const perfResponse = await fetch('/api/trading/performance/multi-strategy');
            if (perfResponse.ok) {
              const perfData = await perfResponse.json();
              const strategies = perfData.strategies || [];
              if (strategies.length > 0) {
                const aggregated = strategies.reduce((acc: any, s: any) => ({
                  winRate: acc.winRate + (s.win_rate || 0),
                  avgWin: acc.avgWin + (s.avg_win || 0),
                  avgLoss: acc.avgLoss + (s.avg_loss || 0),
                  profitFactor: acc.profitFactor + (s.profit_factor || 0),
                  sharpeRatio: acc.sharpeRatio + (s.sharpe || s.sharpe_ratio || 0),
                  totalSignals: acc.totalSignals + (s.total_trades || 0),
                  activeSignals: acc.activeSignals + (s.active_positions || 0)
                }), {
                  winRate: 0, avgWin: 0, avgLoss: 0, profitFactor: 0,
                  sharpeRatio: 0, totalSignals: 0, activeSignals: 0
                });

                const count = strategies.length;
                setPerformance({
                  winRate: aggregated.winRate / count,
                  avgWin: aggregated.avgWin / count,
                  avgLoss: aggregated.avgLoss / count,
                  profitFactor: aggregated.profitFactor / count,
                  sharpeRatio: aggregated.sharpeRatio / count,
                  totalSignals: aggregated.totalSignals,
                  activeSignals: transformedSignals.length
                });
              }
            }
          } catch {
            // Performance endpoint failed, use defaults
            setPerformance(null);
          }
        }
      } else {
        throw new Error(`API returned status ${response.status}`);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error loading signals';
      console.error('Error loading signals:', err);
      setError(errorMessage);
      setSignals([]);
      setPerformance(null);
      setLastUpdate(new Date().toISOString());
      setDataSource('ERROR');
    } finally {
      setLoading(false);
    }
  };

  const filteredSignals = signals.filter(signal => {
    if (filter !== 'ALL' && signal.type !== filter) return false;
    if (signal.confidence < minConfidence) return false;
    return true;
  });

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'BUY': return <TrendingUp className="h-5 w-5 text-green-500" />;
      case 'SELL': return <TrendingDown className="h-5 w-5 text-red-500" />;
      case 'HOLD': return <Activity className="h-5 w-5 text-yellow-500" />;
      default: return <AlertTriangle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 85) return 'text-green-600 bg-green-100';
    if (confidence >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getRiskColor = (risk: number) => {
    if (risk <= 2) return 'bg-green-500';
    if (risk <= 4) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Error Banner */}
      {error && (
        <Card className="border-red-500/50 bg-red-950/20">
          <CardContent className="py-6">
            <div className="flex items-start gap-4">
              <AlertCircle className="h-6 w-6 text-red-500 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-red-400 font-semibold mb-1">Error Loading Trading Signals</h3>
                <p className="text-red-300 text-sm mb-3">{error}</p>
                <Button
                  onClick={loadSignals}
                  variant="outline"
                  size="sm"
                  className="border-red-500/50 text-red-400 hover:bg-red-950/40"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Status Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-cyan-500" />
              <span>ML-Powered Trading Signals</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={dataSource.includes('L5') ? "default" : dataSource.includes('Mock') ? "secondary" : "destructive"}>
                <Database className="h-3 w-3 mr-1" />
                {dataSource}
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <RefreshCw className="h-3 w-3" />
                {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'Never'}
              </Badge>
              <button
                onClick={() => setShowAlerts(!showAlerts)}
                className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
                  showAlerts ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'
                }`}
              >
                <Bell className="h-3 w-3" />
                Alerts
              </button>
            </div>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Alert Panel */}
      {showAlerts && (
        <Card>
          <CardHeader>
            <CardTitle>Signal Alerts & Notifications</CardTitle>
          </CardHeader>
          <CardContent>
            <SignalAlerts />
          </CardContent>
        </Card>
      )}

      {/* Performance Summary */}
      {performance && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Signal Performance</span>
              <Badge variant="outline">Live</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div>
                <p className="text-sm text-gray-500">Win Rate</p>
                <p className="text-2xl font-bold text-green-600">
                  {performance.winRate.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Profit Factor</p>
                <p className="text-2xl font-bold">
                  {performance.profitFactor.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Sharpe Ratio</p>
                <p className="text-2xl font-bold">
                  {performance.sharpeRatio.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Win</p>
                <p className="text-2xl font-bold text-green-600">
                  ${performance.avgWin.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Loss</p>
                <p className="text-2xl font-bold text-red-600">
                  ${Math.abs(performance.avgLoss).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Active</p>
                <p className="text-2xl font-bold">
                  {performance.activeSignals}/{performance.totalSignals}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex gap-2">
              {(['ALL', 'BUY', 'SELL', 'HOLD'] as const).map(type => (
                <button
                  key={type}
                  onClick={() => setFilter(type)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    filter === type 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Min Confidence:</span>
              <input
                type="range"
                min="50"
                max="100"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-sm font-medium">{minConfidence}%</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Active Signals */}
      <div className="space-y-4">
        {loading ? (
          <Card>
            <CardContent className="py-12 text-center">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
              <p className="text-gray-400 text-lg">Loading Trading Signals...</p>
              <p className="text-gray-500 text-sm mt-2">Fetching latest market signals</p>
            </CardContent>
          </Card>
        ) : filteredSignals.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <Brain className="h-12 w-12 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400 text-lg mb-2">No Signals Available</p>
              <p className="text-gray-500 text-sm">
                {error ? 'Unable to load signals' : 'No signals matching current criteria'}
              </p>
            </CardContent>
          </Card>
        ) : (
          filteredSignals.map(signal => (
            <Card key={signal.id} className="hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {getSignalIcon(signal.type)}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-bold">{signal.type}</span>
                        <Badge className={getConfidenceColor(signal.confidence)}>
                          {signal.confidence.toFixed(1)}% confidence
                        </Badge>
                        <Badge variant="outline">
                          <Zap className="h-3 w-3 mr-1" />
                          {signal.latency}ms
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-500">
                        {new Date(signal.timestamp).toLocaleTimeString()} · {signal.modelSource}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold">${signal.price.toFixed(2)}</p>
                    <p className="text-sm text-gray-500">{signal.timeHorizon}</p>
                  </div>
                </div>

                {/* Signal Details */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  {signal.stopLoss && (
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-red-500" />
                      <span className="text-sm">
                        Stop Loss: <span className="font-medium">${signal.stopLoss.toFixed(2)}</span>
                      </span>
                    </div>
                  )}
                  {signal.takeProfit && (
                    <div className="flex items-center gap-2">
                      <Target className="h-4 w-4 text-green-500" />
                      <span className="text-sm">
                        Take Profit: <span className="font-medium">${signal.takeProfit.toFixed(2)}</span>
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-blue-500" />
                    <span className="text-sm">
                      Expected Return: <span className="font-medium">{(signal.expectedReturn * 100).toFixed(2)}%</span>
                    </span>
                  </div>
                </div>

                {/* Risk Score */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Risk Score</span>
                    <span className="font-medium">{signal.riskScore.toFixed(1)}/10</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${getRiskColor(signal.riskScore)}`}
                      style={{ width: `${signal.riskScore * 10}%` }}
                    />
                  </div>
                </div>

                {/* Reasoning */}
                <div className="border-t pt-4">
                  <p className="text-sm font-medium text-gray-700 mb-2">Signal Reasoning:</p>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {signal.reasoning.map((reason, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-blue-500 mt-0.5">•</span>
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}