'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  AlertTriangle, Shield, Bell, BellRing, BellOff, CheckCircle, XCircle, Clock,
  TrendingUp, TrendingDown, Activity, Zap, Target, BarChart3, Settings,
  Filter, Search, Download, RefreshCw, Volume2, VolumeX, Pause, Play,
  AlertOctagon, Info, ArrowUpRight, ArrowDownRight, Eye, EyeOff
} from 'lucide-react';
import { realTimeRiskEngine, RiskAlert } from '@/lib/services/real-time-risk-engine';

// Custom date formatting functions
const formatDate = (date: Date, pattern: string): string => {
  const months = [
    'ene', 'feb', 'mar', 'abr', 'may', 'jun',
    'jul', 'ago', 'sep', 'oct', 'nov', 'dic'
  ];
  
  const monthsFull = [
    'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
  ];
  
  const day = date.getDate();
  const month = date.getMonth();
  const year = date.getFullYear();
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const seconds = date.getSeconds();
  const pad = (num: number) => num.toString().padStart(2, '0');
  
  switch (pattern) {
    case 'HH:mm:ss':
      return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    case 'PPpp':
      return `${day} de ${monthsFull[month]} de ${year} a las ${pad(hours)}:${pad(minutes)}`;
    default:
      return date.toLocaleDateString('es-ES');
  }
};

const formatDistanceToNow = (date: Date, options?: { addSuffix?: boolean }): string => {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  let result = '';
  if (diffMinutes < 1) {
    result = 'menos de un minuto';
  } else if (diffMinutes < 60) {
    result = `${diffMinutes} minuto${diffMinutes > 1 ? 's' : ''}`;
  } else if (diffHours < 24) {
    result = `${diffHours} hora${diffHours > 1 ? 's' : ''}`;
  } else {
    result = `${diffDays} día${diffDays > 1 ? 's' : ''}`;
  }
  
  return options?.addSuffix ? `hace ${result}` : result;
};

interface AlertStats {
  total: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  unacknowledged: number;
  acknowledgedToday: number;
  avgResponseTime: number;
}

interface AlertFilter {
  severity: ('critical' | 'high' | 'medium' | 'low')[];
  type: string[];
  acknowledged: 'all' | 'unacknowledged' | 'acknowledged';
  timeRange: '1H' | '6H' | '1D' | '1W' | 'ALL';
}

interface NotificationSettings {
  enabled: boolean;
  soundEnabled: boolean;
  emailEnabled: boolean;
  smsEnabled: boolean;
  pushEnabled: boolean;
  severityThreshold: 'low' | 'medium' | 'high' | 'critical';
  autoAcknowledge: boolean;
  snoozeEnabled: boolean;
}

interface RiskThreshold {
  id: string;
  name: string;
  metric: string;
  operator: 'greater' | 'less' | 'equals';
  value: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
}

export default function RiskAlertsCenter() {
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [filteredAlerts, setFilteredAlerts] = useState<RiskAlert[]>([]);
  const [alertStats, setAlertStats] = useState<AlertStats | null>(null);
  const [filter, setFilter] = useState<AlertFilter>({
    severity: ['critical', 'high', 'medium', 'low'],
    type: [],
    acknowledged: 'all',
    timeRange: '1D'
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedAlert, setSelectedAlert] = useState<RiskAlert | null>(null);
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings>({
    enabled: true,
    soundEnabled: true,
    emailEnabled: false,
    smsEnabled: false,
    pushEnabled: true,
    severityThreshold: 'medium',
    autoAcknowledge: false,
    snoozeEnabled: true
  });
  const [customThresholds, setCustomThresholds] = useState<RiskThreshold[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  // Audio notification
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Generate mock alerts for demonstration
  const generateMockAlerts = useCallback((): RiskAlert[] => {
    const now = new Date();
    return [
      {
        id: 'alert-001',
        type: 'limit_breach',
        severity: 'critical',
        message: 'Portfolio VaR 95% exceeded limit by 23%',
        details: {
          currentValue: 615000,
          limitValue: 500000,
          breach: 115000,
          metric: 'VaR95',
          position: 'USDCOP_LONG'
        },
        timestamp: new Date(now.getTime() - 5 * 60000), // 5 minutes ago
        acknowledged: false,
        position: 'USDCOP Long Position',
        currentValue: 615000,
        limitValue: 500000,
        recommendation: 'Immediately reduce position size by 20% or implement tail hedge'
      },
      {
        id: 'alert-002',
        type: 'correlation_spike',
        severity: 'high',
        message: 'Cross-asset correlation surge detected - diversification breakdown',
        details: {
          correlation: 0.94,
          threshold: 0.85,
          assets: ['USDCOP', 'Oil', 'EM_Currencies'],
          timeframe: '1H'
        },
        timestamp: new Date(now.getTime() - 12 * 60000), // 12 minutes ago
        acknowledged: false,
        currentValue: 0.94,
        limitValue: 0.85,
        recommendation: 'Review portfolio construction and consider alternative diversifiers'
      },
      {
        id: 'alert-003',
        type: 'volatility_surge',
        severity: 'high',
        message: 'USDCOP implied volatility spike - 40% increase in 1 hour',
        details: {
          currentVol: 28.5,
          previousVol: 20.3,
          increase: 0.40,
          timeframe: '1H'
        },
        timestamp: new Date(now.getTime() - 18 * 60000), // 18 minutes ago
        acknowledged: false,
        position: 'USDCOP Options',
        currentValue: 28.5,
        limitValue: 25.0,
        recommendation: 'Monitor for potential volatility regime change, consider hedging gamma exposure'
      },
      {
        id: 'alert-004',
        type: 'limit_breach',
        severity: 'medium',
        message: 'Position concentration limit breached for Colombian assets',
        details: {
          concentration: 0.87,
          limit: 0.80,
          country: 'Colombia',
          sectors: ['FX', 'Rates', 'Commodities']
        },
        timestamp: new Date(now.getTime() - 25 * 60000), // 25 minutes ago
        acknowledged: true,
        currentValue: 0.87,
        limitValue: 0.80,
        recommendation: 'Diversify geographically or reduce Colombian exposure by 7%'
      },
      {
        id: 'alert-005',
        type: 'liquidity_crisis',
        severity: 'medium',
        message: 'Market liquidity deterioration - bid-ask spreads widening',
        details: {
          avgSpread: 12.5,
          normalSpread: 6.2,
          increase: 1.02,
          affectedPositions: 8
        },
        timestamp: new Date(now.getTime() - 35 * 60000), // 35 minutes ago
        acknowledged: true,
        currentValue: 12.5,
        limitValue: 10.0,
        recommendation: 'Reduce position sizes during illiquid periods'
      },
      {
        id: 'alert-006',
        type: 'model_break',
        severity: 'low',
        message: 'VaR model back-test exception - consider recalibration',
        details: {
          exceptions: 7,
          expectedExceptions: 5,
          testPeriod: '100D',
          confidenceLevel: 0.95
        },
        timestamp: new Date(now.getTime() - 45 * 60000), // 45 minutes ago
        acknowledged: false,
        currentValue: 7,
        limitValue: 5,
        recommendation: 'Review risk model parameters and historical data quality'
      }
    ];
  }, []);

  const calculateAlertStats = useCallback((alerts: RiskAlert[]): AlertStats => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    
    const stats = alerts.reduce((acc, alert) => {
      acc.total++;
      acc[alert.severity as keyof AlertStats]++;
      
      if (!alert.acknowledged) {
        acc.unacknowledged++;
      } else if (alert.timestamp >= today) {
        acc.acknowledgedToday++;
      }
      
      return acc;
    }, {
      total: 0,
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      unacknowledged: 0,
      acknowledgedToday: 0,
      avgResponseTime: 0
    } as AlertStats);
    
    // Calculate average response time (mock calculation)
    const acknowledgedAlerts = alerts.filter(a => a.acknowledged);
    stats.avgResponseTime = acknowledgedAlerts.length > 0 
      ? acknowledgedAlerts.reduce((sum, alert) => sum + 15, 0) / acknowledgedAlerts.length // Mock 15 min avg
      : 0;
    
    return stats;
  }, []);

  useEffect(() => {
    const loadAlerts = async () => {
      setLoading(true);
      
      // Simulate loading delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Get alerts from risk engine and add mock alerts
      const engineAlerts = realTimeRiskEngine.getAlerts() || [];
      const mockAlerts = generateMockAlerts();
      const allAlerts = [...engineAlerts, ...mockAlerts];
      
      setAlerts(allAlerts);
      setAlertStats(calculateAlertStats(allAlerts));
      setLastUpdate(new Date());
      setLoading(false);
    };

    loadAlerts();
    
    // Set up periodic refresh
    const interval = setInterval(loadAlerts, 30000); // Every 30 seconds
    
    return () => clearInterval(interval);
  }, [generateMockAlerts, calculateAlertStats]);

  // Filter and search alerts
  useEffect(() => {
    let filtered = alerts;
    
    // Apply severity filter
    if (filter.severity.length < 4) {
      filtered = filtered.filter(alert => filter.severity.includes(alert.severity));
    }
    
    // Apply acknowledgment filter
    if (filter.acknowledged === 'unacknowledged') {
      filtered = filtered.filter(alert => !alert.acknowledged);
    } else if (filter.acknowledged === 'acknowledged') {
      filtered = filtered.filter(alert => alert.acknowledged);
    }
    
    // Apply time range filter
    if (filter.timeRange !== 'ALL') {
      const now = new Date();
      const timeFilters = {
        '1H': 1 * 60 * 60 * 1000,
        '6H': 6 * 60 * 60 * 1000,
        '1D': 24 * 60 * 60 * 1000,
        '1W': 7 * 24 * 60 * 60 * 1000
      };
      const cutoff = new Date(now.getTime() - timeFilters[filter.timeRange]);
      filtered = filtered.filter(alert => alert.timestamp >= cutoff);
    }
    
    // Apply search query
    if (searchQuery) {
      filtered = filtered.filter(alert =>
        alert.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (alert.position && alert.position.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }
    
    // Sort by severity and timestamp
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    filtered.sort((a, b) => {
      if (a.acknowledged !== b.acknowledged) {
        return a.acknowledged ? 1 : -1;
      }
      if (a.severity !== b.severity) {
        return severityOrder[a.severity] - severityOrder[b.severity];
      }
      return b.timestamp.getTime() - a.timestamp.getTime();
    });
    
    setFilteredAlerts(filtered);
  }, [alerts, filter, searchQuery]);

  // Play notification sound for new critical alerts
  useEffect(() => {
    if (notificationSettings.enabled && notificationSettings.soundEnabled && audioRef.current) {
      const criticalAlerts = alerts.filter(alert => 
        alert.severity === 'critical' && !alert.acknowledged
      );
      
      if (criticalAlerts.length > 0 && !isPlaying) {
        setIsPlaying(true);
        audioRef.current.play().catch(console.error);
        setTimeout(() => setIsPlaying(false), 3000);
      }
    }
  }, [alerts, notificationSettings, isPlaying]);

  const acknowledgeAlert = useCallback(async (alertId: string) => {
    const success = realTimeRiskEngine.acknowledgeAlert(alertId);
    if (success) {
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledged: true }
          : alert
      ));
    }
  }, []);

  const acknowledgeAll = useCallback(() => {
    const unacknowledgedIds = alerts
      .filter(alert => !alert.acknowledged)
      .map(alert => alert.id);
    
    unacknowledgedIds.forEach(id => {
      realTimeRiskEngine.acknowledgeAlert(id);
    });
    
    setAlerts(prev => prev.map(alert => ({ ...alert, acknowledged: true })));
  }, [alerts]);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertOctagon className="h-4 w-4 text-red-500" />;
      case 'high':
        return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      case 'medium':
        return <Info className="h-4 w-4 text-yellow-500" />;
      case 'low':
        return <Bell className="h-4 w-4 text-blue-500" />;
      default:
        return <Bell className="h-4 w-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'border-red-500 bg-red-950/30';
      case 'high':
        return 'border-orange-500 bg-orange-950/30';
      case 'medium':
        return 'border-yellow-500 bg-yellow-950/30';
      case 'low':
        return 'border-blue-500 bg-blue-950/30';
      default:
        return 'border-slate-500 bg-slate-950/30';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Risk Alerts Center...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Audio element for notifications */}
      <audio 
        ref={audioRef}
        src="/notification-sound.mp3"
        preload="auto"
      />

      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono flex items-center gap-2">
            <Shield className="h-6 w-6" />
            RISK ALERTS CENTER
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Real-Time Risk Monitoring • Last Update: {formatDate(lastUpdate, 'HH:mm:ss')} • 
            Active Alerts: {filteredAlerts.filter(a => !a.acknowledged).length}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Notification Status */}
          <div className="flex items-center gap-2">
            {notificationSettings.enabled ? (
              <BellRing className="h-5 w-5 text-green-400" />
            ) : (
              <BellOff className="h-5 w-5 text-slate-500" />
            )}
            {notificationSettings.soundEnabled ? (
              <Volume2 className="h-4 w-4 text-blue-400" />
            ) : (
              <VolumeX className="h-4 w-4 text-slate-500" />
            )}
          </div>
          
          {/* Action Buttons */}
          <Button
            onClick={acknowledgeAll}
            disabled={alerts.filter(a => !a.acknowledged).length === 0}
            className="bg-green-900 hover:bg-green-800 text-green-100 border-green-500/20"
          >
            <CheckCircle className="h-4 w-4 mr-2" />
            Ack All
          </Button>
          
          <Button
            onClick={() => setShowSettings(!showSettings)}
            className="bg-slate-900 hover:bg-slate-800 text-amber-500 border-amber-500/20"
          >
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
          
          <Button
            onClick={() => window.location.reload()}
            className="bg-slate-900 hover:bg-slate-800 text-amber-500 border-amber-500/20"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Alert Statistics */}
      {alertStats && (
        <div className="grid grid-cols-2 lg:grid-cols-7 gap-4">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-white font-mono">{alertStats.total}</div>
                <div className="text-xs text-slate-400">Total Alerts</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-red-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400 font-mono">{alertStats.critical}</div>
                <div className="text-xs text-slate-400">Critical</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-orange-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400 font-mono">{alertStats.high}</div>
                <div className="text-xs text-slate-400">High</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-yellow-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-400 font-mono">{alertStats.medium}</div>
                <div className="text-xs text-slate-400">Medium</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-blue-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400 font-mono">{alertStats.low}</div>
                <div className="text-xs text-slate-400">Low</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-amber-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-amber-400 font-mono">{alertStats.unacknowledged}</div>
                <div className="text-xs text-slate-400">Unack'd</div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-slate-900 border-green-500/20">
            <CardContent className="p-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400 font-mono">{alertStats.avgResponseTime.toFixed(0)}m</div>
                <div className="text-xs text-slate-400">Avg Response</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters and Search */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Search */}
            <div className="relative flex-1 min-w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search alerts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-amber-500 focus:outline-none"
              />
            </div>
            
            {/* Severity Filter */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-sm">Severity:</span>
              {['critical', 'high', 'medium', 'low'].map((severity) => (
                <Button
                  key={severity}
                  onClick={() => {
                    setFilter(prev => ({
                      ...prev,
                      severity: prev.severity.includes(severity as any)
                        ? prev.severity.filter(s => s !== severity)
                        : [...prev.severity, severity as any]
                    }));
                  }}
                  className={`px-3 py-1 text-xs ${
                    filter.severity.includes(severity as any)
                      ? 'bg-amber-500 text-slate-950'
                      : 'bg-slate-800 text-slate-400 border-slate-600'
                  }`}
                >
                  {severity.toUpperCase()}
                </Button>
              ))}
            </div>
            
            {/* Acknowledgment Filter */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-sm">Status:</span>
              <select
                value={filter.acknowledged}
                onChange={(e) => setFilter(prev => ({ ...prev, acknowledged: e.target.value as any }))}
                className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-amber-500 focus:outline-none"
              >
                <option value="all">All</option>
                <option value="unacknowledged">Unacknowledged</option>
                <option value="acknowledged">Acknowledged</option>
              </select>
            </div>
            
            {/* Time Range Filter */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-sm">Time:</span>
              <select
                value={filter.timeRange}
                onChange={(e) => setFilter(prev => ({ ...prev, timeRange: e.target.value as any }))}
                className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:border-amber-500 focus:outline-none"
              >
                <option value="1H">1 Hour</option>
                <option value="6H">6 Hours</option>
                <option value="1D">1 Day</option>
                <option value="1W">1 Week</option>
                <option value="ALL">All Time</option>
              </select>
            </div>
            
            <div className="text-slate-400 text-sm">
              Showing {filteredAlerts.length} of {alerts.length} alerts
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Alerts List */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <Card className="bg-slate-900 border-amber-500/20">
            <CardContent className="p-8 text-center">
              <div className="text-slate-400">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-400" />
                <h3 className="text-lg font-semibold text-white mb-2">No Active Alerts</h3>
                <p className="text-sm">All systems operating within normal parameters</p>
              </div>
            </CardContent>
          </Card>
        ) : (
          filteredAlerts.map((alert) => (
            <Card key={alert.id} className={`border ${getSeverityColor(alert.severity)} ${
              alert.acknowledged ? 'opacity-70' : ''
            } hover:border-opacity-100 transition-all`}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    {getSeverityIcon(alert.severity)}
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-white">{alert.message}</span>
                        {alert.acknowledged && (
                          <Badge className="bg-green-950 text-green-400 text-xs">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            ACK
                          </Badge>
                        )}
                        <Badge className={`text-xs ${
                          alert.severity === 'critical' ? 'bg-red-950 text-red-400' :
                          alert.severity === 'high' ? 'bg-orange-950 text-orange-400' :
                          alert.severity === 'medium' ? 'bg-yellow-950 text-yellow-400' :
                          'bg-blue-950 text-blue-400'
                        }`}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-slate-400 mb-1">Alert Details:</div>
                          {alert.position && (
                            <div className="text-slate-300">Position: {alert.position}</div>
                          )}
                          <div className="text-slate-300">Type: {alert.type.replace('_', ' ').toUpperCase()}</div>
                          <div className="text-slate-300">
                            Time: {formatDate(alert.timestamp, 'HH:mm:ss')} 
                            <span className="text-slate-500 ml-2">
                              ({formatDistanceToNow(alert.timestamp, { addSuffix: true })})
                            </span>
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-slate-400 mb-1">Risk Metrics:</div>
                          <div className="text-slate-300">
                            Current: <span className="font-mono text-amber-400">
                              {typeof alert.currentValue === 'number' && alert.currentValue > 1000 
                                ? formatCurrency(alert.currentValue)
                                : alert.currentValue?.toFixed(4) || 'N/A'
                              }
                            </span>
                          </div>
                          <div className="text-slate-300">
                            Limit: <span className="font-mono text-red-400">
                              {typeof alert.limitValue === 'number' && alert.limitValue > 1000
                                ? formatCurrency(alert.limitValue)
                                : alert.limitValue?.toFixed(4) || 'N/A'
                              }
                            </span>
                          </div>
                          {alert.currentValue && alert.limitValue && (
                            <div className="text-slate-300">
                              Breach: <span className="font-mono text-orange-400">
                                {((Math.abs(alert.currentValue - alert.limitValue) / alert.limitValue) * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {alert.recommendation && (
                        <div className="mt-3 p-3 bg-slate-800 rounded-lg">
                          <div className="text-slate-400 text-xs mb-1">RECOMMENDATION:</div>
                          <div className="text-amber-300 text-sm">{alert.recommendation}</div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 ml-4">
                    {!alert.acknowledged ? (
                      <Button
                        onClick={() => acknowledgeAlert(alert.id)}
                        className="bg-green-900 hover:bg-green-800 text-green-100 text-sm px-3 py-1"
                      >
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Acknowledge
                      </Button>
                    ) : (
                      <div className="text-green-400 text-sm flex items-center">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Acknowledged
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Progress bar for alerts with quantifiable metrics */}
                {alert.currentValue && alert.limitValue && (
                  <div className="mt-3">
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>Risk Level</span>
                      <span>{((alert.currentValue / alert.limitValue) * 100).toFixed(0)}% of limit</span>
                    </div>
                    <Progress 
                      value={Math.min((alert.currentValue / alert.limitValue) * 100, 100)} 
                      className="h-2"
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Risk Alerts Center • Real-Time Monitoring • Automated Notifications • 
          Professional Risk Management • Generated {formatDate(new Date(), 'PPpp')}
        </p>
      </div>
    </div>
  );
}