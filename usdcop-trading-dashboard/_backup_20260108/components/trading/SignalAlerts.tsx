'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Bell,
  BellOff,
  Volume2,
  VolumeX,
  Settings,
  Filter,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  Target,
  Zap
} from 'lucide-react';

interface SignalAlert {
  id: string;
  timestamp: string;
  type: 'BUY' | 'SELL' | 'HOLD' | 'SYSTEM' | 'RISK';
  title: string;
  message: string;
  confidence?: number;
  price?: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  acknowledged: boolean;
  actionRequired: boolean;
  source: string;
}

interface AlertSettings {
  enabled: boolean;
  sound: boolean;
  minConfidence: number;
  signalTypes: string[];
  riskThreshold: number;
  systemAlerts: boolean;
}

export default function SignalAlerts() {
  const [alerts, setAlerts] = useState<SignalAlert[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<AlertSettings>({
    enabled: true,
    sound: true,
    minConfidence: 75,
    signalTypes: ['BUY', 'SELL'],
    riskThreshold: 5,
    systemAlerts: true
  });
  const [showSettings, setShowSettings] = useState(false);
  const [filter, setFilter] = useState<'ALL' | 'UNREAD' | 'HIGH'>('UNREAD');

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 3000);
    return () => clearInterval(interval);
  }, [settings]);

  useEffect(() => {
    // Request notification permissions
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const loadAlerts = async () => {
    try {
      const response = await fetch('/api/trading/signals-test');
      if (response.ok) {
        const data = await response.json();
        if (data.signals) {
          setError(null); // Clear any previous errors on success
          const newAlerts = data.signals
            .filter((signal: any) => signal.confidence >= settings.minConfidence)
            .filter((signal: any) => settings.signalTypes.includes(signal.type))
            .map((signal: any) => ({
              id: `alert_${signal.id}`,
              timestamp: signal.timestamp,
              type: signal.type,
              title: `${signal.type} Signal`,
              message: `${signal.type} signal at $${signal.price.toFixed(2)} with ${signal.confidence.toFixed(1)}% confidence`,
              confidence: signal.confidence,
              price: signal.price,
              severity: getSeverity(signal.confidence, signal.riskScore),
              acknowledged: false,
              actionRequired: signal.type !== 'HOLD',
              source: signal.modelSource
            }));

          // Add system alerts if enabled
          if (settings.systemAlerts) {
            newAlerts.push({
              id: `system_${Date.now()}`,
              timestamp: new Date().toISOString(),
              type: 'SYSTEM' as any,
              title: 'System Status',
              message: `ML pipeline active. Generated ${data.signals.length} signals`,
              severity: 'low' as any,
              acknowledged: false,
              actionRequired: false,
              source: 'System Monitor'
            });
          }

          // Check for new high-severity alerts
          const highSeverityAlerts = newAlerts.filter(alert =>
            alert.severity === 'high' || alert.severity === 'critical'
          );

          if (highSeverityAlerts.length > 0 && settings.enabled) {
            highSeverityAlerts.forEach(alert => {
              showNotification(alert);
              if (settings.sound) {
                playAlertSound(alert.severity);
              }
            });
          }

          setAlerts(prev => {
            const existing = prev.filter(alert =>
              !newAlerts.some(newAlert => newAlert.id === alert.id)
            );
            return [...newAlerts, ...existing].slice(0, 50); // Keep latest 50 alerts
          });
        }
      } else {
        console.error('Failed to fetch signals:', response.status);
        setError('Failed to load trading signals');
      }
    } catch (error) {
      console.error('Error loading alerts:', error);
      setError('Failed to load trading signals');
    }
  };

  const getSeverity = (confidence: number, riskScore: number): 'low' | 'medium' | 'high' | 'critical' => {
    if (confidence >= 90 && riskScore <= 2) return 'critical';
    if (confidence >= 80 || riskScore >= 7) return 'high';
    if (confidence >= 70) return 'medium';
    return 'low';
  };

  const showNotification = (alert: SignalAlert) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`${alert.title} - ${alert.source}`, {
        body: alert.message,
        icon: '/favicon.ico',
        tag: alert.id
      });
    }
  };

  const playAlertSound = (severity: string) => {
    try {
      const audioContext = new AudioContext();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      // Different tones for different severities
      const frequency = severity === 'critical' ? 800 : severity === 'high' ? 600 : 400;
      oscillator.frequency.value = frequency;
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.3);
    } catch (error) {
      console.log('Could not play alert sound:', error);
    }
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  const clearAllAlerts = () => {
    setAlerts(prev => prev.map(alert => ({ ...alert, acknowledged: true })));
  };

  const dismissAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const getAlertIcon = (type: string, severity: string) => {
    const iconClass = severity === 'critical' ? 'text-red-500' :
      severity === 'high' ? 'text-orange-500' :
      severity === 'medium' ? 'text-yellow-500' : 'text-blue-500';

    switch (type) {
      case 'BUY': return <TrendingUp className={`h-5 w-5 ${iconClass}`} />;
      case 'SELL': return <TrendingDown className={`h-5 w-5 ${iconClass}`} />;
      case 'HOLD': return <Activity className={`h-5 w-5 ${iconClass}`} />;
      case 'SYSTEM': return <Zap className={`h-5 w-5 ${iconClass}`} />;
      case 'RISK': return <AlertTriangle className={`h-5 w-5 ${iconClass}`} />;
      default: return <Bell className={`h-5 w-5 ${iconClass}`} />;
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    switch (filter) {
      case 'UNREAD': return !alert.acknowledged;
      case 'HIGH': return alert.severity === 'high' || alert.severity === 'critical';
      default: return true;
    }
  });

  const unreadCount = alerts.filter(alert => !alert.acknowledged).length;
  const highPriorityCount = alerts.filter(alert => 
    (alert.severity === 'high' || alert.severity === 'critical') && !alert.acknowledged
  ).length;

  return (
    <div className="space-y-4">
      {/* Error Banner */}
      {error && (
        <div className="mb-4 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
          <div className="flex items-center gap-2 text-yellow-400 text-sm">
            <AlertTriangle className="w-4 h-4" />
            <span>{error}</span>
            <button
              onClick={() => { setError(null); loadAlerts(); }}
              className="ml-auto text-xs underline hover:text-yellow-300 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Alert Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Button
                  variant={settings.enabled ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
                  className="flex items-center gap-2"
                >
                  {settings.enabled ? <Bell className="h-4 w-4" /> : <BellOff className="h-4 w-4" />}
                  {settings.enabled ? 'Enabled' : 'Disabled'}
                </Button>
                
                <Button
                  variant={settings.sound ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSettings(prev => ({ ...prev, sound: !prev.sound }))}
                  className="flex items-center gap-2"
                >
                  {settings.sound ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                  Sound
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSettings(!showSettings)}
                  className="flex items-center gap-2"
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <Badge variant={unreadCount > 0 ? "destructive" : "outline"}>
                  {unreadCount} unread
                </Badge>
                {highPriorityCount > 0 && (
                  <Badge variant="destructive">
                    {highPriorityCount} high priority
                  </Badge>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                {(['ALL', 'UNREAD', 'HIGH'] as const).map(filterType => (
                  <Button
                    key={filterType}
                    variant={filter === filterType ? "default" : "outline"}
                    size="sm"
                    onClick={() => setFilter(filterType)}
                  >
                    {filterType}
                  </Button>
                ))}
              </div>
              
              <Button variant="outline" size="sm" onClick={clearAllAlerts}>
                <CheckCircle className="h-4 w-4 mr-2" />
                Clear All
              </Button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="mt-4 p-4 border rounded-lg bg-gray-50">
              <h4 className="font-medium mb-3">Alert Settings</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Minimum Confidence: {settings.minConfidence}%
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="95"
                    value={settings.minConfidence}
                    onChange={(e) => setSettings(prev => ({ ...prev, minConfidence: Number(e.target.value) }))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Signal Types</label>
                  <div className="flex gap-2">
                    {['BUY', 'SELL', 'HOLD'].map(type => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={settings.signalTypes.includes(type)}
                          onChange={(e) => {
                            const types = e.target.checked
                              ? [...settings.signalTypes, type]
                              : settings.signalTypes.filter(t => t !== type);
                            setSettings(prev => ({ ...prev, signalTypes: types }));
                          }}
                          className="mr-1"
                        />
                        {type}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alerts List */}
      <div className="space-y-2">
        {filteredAlerts.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center">
              <p className="text-gray-500">No alerts matching criteria</p>
            </CardContent>
          </Card>
        ) : (
          filteredAlerts.map(alert => (
            <Card 
              key={alert.id} 
              className={`${alert.acknowledged ? 'opacity-60' : ''} ${
                alert.severity === 'critical' ? 'border-red-500' :
                alert.severity === 'high' ? 'border-orange-500' :
                alert.severity === 'medium' ? 'border-yellow-500' : ''
              }`}
            >
              <CardContent className="pt-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    {getAlertIcon(alert.type, alert.severity)}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{alert.title}</h4>
                        <Badge 
                          variant={
                            alert.severity === 'critical' ? 'destructive' :
                            alert.severity === 'high' ? 'destructive' :
                            alert.severity === 'medium' ? 'default' : 'outline'
                          }
                        >
                          {alert.severity}
                        </Badge>
                        {alert.confidence && (
                          <Badge variant="outline">
                            {alert.confidence.toFixed(1)}%
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{alert.message}</p>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                        <span>{alert.source}</span>
                        {alert.actionRequired && (
                          <Badge variant="outline" className="text-xs">
                            <Target className="h-3 w-3 mr-1" />
                            Action Required
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {!alert.acknowledged && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => acknowledgeAlert(alert.id)}
                        className="flex items-center gap-1"
                      >
                        <CheckCircle className="h-3 w-3" />
                        Ack
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => dismissAlert(alert.id)}
                      className="flex items-center gap-1"
                    >
                      <XCircle className="h-3 w-3" />
                      Dismiss
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}