'use client';

import React, { useState, useEffect } from 'react';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { 
  enhancedApiMonitor, 
  APIAlert
} from '@/lib/services/enhanced-api-monitor';
import { 
  AlertTriangle, 
  X, 
  Bell, 
  BellOff, 
  Clock,
  DollarSign,
  Activity,
  Zap
} from 'lucide-react';
import toast from 'react-hot-toast';

interface APIAlertSystemProps {
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  maxAlerts?: number;
  autoHideAfter?: number; // seconds
}

export default function APIAlertSystem({ 
  position = 'top-right',
  maxAlerts = 5,
  autoHideAfter = 30
}: APIAlertSystemProps) {
  const [alerts, setAlerts] = useState<APIAlert[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [allAlerts, setAllAlerts] = useState<APIAlert[]>([]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (alertsEnabled) {
      // Check for alerts every 30 seconds
      interval = setInterval(async () => {
        try {
          const alertResponse = await enhancedApiMonitor.getAlerts();
          const newAlerts = alertResponse.alerts.filter(
            alert => !dismissed.has(alert.timestamp)
          );
          
          setAlerts(newAlerts);
          setAllAlerts(alertResponse.alerts);
          
          // Auto-dismiss alerts after specified time
          if (autoHideAfter > 0) {
            setTimeout(() => {
              newAlerts.forEach(alert => {
                setDismissed(prev => new Set([...prev, alert.timestamp]));
              });
            }, autoHideAfter * 1000);
          }
          
        } catch (error) {
          console.error('Failed to fetch alerts:', error);
        }
      }, 30000);
      
      // Initial fetch
      enhancedApiMonitor.getAlerts().then(response => {
        setAlerts(response.alerts);
        setAllAlerts(response.alerts);
      }).catch(console.error);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [alertsEnabled, dismissed, autoHideAfter]);

  const dismissAlert = (timestamp: string) => {
    setDismissed(prev => new Set([...prev, timestamp]));
    setAlerts(prev => prev.filter(alert => alert.timestamp !== timestamp));
  };

  const dismissAllAlerts = () => {
    alerts.forEach(alert => {
      setDismissed(prev => new Set([...prev, alert.timestamp]));
    });
    setAlerts([]);
  };

  const toggleAlerts = () => {
    setAlertsEnabled(!alertsEnabled);
    if (!alertsEnabled) {
      toast.success('API alerts enabled');
    } else {
      toast('API alerts disabled', { icon: '🔕' });
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'RATE_LIMIT_WARNING': return <Clock className="w-4 h-4" />;
      case 'COST_WARNING': return <DollarSign className="w-4 h-4" />;
      case 'ERROR_RATE_WARNING': return <AlertTriangle className="w-4 h-4" />;
      case 'RESPONSE_TIME_WARNING': return <Zap className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'border-red-500 bg-red-50 text-red-800';
      case 'WARNING': return 'border-yellow-500 bg-yellow-50 text-yellow-800';
      default: return 'border-blue-500 bg-blue-50 text-blue-800';
    }
  };

  const getPositionClasses = () => {
    switch (position) {
      case 'top-left': return 'top-4 left-4';
      case 'top-right': return 'top-4 right-4';
      case 'bottom-left': return 'bottom-4 left-4';
      case 'bottom-right': return 'bottom-4 right-4';
      default: return 'top-4 right-4';
    }
  };

  const visibleAlerts = alerts.slice(0, maxAlerts);
  const hasMoreAlerts = alerts.length > maxAlerts;

  return (
    <>
      {/* Alert Control Button */}
      <div className={`fixed ${getPositionClasses()} z-50`}>
        <div className="flex flex-col space-y-2">
          {/* Toggle Button */}
          <Button
            onClick={toggleAlerts}
            variant={alertsEnabled ? "default" : "outline"}
            size="sm"
            className="shadow-lg"
          >
            {alertsEnabled ? (
              <Bell className="w-4 h-4 mr-1" />
            ) : (
              <BellOff className="w-4 h-4 mr-1" />
            )}
            {alerts.length > 0 && (
              <Badge variant="secondary" className="ml-1 px-1 text-xs">
                {alerts.length}
              </Badge>
            )}
          </Button>

          {/* Alert List */}
          {alertsEnabled && visibleAlerts.length > 0 && (
            <div className="space-y-2 max-w-sm">
              {visibleAlerts.map((alert, index) => (
                <Alert 
                  key={alert.timestamp}
                  className={`${getSeverityColor(alert.severity)} shadow-lg border animate-in slide-in-from-right-full duration-300`}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-start justify-between w-full">
                    <div className="flex items-start space-x-2 flex-1">
                      <div className="flex-shrink-0 mt-0.5">
                        {getAlertIcon(alert.type)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <Badge 
                            variant={alert.severity === 'CRITICAL' ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            {alert.severity}
                          </Badge>
                          <span className="text-xs opacity-70">
                            {alert.api_name}
                          </span>
                        </div>
                        
                        <p className="text-sm font-medium mb-1">
                          {alert.message}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-xs opacity-70">
                            {alert.key_id}
                          </span>
                          <span className="text-xs opacity-70">
                            {new Date(alert.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <Button
                      onClick={() => dismissAlert(alert.timestamp)}
                      variant="ghost"
                      size="sm"
                      className="flex-shrink-0 p-1 h-auto w-auto opacity-70 hover:opacity-100"
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                </Alert>
              ))}
              
              {/* More alerts indicator */}
              {hasMoreAlerts && (
                <Alert className="bg-gray-50 border-gray-300 text-gray-700 text-center">
                  <p className="text-sm">
                    + {alerts.length - maxAlerts} more alert{alerts.length - maxAlerts !== 1 ? 's' : ''}
                  </p>
                  <Button
                    onClick={() => setShowHistory(true)}
                    variant="ghost"
                    size="sm"
                    className="mt-1 p-1 h-auto text-xs"
                  >
                    View All
                  </Button>
                </Alert>
              )}
              
              {/* Dismiss All Button */}
              {alerts.length > 1 && (
                <Button
                  onClick={dismissAllAlerts}
                  variant="outline"
                  size="sm"
                  className="w-full shadow-lg"
                >
                  Dismiss All
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Alert History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-[100] flex items-center justify-center p-4">
          <Card className="w-full max-w-2xl max-h-[80vh] overflow-hidden">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Alert History</h3>
                <Button
                  onClick={() => setShowHistory(false)}
                  variant="ghost"
                  size="sm"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {allAlerts.length === 0 ? (
                  <p className="text-center text-gray-500 py-8">
                    No alerts found
                  </p>
                ) : (
                  allAlerts.map((alert, index) => (
                    <div 
                      key={alert.timestamp}
                      className={`p-3 rounded-lg border ${getSeverityColor(alert.severity)}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-2 flex-1">
                          <div className="flex-shrink-0 mt-0.5">
                            {getAlertIcon(alert.type)}
                          </div>
                          
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-1">
                              <Badge 
                                variant={alert.severity === 'CRITICAL' ? 'destructive' : 'secondary'}
                                className="text-xs"
                              >
                                {alert.severity}
                              </Badge>
                              <span className="text-xs opacity-70">
                                {alert.api_name} • {alert.key_id}
                              </span>
                            </div>
                            
                            <p className="text-sm font-medium mb-1">
                              {alert.message}
                            </p>
                            
                            <p className="text-xs opacity-70">
                              {new Date(alert.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              <div className="mt-4 pt-4 border-t">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600">
                    Total alerts: {allAlerts.length}
                  </p>
                  <Button
                    onClick={() => setShowHistory(false)}
                    variant="outline"
                    size="sm"
                  >
                    Close
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}