/**
 * Error Notifications System
 * User-friendly error messages and toast notifications
 */

'use client';

import React, { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertTriangle, 
  X, 
  RefreshCw, 
  Wifi, 
  WifiOff, 
  AlertCircle, 
  CheckCircle, 
  Info,
  Bug,
  Activity,
  Clock
} from 'lucide-react';
import { errorMonitoring, ErrorType, ErrorSeverity, ErrorReport } from '@/lib/services/error-monitoring';
import { networkErrorHandler, NetworkStatus } from '@/lib/services/network-error-handler';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  duration?: number;
  persistent?: boolean;
  actions?: NotificationAction[];
  metadata?: Record<string, any>;
  createdAt: string;
}

export type NotificationType = 'error' | 'warning' | 'info' | 'success' | 'network';

export interface NotificationAction {
  label: string;
  action: () => void;
  style?: 'primary' | 'secondary' | 'danger';
}

interface NotificationContextType {
  notifications: Notification[];
  showNotification: (notification: Omit<Notification, 'id' | 'createdAt'>) => string;
  dismissNotification: (id: string) => void;
  clearAll: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};

interface NotificationProviderProps {
  children: ReactNode;
  maxNotifications?: number;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ 
  children, 
  maxNotifications = 5 
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({ online: true });

  useEffect(() => {
    // Subscribe to error monitoring
    const unsubscribeErrors = errorMonitoring.onError((error: ErrorReport) => {
      showErrorNotification(error);
    });

    // Subscribe to network status changes
    const unsubscribeNetwork = networkErrorHandler.onNetworkStatusChange((status: NetworkStatus) => {
      const wasOffline = !networkStatus.online;
      const isNowOnline = status.online;
      
      setNetworkStatus(status);

      if (wasOffline && isNowOnline) {
        showNotification({
          type: 'success',
          title: 'Connection Restored',
          message: 'Network connection has been restored. Retrying failed requests.',
          duration: 5000
        });
      } else if (!status.online) {
        showNotification({
          type: 'network',
          title: 'Connection Lost',
          message: 'Network connection has been lost. Some features may not work properly.',
          persistent: true,
          actions: [{
            label: 'Retry',
            action: () => window.location.reload()
          }]
        });
      }
    });

    return () => {
      unsubscribeErrors();
      unsubscribeNetwork();
    };
  }, [networkStatus.online]);

  const showErrorNotification = (error: ErrorReport) => {
    const { type, severity, message } = error;
    
    let notificationData: Omit<Notification, 'id' | 'createdAt'>;

    switch (type) {
      case ErrorType.NETWORK:
        notificationData = {
          type: 'network',
          title: 'Network Error',
          message: 'Unable to connect to the server. Please check your connection.',
          persistent: true,
          actions: [
            {
              label: 'Retry',
              action: () => networkErrorHandler.retryFailedCalls(),
              style: 'primary'
            },
            {
              label: 'Dismiss',
              action: () => dismissNotification(error.id),
              style: 'secondary'
            }
          ]
        };
        break;

      case ErrorType.API:
        notificationData = {
          type: 'error',
          title: 'API Error',
          message: 'Server request failed. Data may not be up to date.',
          duration: 8000,
          actions: [
            {
              label: 'Retry',
              action: () => window.location.reload(),
              style: 'primary'
            }
          ]
        };
        break;

      case ErrorType.CHART:
        notificationData = {
          type: 'warning',
          title: 'Chart Error',
          message: 'Chart failed to load. Some visualizations may be unavailable.',
          duration: 6000,
          actions: [
            {
              label: 'Refresh Charts',
              action: () => window.location.reload(),
              style: 'primary'
            }
          ]
        };
        break;

      case ErrorType.DATA:
        notificationData = {
          type: 'warning',
          title: 'Data Error',
          message: 'Data processing failed. Some information may be incomplete.',
          duration: 6000
        };
        break;

      case ErrorType.COMPONENT:
        if (severity === ErrorSeverity.CRITICAL) {
          notificationData = {
            type: 'error',
            title: 'Critical Error',
            message: 'A critical component error occurred. Please refresh the page.',
            persistent: true,
            actions: [
              {
                label: 'Refresh Page',
                action: () => window.location.reload(),
                style: 'primary'
              }
            ]
          };
        } else {
          notificationData = {
            type: 'warning',
            title: 'Component Error',
            message: 'A component failed to load properly.',
            duration: 5000
          };
        }
        break;

      default:
        notificationData = {
          type: 'error',
          title: 'Application Error',
          message: severity === ErrorSeverity.CRITICAL 
            ? 'A critical error occurred. Please refresh the page.'
            : 'An error occurred while processing your request.',
          duration: severity === ErrorSeverity.CRITICAL ? undefined : 5000,
          persistent: severity === ErrorSeverity.CRITICAL
        };
    }

    showNotification(notificationData);
  };

  const showNotification = (notification: Omit<Notification, 'id' | 'createdAt'>): string => {
    const id = `notification_${Date.now()}_${Math.random().toString(36).substring(2)}`;
    const newNotification: Notification = {
      ...notification,
      id,
      createdAt: new Date().toISOString()
    };

    setNotifications(prev => {
      const updated = [newNotification, ...prev].slice(0, maxNotifications);
      return updated;
    });

    // Auto-dismiss non-persistent notifications
    if (!notification.persistent && notification.duration !== undefined) {
      setTimeout(() => {
        dismissNotification(id);
      }, notification.duration);
    }

    return id;
  };

  const dismissNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  return (
    <NotificationContext.Provider value={{ 
      notifications, 
      showNotification, 
      dismissNotification, 
      clearAll 
    }}>
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  );
};

const NotificationContainer: React.FC = () => {
  const { notifications, dismissNotification } = useNotifications();

  return (
    <div className="fixed top-4 right-4 z-50 space-y-3 pointer-events-none">
      <AnimatePresence>
        {notifications.map((notification) => (
          <NotificationToast
            key={notification.id}
            notification={notification}
            onDismiss={() => dismissNotification(notification.id)}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

interface NotificationToastProps {
  notification: Notification;
  onDismiss: () => void;
}

const NotificationToast: React.FC<NotificationToastProps> = ({ notification, onDismiss }) => {
  const getIcon = () => {
    switch (notification.type) {
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'info':
        return <Info className="w-5 h-5 text-blue-400" />;
      case 'network':
        return <WifiOff className="w-5 h-5 text-orange-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  const getColorClasses = () => {
    switch (notification.type) {
      case 'error':
        return 'border-red-500/30 bg-red-500/10 text-red-400';
      case 'warning':
        return 'border-yellow-500/30 bg-yellow-500/10 text-yellow-400';
      case 'success':
        return 'border-green-500/30 bg-green-500/10 text-green-400';
      case 'info':
        return 'border-blue-500/30 bg-blue-500/10 text-blue-400';
      case 'network':
        return 'border-orange-500/30 bg-orange-500/10 text-orange-400';
      default:
        return 'border-slate-500/30 bg-slate-500/10 text-slate-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 300, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 300, scale: 0.9 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`pointer-events-auto max-w-sm w-full border rounded-lg shadow-lg backdrop-blur-xl ${getColorClasses()}`}
    >
      <div className="p-4">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0 mt-0.5">
            {getIcon()}
          </div>
          
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-white mb-1">
              {notification.title}
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              {notification.message}
            </p>
            
            {notification.actions && notification.actions.length > 0 && (
              <div className="flex space-x-2 mt-3">
                {notification.actions.map((action, index) => (
                  <button
                    key={index}
                    onClick={action.action}
                    className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
                      action.style === 'primary'
                        ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 border border-blue-500/40'
                        : action.style === 'danger'
                        ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/40'
                        : 'bg-slate-500/20 text-slate-300 hover:bg-slate-500/30 border border-slate-500/40'
                    }`}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={onDismiss}
            className="flex-shrink-0 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );
};

// Status indicator component for network status
export const NetworkStatusIndicator: React.FC = () => {
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({ online: true });
  const [activeCalls, setActiveCalls] = useState(0);

  useEffect(() => {
    const unsubscribeNetwork = networkErrorHandler.onNetworkStatusChange(setNetworkStatus);
    const unsubscribeCall = networkErrorHandler.onCallStatusChange((call) => {
      setActiveCalls(networkErrorHandler.getActiveCalls().length);
    });

    // Initial state
    setNetworkStatus(networkErrorHandler.getNetworkStatus());
    setActiveCalls(networkErrorHandler.getActiveCalls().length);

    return () => {
      unsubscribeNetwork();
      unsubscribeCall();
    };
  }, []);

  const getConnectionSpeed = () => {
    const speed = networkErrorHandler.getConnectionSpeed();
    switch (speed) {
      case 'fast': return 'text-green-400';
      case 'moderate': return 'text-yellow-400';
      case 'slow': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  if (!networkStatus.online) {
    return (
      <div className="flex items-center space-x-2 px-3 py-1 bg-red-500/10 border border-red-500/30 rounded-full">
        <WifiOff className="w-4 h-4 text-red-400" />
        <span className="text-xs text-red-400 font-medium">Offline</span>
      </div>
    );
  }

  return (
    <div className="flex items-center space-x-2 px-3 py-1 bg-slate-500/10 border border-slate-500/30 rounded-full">
      <Wifi className={`w-4 h-4 ${getConnectionSpeed()}`} />
      <span className="text-xs text-slate-300 font-medium">
        Online
      </span>
      {activeCalls > 0 && (
        <>
          <div className="w-1 h-1 bg-slate-500 rounded-full" />
          <Activity className="w-3 h-3 text-blue-400 animate-pulse" />
          <span className="text-xs text-blue-400">{activeCalls}</span>
        </>
      )}
    </div>
  );
};

// Helper functions for common notification patterns
export const showErrorNotification = (title: string, message: string, actions?: NotificationAction[]) => {
  const { showNotification } = useNotifications();
  return showNotification({
    type: 'error',
    title,
    message,
    duration: 8000,
    actions
  });
};

export const showSuccessNotification = (title: string, message: string) => {
  const { showNotification } = useNotifications();
  return showNotification({
    type: 'success',
    title,
    message,
    duration: 4000
  });
};

export const showWarningNotification = (title: string, message: string) => {
  const { showNotification } = useNotifications();
  return showNotification({
    type: 'warning',
    title,
    message,
    duration: 6000
  });
};

export const showInfoNotification = (title: string, message: string) => {
  const { showNotification } = useNotifications();
  return showNotification({
    type: 'info',
    title,
    message,
    duration: 5000
  });
};