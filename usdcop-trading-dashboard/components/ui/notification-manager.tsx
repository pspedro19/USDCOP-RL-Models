'use client';

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import NotificationItem, { Notification } from './alert-notification';
import { getWebSocketManager } from '@/lib/services/websocket-manager';

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
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
  defaultDuration?: number;
}

export const NotificationProvider = ({
  children,
  maxNotifications = 5,
  defaultDuration = 5000
}: NotificationProviderProps) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [audioEnabled, setAudioEnabled] = useState(true);

  // Play notification sound
  const playSound = useCallback(() => {
    if (!audioEnabled) return;

    // Only play sound in browser environment
    if (typeof window === 'undefined') return;

    try {
      // Create a simple beep sound
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 800;
      oscillator.type = 'sine';

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.2);
    } catch (error) {
      console.error('[NotificationManager] Failed to play sound:', error);
    }
  }, [audioEnabled]);

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      duration: notification.duration ?? defaultDuration
    };

    setNotifications(prev => {
      const updated = [newNotification, ...prev];
      // Keep only max notifications
      return updated.slice(0, maxNotifications);
    });

    // Play sound for important notifications
    if (['signal', 'trade', 'alert', 'error'].includes(notification.type)) {
      playSound();
    }

    // Show browser notification if permitted (only in browser)
    if (typeof window !== 'undefined' && typeof Notification !== 'undefined') {
      if ('Notification' in window && Notification.permission === 'granted') {
        try {
          new Notification(notification.title, {
            body: notification.message,
            icon: '/favicon.ico',
            tag: newNotification.id,
            requireInteraction: false
          });
        } catch (err) {
          console.warn('[NotificationManager] Failed to show browser notification:', err);
        }
      }
    }
  }, [defaultDuration, maxNotifications, playSound]);

  const dismissNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  // Request browser notification permission on mount
  useEffect(() => {
    // Only run in browser environment
    if (typeof window === 'undefined') return;

    // Check if Notification API is available
    if (typeof Notification !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
          console.log('[NotificationManager] Browser notification permission:', permission);
        }).catch(err => {
          console.warn('[NotificationManager] Failed to request notification permission:', err);
        });
      }
    }
  }, []);

  // Setup WebSocket listeners for automatic notifications
  useEffect(() => {
    const wsManager = getWebSocketManager();

    // Signal notifications
    const handleSignal = (signal: any) => {
      addNotification({
        type: 'signal',
        title: `New ${signal.signal.toUpperCase()} Signal`,
        message: `${signal.strategy_name}: ${signal.reasoning?.slice(0, 100) || 'New trading signal generated'}`,
        strategy: signal.strategy_code,
        signal: signal.signal,
        duration: 8000
      });
    };

    // Trade notifications
    const handleTrade = (trade: any) => {
      addNotification({
        type: 'trade',
        title: `Trade ${trade.status === 'open' ? 'Opened' : 'Closed'}`,
        message: `${trade.strategy_code}: ${trade.side.toUpperCase()} at $${trade.entry_price.toFixed(2)}${trade.pnl ? ` | P&L: $${trade.pnl.toFixed(2)}` : ''}`,
        strategy: trade.strategy_code,
        duration: 6000
      });
    };

    wsManager.on('signals', handleSignal);
    wsManager.on('trades', handleTrade);

    return () => {
      wsManager.off('signals', handleSignal);
      wsManager.off('trades', handleTrade);
    };
  }, [addNotification]);

  return (
    <NotificationContext.Provider
      value={{ notifications, addNotification, dismissNotification, clearAll }}
    >
      {children}

      {/* Notification Container (Fixed top-right) */}
      <div className="fixed top-4 right-4 z-50 space-y-2 pointer-events-none">
        <div className="pointer-events-auto">
          <AnimatePresence>
            {notifications.map(notification => (
              <div key={notification.id} className="mb-2">
                <NotificationItem
                  notification={notification}
                  onDismiss={dismissNotification}
                />
              </div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </NotificationContext.Provider>
  );
};
