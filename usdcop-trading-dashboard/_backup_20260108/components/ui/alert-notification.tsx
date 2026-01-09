'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Bell, TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';

export interface Notification {
  id: string;
  type: 'signal' | 'trade' | 'alert' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  duration?: number; // Auto-dismiss after X ms
  strategy?: string;
  signal?: 'long' | 'short' | 'flat';
}

interface NotificationProps {
  notification: Notification;
  onDismiss: (id: string) => void;
}

const NotificationItem = ({ notification, onDismiss }: NotificationProps) => {
  useEffect(() => {
    if (notification.duration) {
      const timer = setTimeout(() => {
        onDismiss(notification.id);
      }, notification.duration);

      return () => clearTimeout(timer);
    }
  }, [notification.id, notification.duration, onDismiss]);

  const getIcon = () => {
    switch (notification.type) {
      case 'signal':
        return notification.signal === 'long' ? (
          <TrendingUp className="h-5 w-5 text-green-400" />
        ) : notification.signal === 'short' ? (
          <TrendingDown className="h-5 w-5 text-red-400" />
        ) : (
          <Bell className="h-5 w-5 text-yellow-400" />
        );
      case 'trade':
        return <CheckCircle className="h-5 w-5 text-cyan-400" />;
      case 'alert':
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-400" />;
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      default:
        return <Bell className="h-5 w-5 text-cyan-400" />;
    }
  };

  const getColor = () => {
    switch (notification.type) {
      case 'signal':
        return notification.signal === 'long'
          ? 'bg-green-500/10 border-green-500/30'
          : notification.signal === 'short'
            ? 'bg-red-500/10 border-red-500/30'
            : 'bg-yellow-500/10 border-yellow-500/30';
      case 'trade':
        return 'bg-cyan-500/10 border-cyan-500/30';
      case 'success':
        return 'bg-green-500/10 border-green-500/30';
      case 'warning':
      case 'alert':
        return 'bg-yellow-500/10 border-yellow-500/30';
      case 'error':
        return 'bg-red-500/10 border-red-500/30';
      default:
        return 'bg-slate-700/10 border-slate-500/30';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 300, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 300, scale: 0.9 }}
      className={`w-full max-w-sm bg-slate-900/95 backdrop-blur border ${getColor()} rounded-lg shadow-2xl p-4`}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <p className="text-white font-mono font-bold text-sm">
                {notification.title}
              </p>
              {notification.strategy && (
                <p className="text-slate-400 text-xs font-mono mt-0.5">
                  {notification.strategy}
                </p>
              )}
            </div>
            <button
              onClick={() => onDismiss(notification.id)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <p className="text-slate-300 text-xs font-mono mt-2 line-clamp-2">
            {notification.message}
          </p>

          <p className="text-slate-500 text-xs font-mono mt-2">
            {notification.timestamp.toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Progress Bar for auto-dismiss */}
      {notification.duration && (
        <motion.div
          className="h-1 bg-cyan-500/30 rounded-full mt-3"
          initial={{ width: '100%' }}
          animate={{ width: '0%' }}
          transition={{ duration: notification.duration / 1000, ease: 'linear' }}
        />
      )}
    </motion.div>
  );
};

export default NotificationItem;
