import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'optimal' | 'warning' | 'critical';
  icon: React.ComponentType<any>;
  trend?: 'up' | 'down' | 'neutral';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title, value, subtitle, status, icon: Icon, trend
}) => {
  const statusColors = {
    optimal: 'border-market-up text-market-up shadow-market-up',
    warning: 'border-fintech-purple-400 text-fintech-purple-400 shadow-glow-purple',
    critical: 'border-market-down text-market-down shadow-market-down'
  };

  const bgColors = {
    optimal: 'from-market-up/10 to-market-up/5',
    warning: 'from-fintech-purple-400/10 to-fintech-purple-400/5',
    critical: 'from-market-down/10 to-market-down/5'
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-6 rounded-xl border ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        {trend && (
          <div className={`${trend === 'up' ? 'text-market-up' : trend === 'down' ? 'text-market-down' : 'text-fintech-dark-400'}`}>
            {trend === 'up' && <TrendingUp className="w-4 h-4" />}
            {trend === 'down' && <TrendingDown className="w-4 h-4" />}
          </div>
        )}
      </div>

      <div className="text-2xl font-bold text-white mb-1">
        {typeof value === 'number' ? value.toFixed(2) : value}
      </div>
      <div className="text-xs text-fintech-dark-300">{subtitle}</div>
    </motion.div>
  );
};
