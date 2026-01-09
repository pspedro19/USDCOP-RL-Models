import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle } from 'lucide-react';

interface ComplianceMetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'compliant' | 'warning' | 'non-compliant';
  icon: React.ComponentType<any>;
  format?: 'percentage' | 'ratio' | 'currency' | 'count' | 'text';
}

export const ComplianceMetricCard: React.FC<ComplianceMetricCardProps> = ({
  title, value, subtitle, status, icon: Icon, format = 'text'
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;

    switch (format) {
      case 'percentage':
        return `${val.toFixed(1)}%`;
      case 'ratio':
        return val.toFixed(3);
      case 'currency':
        return `$${val.toLocaleString()}`;
      case 'count':
        return val.toLocaleString();
      default:
        return val.toString();
    }
  };

  const statusColors = {
    compliant: 'border-market-up text-market-up shadow-market-up',
    warning: 'border-fintech-purple-400 text-fintech-purple-400 shadow-glow-purple',
    'non-compliant': 'border-market-down text-market-down shadow-market-down'
  };

  const bgColors = {
    compliant: 'from-market-up/10 to-market-up/5',
    warning: 'from-fintech-purple-400/10 to-fintech-purple-400/5',
    'non-compliant': 'from-market-down/10 to-market-down/5'
  };

  const statusIcons = {
    compliant: CheckCircle,
    warning: AlertTriangle,
    'non-compliant': AlertTriangle
  };

  const StatusIcon = statusIcons[status];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-4 rounded-xl border ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        <StatusIcon className="w-5 h-5" />
      </div>

      <div className="space-y-2">
        <div className="text-2xl font-bold text-white">
          {formatValue(value)}
        </div>
        <div className="text-xs text-fintech-dark-300">{subtitle}</div>
      </div>
    </motion.div>
  );
};
