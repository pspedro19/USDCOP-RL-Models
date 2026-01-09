'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon?: LucideIcon;
  trend?: number;
  trendLabel?: string;
  variant?: 'default' | 'compact' | 'large';
  className?: string;
  iconColor?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  icon: Icon,
  trend,
  trendLabel,
  variant = 'default',
  className = '',
  iconColor = 'text-blue-400'
}) => {
  const variantClasses = {
    default: 'p-3',
    compact: 'p-2',
    large: 'p-4'
  };

  const valueClasses = {
    default: 'text-lg',
    compact: 'text-base',
    large: 'text-xl'
  };

  const labelClasses = {
    default: 'text-sm',
    compact: 'text-xs',
    large: 'text-base'
  };

  return (
    <div
      className={`flex items-center justify-between ${variantClasses[variant]} bg-slate-800/50 rounded-lg border border-slate-700/50 ${className}`}
    >
      <div className="flex items-center gap-3">
        {Icon && <Icon className={`w-5 h-5 ${iconColor}`} />}
        <div>
          <div className={`${labelClasses[variant]} text-slate-400`}>{label}</div>
          <div className={`${valueClasses[variant]} font-semibold text-white`}>
            {value}
            {unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
          </div>
        </div>
      </div>
      {trend !== undefined && (
        <div className={`text-sm font-medium ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {trend >= 0 ? '+' : ''}{typeof trend === 'number' ? trend.toFixed(1) : trend}%
          {trendLabel && <span className="text-slate-500 ml-1 text-xs">{trendLabel}</span>}
        </div>
      )}
    </div>
  );
};

export default MetricCard;
