'use client';

import React from 'react';
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, HelpCircle } from 'lucide-react';

export type StatusType = 'pass' | 'fail' | 'warning' | 'loading' | 'unknown' | 'success' | 'error' | 'info';

interface StatusBadgeProps {
  status: StatusType;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
}

const statusVariants: Record<StatusType, { bg: string; text: string; border: string; icon: React.ElementType }> = {
  pass: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/50', icon: CheckCircle },
  success: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/50', icon: CheckCircle },
  fail: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', icon: XCircle },
  error: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', icon: XCircle },
  warning: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/50', icon: AlertTriangle },
  loading: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50', icon: RefreshCw },
  info: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50', icon: HelpCircle },
  unknown: { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/50', icon: HelpCircle }
};

const sizeClasses = {
  sm: { badge: 'px-2 py-0.5', icon: 'w-3 h-3', text: 'text-xs' },
  md: { badge: 'px-3 py-1', icon: 'w-4 h-4', text: 'text-sm' },
  lg: { badge: 'px-4 py-1.5', icon: 'w-5 h-5', text: 'text-base' }
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
  showIcon = true,
  className = ''
}) => {
  const variant = statusVariants[status] || statusVariants.unknown;
  const sizes = sizeClasses[size];
  const Icon = variant.icon;
  const displayLabel = label || status;

  return (
    <div
      className={`inline-flex items-center gap-2 ${sizes.badge} rounded-full border ${variant.bg} ${variant.border} ${className}`}
    >
      {showIcon && (
        <Icon
          className={`${sizes.icon} ${variant.text} ${status === 'loading' ? 'animate-spin' : ''}`}
        />
      )}
      <span className={`${sizes.text} font-medium ${variant.text} uppercase`}>
        {displayLabel}
      </span>
    </div>
  );
};

export default StatusBadge;
