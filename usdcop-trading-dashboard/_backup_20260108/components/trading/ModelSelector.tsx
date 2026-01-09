'use client';

/**
 * ModelSelector Component
 * ========================
 * Allows users to select which trading model to view.
 * Shows model tabs with status indicators.
 */

import React from 'react';
import { useModel, useModelComparison } from '@/contexts/ModelContext';
import { getStatusBadgeProps } from '@/lib/config/models.config';
import { cn } from '@/lib/utils';

interface ModelSelectorProps {
  className?: string;
  showComparison?: boolean;
  compact?: boolean;
}

export function ModelSelector({
  className,
  showComparison = true,
  compact = false,
}: ModelSelectorProps) {
  const { selectedModelId, selectedModel, models, isLoading, error, setSelectedModel } =
    useModel();
  const { isComparing, setIsComparing } = useModelComparison();

  if (isLoading) {
    return (
      <div className={cn('flex items-center gap-2 p-2', className)}>
        <div className="h-8 w-24 animate-pulse rounded bg-slate-700" />
        <div className="h-8 w-24 animate-pulse rounded bg-slate-700" />
        <div className="h-8 w-24 animate-pulse rounded bg-slate-700" />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex items-center gap-2 p-2 text-red-400', className)}>
        <span className="text-sm">Error loading models: {error}</span>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className={cn('flex items-center gap-2 p-2 text-slate-400', className)}>
        <span className="text-sm">No models available</span>
      </div>
    );
  }

  return (
    <div className={cn('flex items-center gap-3', className)}>
      {/* Model label */}
      {!compact && (
        <span className="text-sm font-medium text-slate-400">MODELO:</span>
      )}

      {/* Model tabs */}
      <div className="flex gap-2">
        {models.map((model) => {
          const isSelected = model.id === selectedModelId;
          const isReal = model.isRealData === true;

          return (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={cn(
                'group relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all',
                'border-2 hover:bg-opacity-20',
                isSelected
                  ? 'border-current bg-opacity-20'
                  : 'border-transparent bg-slate-800 hover:border-slate-600'
              )}
              style={{
                color: isSelected ? model.color : undefined,
                backgroundColor: isSelected ? `${model.color}20` : undefined,
              }}
            >
              {/* Selected indicator dot */}
              {isSelected && (
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: model.color }}
                />
              )}

              {/* Model name */}
              <span className={cn(!isSelected && 'text-slate-300')}>
                {compact ? model.version : model.name}
              </span>

              {/* Real Data vs Demo badge */}
              {!compact && (
                <span
                  className={cn(
                    'rounded px-1.5 py-0.5 text-xs font-semibold',
                    isReal
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-amber-500/20 text-amber-400'
                  )}
                >
                  {isReal ? 'REAL' : 'DEMO'}
                </span>
              )}

              {/* Hover tooltip for compact mode */}
              {compact && (
                <div
                  className={cn(
                    'absolute -bottom-8 left-1/2 z-50 -translate-x-1/2 transform',
                    'whitespace-nowrap rounded bg-slate-900 px-2 py-1 text-xs text-slate-300',
                    'opacity-0 transition-opacity group-hover:opacity-100',
                    'pointer-events-none border border-slate-700'
                  )}
                >
                  {model.name} {isReal ? '(Real)' : '(Demo)'}
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Comparison toggle */}
      {showComparison && models.length > 1 && (
        <button
          onClick={() => setIsComparing(!isComparing)}
          className={cn(
            'ml-auto flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-all',
            isComparing
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          )}
        >
          <span>⚖️</span>
          <span>{isComparing ? 'Cerrar' : 'Comparar'}</span>
        </button>
      )}
    </div>
  );
}

/**
 * ModelSelectorCompact - Dropdown version for narrow spaces
 */
export function ModelSelectorCompact({ className }: { className?: string }) {
  const { selectedModelId, selectedModel, models, isLoading, setSelectedModel } =
    useModel();

  if (isLoading || models.length === 0) {
    return null;
  }

  return (
    <select
      value={selectedModelId || ''}
      onChange={(e) => setSelectedModel(e.target.value)}
      className={cn(
        'rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300',
        'focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500',
        className
      )}
      style={{
        color: selectedModel?.color,
      }}
    >
      {models.map((model) => (
        <option key={model.id} value={model.id}>
          {model.name}
        </option>
      ))}
    </select>
  );
}

export default ModelSelector;
