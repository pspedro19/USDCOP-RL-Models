'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

interface SliderProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number | [number, number];
  onValueChange?: (value: number | [number, number]) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

export const Slider = React.forwardRef<HTMLDivElement, SliderProps>(
  ({ className, value, onValueChange, min = 0, max = 100, step = 1, disabled = false, ...props }, ref) => {
    const isRange = Array.isArray(value);
    const [localValue, setLocalValue] = React.useState(value);
    const [isDragging, setIsDragging] = React.useState<'start' | 'end' | null>(null);
    const sliderRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
      setLocalValue(value);
    }, [value]);

    const handleMouseDown = (e: React.MouseEvent, thumb: 'start' | 'end') => {
      if (disabled) return;
      e.preventDefault();
      setIsDragging(thumb);
    };

    const handleMouseMove = React.useCallback((e: MouseEvent) => {
      if (!isDragging || !sliderRef.current || disabled) return;

      const rect = sliderRef.current.getBoundingClientRect();
      const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const newValue = Math.round((min + percent * (max - min)) / step) * step;

      if (isRange && Array.isArray(localValue)) {
        const [start, end] = localValue;
        if (isDragging === 'start') {
          const newStart = Math.min(newValue, end);
          const newRange: [number, number] = [newStart, end];
          setLocalValue(newRange);
          onValueChange?.(newRange);
        } else {
          const newEnd = Math.max(newValue, start);
          const newRange: [number, number] = [start, newEnd];
          setLocalValue(newRange);
          onValueChange?.(newRange);
        }
      } else {
        setLocalValue(newValue);
        onValueChange?.(newValue);
      }
    }, [isDragging, min, max, step, isRange, localValue, onValueChange, disabled]);

    const handleMouseUp = React.useCallback(() => {
      setIsDragging(null);
    }, []);

    React.useEffect(() => {
      if (isDragging) {
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
          document.removeEventListener('mousemove', handleMouseMove);
          document.removeEventListener('mouseup', handleMouseUp);
        };
      }
    }, [isDragging, handleMouseMove, handleMouseUp]);

    const getPercentage = (val: number) => ((val - min) / (max - min)) * 100;

    return (
      <div
        ref={ref}
        className={cn(
          'relative flex w-full touch-none select-none items-center',
          disabled && 'opacity-50 cursor-not-allowed',
          className
        )}
        {...props}
      >
        <div
          ref={sliderRef}
          className="relative h-3 w-full grow overflow-hidden rounded-full bg-slate-800/60 border border-slate-700/50 shadow-inner"
        >
          {/* Animated track background */}
          <div className="absolute inset-0 bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 animate-pulse opacity-30" />
          
          {/* Main track fill */}
          <div
            className="absolute h-full bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 transition-all duration-300 ease-out relative overflow-hidden"
            style={{
              left: isRange ? `${getPercentage((localValue as [number, number])[0])}%` : '0%',
              right: isRange 
                ? `${100 - getPercentage((localValue as [number, number])[1])}%` 
                : `${100 - getPercentage(localValue as number)}%`
            }}
          >
            {/* Shimmer effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent translate-x-[-100%] animate-shimmer" />
            
            {/* Glow effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 blur-sm opacity-60 -z-10" />
          </div>
        </div>
        
        {isRange && Array.isArray(localValue) ? (
          <>
            {/* Start thumb with enhanced styling */}
            <button
              className={cn(
                'absolute h-5 w-5 rounded-full border-2 transition-all duration-200 transform hover:scale-110 active:scale-95',
                'bg-gradient-to-br from-cyan-400 to-emerald-400 border-white shadow-lg shadow-cyan-400/50',
                'hover:shadow-xl hover:shadow-cyan-400/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-slate-900',
                isDragging === 'start' && 'scale-110 shadow-xl shadow-cyan-400/70',
                disabled && 'pointer-events-none opacity-50'
              )}
              style={{ left: `${getPercentage(localValue[0])}%`, transform: 'translateX(-50%)' }}
              onMouseDown={(e) => handleMouseDown(e, 'start')}
              disabled={disabled}
            >
              {/* Inner glow */}
              <div className="absolute inset-0.5 rounded-full bg-gradient-to-br from-cyan-300 to-emerald-300 animate-pulse opacity-60" />
            </button>
            
            {/* End thumb with enhanced styling */}
            <button
              className={cn(
                'absolute h-5 w-5 rounded-full border-2 transition-all duration-200 transform hover:scale-110 active:scale-95',
                'bg-gradient-to-br from-purple-400 to-pink-400 border-white shadow-lg shadow-purple-400/50',
                'hover:shadow-xl hover:shadow-purple-400/60 focus:outline-none focus:ring-2 focus:ring-purple-400/50 focus:ring-offset-2 focus:ring-offset-slate-900',
                isDragging === 'end' && 'scale-110 shadow-xl shadow-purple-400/70',
                disabled && 'pointer-events-none opacity-50'
              )}
              style={{ left: `${getPercentage(localValue[1])}%`, transform: 'translateX(-50%)' }}
              onMouseDown={(e) => handleMouseDown(e, 'end')}
              disabled={disabled}
            >
              {/* Inner glow */}
              <div className="absolute inset-0.5 rounded-full bg-gradient-to-br from-purple-300 to-pink-300 animate-pulse opacity-60" />
            </button>
          </>
        ) : (
          <button
            className={cn(
              'absolute h-5 w-5 rounded-full border-2 transition-all duration-200 transform hover:scale-110 active:scale-95',
              'bg-gradient-to-br from-cyan-400 to-emerald-400 border-white shadow-lg shadow-cyan-400/50',
              'hover:shadow-xl hover:shadow-cyan-400/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-slate-900',
              isDragging && 'scale-110 shadow-xl shadow-cyan-400/70',
              disabled && 'pointer-events-none opacity-50'
            )}
            style={{ left: `${getPercentage(localValue as number)}%`, transform: 'translateX(-50%)' }}
            onMouseDown={(e) => handleMouseDown(e, 'end')}
            disabled={disabled}
          >
            {/* Inner glow */}
            <div className="absolute inset-0.5 rounded-full bg-gradient-to-br from-cyan-300 to-emerald-300 animate-pulse opacity-60" />
          </button>
        )}
      </div>
    );
  }
);

Slider.displayName = 'Slider';