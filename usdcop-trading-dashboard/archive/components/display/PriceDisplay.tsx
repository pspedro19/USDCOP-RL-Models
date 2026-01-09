/**
 * Price Display Component
 * Shows price with semantic coloring and smooth animations
 */

import React, { useEffect, useRef, useState } from 'react';
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface PriceDisplayProps {
  value: number;
  previousValue?: number;
  currency?: string;
  decimals?: number;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showChange?: boolean;
  showPercentage?: boolean;
  showTrend?: boolean;
  animate?: boolean;
  highlightDuration?: number;
  className?: string;
}

export const PriceDisplay: React.FC<PriceDisplayProps> = ({
  value,
  previousValue,
  currency = 'COP',
  decimals = 2,
  size = 'md',
  showChange = true,
  showPercentage = true,
  showTrend = true,
  animate = true,
  highlightDuration = 600,
  className
}) => {
  const [displayValue, setDisplayValue] = useState(value);
  const [isAnimating, setIsAnimating] = useState(false);
  const [changeDirection, setChangeDirection] = useState<'up' | 'down' | 'neutral'>('neutral');
  const previousValueRef = useRef(value);
  const animationTimeoutRef = useRef<NodeJS.Timeout>();
  
  // Calculate change metrics
  const change = previousValue ? value - previousValue : 0;
  const changePercent = previousValue ? (change / previousValue) * 100 : 0;
  
  // Handle value changes with animation
  useEffect(() => {
    if (animate && previousValueRef.current !== value) {
      const direction = value > previousValueRef.current ? 'up' : 
                       value < previousValueRef.current ? 'down' : 'neutral';
      
      setChangeDirection(direction);
      setIsAnimating(true);
      
      // Clear existing timeout
      if (animationTimeoutRef.current) {
        clearTimeout(animationTimeoutRef.current);
      }
      
      // Animate the number change
      if (animate && direction !== 'neutral') {
        animateValue(previousValueRef.current, value, 300);
      } else {
        setDisplayValue(value);
      }
      
      // Reset animation state after duration
      animationTimeoutRef.current = setTimeout(() => {
        setIsAnimating(false);
      }, highlightDuration);
      
      previousValueRef.current = value;
    } else {
      setDisplayValue(value);
    }
    
    return () => {
      if (animationTimeoutRef.current) {
        clearTimeout(animationTimeoutRef.current);
      }
    };
  }, [value, animate, highlightDuration]);
  
  // Smooth number animation
  const animateValue = (start: number, end: number, duration: number) => {
    const startTime = performance.now();
    const diff = end - start;
    
    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function for smooth animation
      const easeOutQuad = (t: number) => t * (2 - t);
      const currentValue = start + diff * easeOutQuad(progress);
      
      setDisplayValue(currentValue);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  };
  
  // Format price for display
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(price);
  };
  
  // Format percentage
  const formatPercent = (percent: number) => {
    const prefix = percent > 0 ? '+' : '';
    return `${prefix}${percent.toFixed(2)}%`;
  };
  
  // Get size classes
  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return {
          price: 'text-lg',
          change: 'text-xs',
          icon: 12
        };
      case 'md':
        return {
          price: 'text-2xl',
          change: 'text-sm',
          icon: 14
        };
      case 'lg':
        return {
          price: 'text-3xl',
          change: 'text-base',
          icon: 16
        };
      case 'xl':
        return {
          price: 'text-4xl',
          change: 'text-lg',
          icon: 20
        };
      default:
        return {
          price: 'text-2xl',
          change: 'text-sm',
          icon: 14
        };
    }
  };
  
  const sizeClasses = getSizeClasses();
  
  // Get trend icon
  const getTrendIcon = () => {
    if (changeDirection === 'up') {
      return showTrend ? <TrendingUp size={sizeClasses.icon} /> : <ArrowUp size={sizeClasses.icon} />;
    }
    if (changeDirection === 'down') {
      return showTrend ? <TrendingDown size={sizeClasses.icon} /> : <ArrowDown size={sizeClasses.icon} />;
    }
    return <Minus size={sizeClasses.icon} />;
  };
  
  return (
    <div className={cn('inline-flex flex-col', className)}>
      {/* Main Price Display */}
      <div className={cn(
        'font-mono font-bold transition-all duration-300',
        sizeClasses.price,
        isAnimating && changeDirection === 'up' && 'price-update-up',
        isAnimating && changeDirection === 'down' && 'price-update-down',
        changeDirection === 'up' ? 'text-market-up' : 
        changeDirection === 'down' ? 'text-market-down' : 
        'text-terminal-text'
      )}>
        {formatPrice(displayValue)}
      </div>
      
      {/* Change Display */}
      {showChange && previousValue && (
        <div className={cn(
          'flex items-center space-x-2 font-mono transition-all duration-300',
          sizeClasses.change,
          change >= 0 ? 'text-positive' : 'text-negative'
        )}>
          <span className="flex items-center space-x-1">
            {getTrendIcon()}
            <span className="font-semibold">
              {formatPrice(Math.abs(change))}
            </span>
          </span>
          
          {showPercentage && (
            <span className={cn(
              'px-2 py-0.5 rounded-md',
              change >= 0 ? 'bg-up-10' : 'bg-down-10'
            )}>
              {formatPercent(changePercent)}
            </span>
          )}
        </div>
      )}
      
      {/* Visual indicator for large changes */}
      {Math.abs(changePercent) > 1 && animate && (
        <div className={cn(
          'h-1 mt-1 rounded-full transition-all duration-1000',
          changePercent > 0 ? 'bg-gradient-positive' : 'bg-gradient-negative',
          isAnimating ? 'opacity-100 w-full' : 'opacity-0 w-0'
        )} />
      )}
    </div>
  );
};

// Compact version for use in tables or lists
export const PriceDisplayCompact: React.FC<{
  value: number;
  previousValue?: number;
  showArrow?: boolean;
  className?: string;
}> = ({ value, previousValue, showArrow = true, className }) => {
  const change = previousValue ? value - previousValue : 0;
  const isUp = change > 0;
  const isDown = change < 0;
  
  return (
    <div className={cn(
      'inline-flex items-center space-x-1 font-mono',
      isUp && 'text-positive',
      isDown && 'text-negative',
      !isUp && !isDown && 'text-terminal-text',
      className
    )}>
      {showArrow && change !== 0 && (
        <span className="text-xs">
          {isUp ? '▲' : '▼'}
        </span>
      )}
      <span className="font-semibold">
        {new Intl.NumberFormat('es-CO', {
          style: 'currency',
          currency: 'COP',
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        }).format(value)}
      </span>
    </div>
  );
};

export default PriceDisplay;