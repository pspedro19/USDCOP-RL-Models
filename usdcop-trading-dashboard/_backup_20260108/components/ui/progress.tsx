"use client"

import * as React from "react"
import * as ProgressPrimitive from "@radix-ui/react-progress"
import { cn } from "@/lib/utils"

interface EnhancedProgressProps extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  variant?: 'default' | 'gradient' | 'shimmer' | 'glow';
  indicatorClassName?: string;
}

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  EnhancedProgressProps
>(({ className, value, variant = 'default', indicatorClassName, ...props }, ref) => {
  const progressValue = value || 0;
  
  const trackClasses = {
    default: "bg-slate-800/60 border border-slate-700/50",
    gradient: "bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 border border-slate-600/50",
    shimmer: "bg-slate-800/60 border border-slate-700/50 shadow-inner",
    glow: "bg-slate-900/80 border border-slate-600/50 shadow-inner shadow-cyan-500/10"
  };

  const indicatorClasses = {
    default: "bg-gradient-to-r from-cyan-500 to-emerald-500",
    gradient: "bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500",
    shimmer: "bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 relative overflow-hidden",
    glow: "bg-gradient-to-r from-cyan-400 to-emerald-400 shadow-lg shadow-cyan-400/30"
  };

  return (
    <ProgressPrimitive.Root
      ref={ref}
      className={cn(
        "relative h-3 w-full overflow-hidden rounded-full transition-all duration-300",
        trackClasses[variant],
        className
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className={cn(
          "h-full w-full flex-1 transition-all duration-500 ease-out rounded-full",
          indicatorClassName || indicatorClasses[variant]
        )}
        style={{ transform: `translateX(-${100 - progressValue}%)` }}
      >
        {variant === 'shimmer' && (
          <>
            {/* Shimmer overlay */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent translate-x-[-100%] animate-shimmer" />
            
            {/* Additional glow */}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 blur-sm opacity-50 -z-10" />
          </>
        )}
        
        {variant === 'glow' && (
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-emerald-400 animate-pulse opacity-40" />
        )}
        
        {variant === 'gradient' && (
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 via-emerald-400/20 to-purple-400/20 animate-pulse" />
        )}
      </ProgressPrimitive.Indicator>
      
      {/* Progress value indicator */}
      {progressValue > 0 && (
        <div 
          className="absolute top-1/2 transform -translate-y-1/2 -translate-x-1/2 w-2 h-2 bg-white rounded-full shadow-lg shadow-cyan-400/50 border border-cyan-400/30 transition-all duration-500 ease-out"
          style={{ left: `${progressValue}%` }}
        >
          <div className="absolute inset-0 bg-white rounded-full animate-ping opacity-60" />
        </div>
      )}
    </ProgressPrimitive.Root>
  );
})
Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }