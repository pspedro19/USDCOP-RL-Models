import * as React from "react";
import { motion, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import { motionLibrary } from "@/lib/motion";

export interface ButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onAnimationStart' | 'onDrag' | 'onDragEnd' | 'onDragStart'>,
    MotionProps {
  variant?: 'default' | 'outline' | 'secondary' | 'ghost' | 'link' | 'destructive' | 'gradient' | 'glow' | 'glass' | 'glass-primary' | 'glass-secondary' | 'glass-accent' | 'professional' | 'terminal';
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'xs';
  asChild?: boolean;
  glow?: boolean;
  animated?: boolean;
  loading?: boolean;
  pulse?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant = 'default', 
    size = 'default', 
    glow = false, 
    animated = true, 
    loading = false,
    pulse = false,
    asChild = false, 
    children,
    ...props 
  }, ref) => {
    const baseClasses = cn(
      "inline-flex items-center justify-center whitespace-nowrap text-sm font-medium relative overflow-hidden",
      "focus-visible:outline-none glass-focus disabled:pointer-events-none disabled:opacity-50",
      "transition-all duration-300 cubic-bezier(0.4, 0, 0.2, 1)",
      glow && "shadow-glow-cyan",
      loading && "cursor-wait",
      pulse && "animate-pulse"
    );
    
    const variantClasses = {
      default: "bg-slate-800 text-slate-100 hover:bg-slate-700 border border-slate-600/50 hover:border-cyan-400/50 shadow-lg hover:shadow-cyan-400/20",
      destructive: "bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800 shadow-lg shadow-red-500/20 hover:shadow-red-500/40",
      outline: "border border-slate-600/50 bg-transparent text-slate-300 hover:bg-slate-800/50 hover:text-white hover:border-cyan-400/50",
      secondary: "bg-slate-700/60 text-slate-200 hover:bg-slate-600/60 hover:text-white border border-slate-600/30 hover:border-slate-500/50",
      ghost: "text-slate-400 hover:bg-slate-800/50 hover:text-white rounded-lg",
      link: "text-cyan-400 underline-offset-4 hover:underline hover:text-cyan-300",
      gradient: "bg-gradient-to-r from-cyan-500 to-emerald-500 text-white hover:from-cyan-600 hover:to-emerald-600 shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50",
      glow: "bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600 shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-r before:from-white/0 before:via-white/20 before:to-white/0 before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-700",
      glass: "glass-surface-secondary hover:glass-card-interactive",
      
      // Professional Bloomberg Terminal Button Variants
      'glass-primary': cn(
        "glass-button-primary backdrop-blur-professional",
        "before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/10 before:to-transparent",
        "before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-500",
        "after:absolute after:inset-0 after:rounded-inherit after:border after:border-transparent",
        "hover:after:border-cyan-400/30 after:transition-colors after:duration-300"
      ),
      
      'glass-secondary': cn(
        "glass-button-secondary backdrop-blur-md",
        "bg-gradient-to-br from-slate-800/60 via-slate-700/40 to-slate-900/80",
        "border border-slate-600/30 hover:border-emerald-400/50",
        "shadow-glass-md hover:shadow-glass-lg hover:shadow-emerald-500/20",
        "text-slate-200 hover:text-emerald-100"
      ),
      
      'glass-accent': cn(
        "bg-gradient-to-br from-amber-500/20 via-orange-500/10 to-red-500/20",
        "backdrop-blur-lg border border-amber-500/30 hover:border-amber-400/60",
        "shadow-glass-md hover:shadow-glass-lg hover:shadow-amber-500/30",
        "text-amber-100 hover:text-white",
        "hover:bg-gradient-to-br hover:from-amber-500/30 hover:via-orange-500/20 hover:to-red-500/30"
      ),
      
      'professional': cn(
        "glass-surface-elevated backdrop-blur-intense",
        "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
        "border border-cyan-500/20 hover:border-cyan-400/40",
        "shadow-glass-lg hover:shadow-glass-xl",
        "text-cyan-100 hover:text-white font-semibold",
        "hover:shadow-glow-cyan/50",
        "group relative"
      ),
      
      'terminal': cn(
        "bg-gradient-to-br from-slate-950/90 via-slate-900/80 to-slate-950/95",
        "backdrop-blur-xl border-2 border-cyan-500/30 hover:border-cyan-400/60",
        "shadow-glass-xl hover:shadow-glow-cyan",
        "text-cyan-300 hover:text-cyan-100 font-mono font-bold tracking-wide",
        "hover:bg-gradient-to-br hover:from-slate-900/95 hover:via-slate-800/85 hover:to-slate-900/98",
        "relative overflow-hidden",
        "before:absolute before:top-0 before:left-0 before:right-0 before:h-[1px]",
        "before:bg-gradient-to-r before:from-transparent before:via-cyan-400 before:to-transparent",
        "after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[1px]",
        "after:bg-gradient-to-r after:from-transparent after:via-cyan-400/50 after:to-transparent"
      ),
    };
    
    const sizeClasses = {
      xs: "h-7 px-2 text-xs rounded-xl",
      default: "h-10 px-4 py-2 rounded-xl",
      sm: "h-9 px-3 rounded-xl", 
      lg: "h-12 px-8 text-base rounded-2xl",
      icon: "h-10 w-10 rounded-xl",
    };
    
    const classes = cn(
      baseClasses,
      variantClasses[variant],
      sizeClasses[size],
      className
    );

    const MotionButton = motion.button;
    
    const buttonVariants = animated ? {
      ...motionLibrary.components.glassButton,
      initial: {
        scale: 1,
        ...(glow && { boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)' })
      }
    } : {};

    const buttonContent = (
      <>
        {loading && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            variants={motionLibrary.loading.spinner}
            animate="animate"
          >
            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
          </motion.div>
        )}
        <motion.div
          className={cn(
            "flex items-center justify-center gap-2 transition-opacity duration-200",
            loading && "opacity-0"
          )}
          variants={animated ? {
            hover: { scale: 1.02 },
            tap: { scale: 0.98 }
          } : {}}
        >
          {children}
        </motion.div>
        
        {/* Shimmer effect for glass buttons */}
        {variant.includes('glass') && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
            initial={{ x: '-100%' }}
            whileHover={{ x: '100%' }}
            transition={{ duration: 0.6, ease: 'easeInOut' }}
          />
        )}
        
        {/* Pulse ring for pulse variant */}
        {pulse && (
          <motion.div
            className="absolute inset-0 rounded-inherit border-2 border-current"
            animate={{
              scale: [1, 1.05, 1],
              opacity: [0.5, 0, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        )}
      </>
    );

    if (asChild) {
      return (
        <motion.button
          ref={ref}
          className={classes}
          variants={buttonVariants}
          {...(animated && {
            initial: "initial",
            whileHover: "hover",
            whileTap: "tap"
          })}
          disabled={loading || props.disabled}
          {...props}
        >
          {buttonContent}
        </motion.button>
      );
    }
    
    return (
      <MotionButton
        ref={ref}
        className={classes}
        variants={buttonVariants}
        {...(animated && {
          initial: "initial",
          whileHover: "hover",
          whileTap: "tap"
        })}
        disabled={loading || props.disabled}
        {...props}
      >
        {buttonContent}
      </MotionButton>
    );
  }
);

Button.displayName = "Button";

export { Button };