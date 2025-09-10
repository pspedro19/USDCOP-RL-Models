import * as React from "react"
import { motion, MotionProps } from "framer-motion"
import { cn } from "@/lib/utils"
import { motionLibrary } from "@/lib/motion"

interface CardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>, MotionProps {
  variant?: 'default' | 'glass' | 'glow' | 'gradient' | 'premium' | 'terminal' | 'professional' | 'trading' | 'analytics' | 'status';
  animated?: boolean;
  glow?: boolean;
  loading?: boolean;
  hover?: boolean;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', animated = true, glow = false, loading = false, hover = true, children, ...props }, ref) => {
  const baseClasses = cn(
    "relative overflow-hidden text-slate-100",
    "transition-all duration-300 cubic-bezier(0.4, 0, 0.2, 1)",
    animated && "hover:scale-[1.01] transform",
    glow && "shadow-glow-mixed"
  );

  const variantClasses = {
    default: "rounded-xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm shadow-lg hover:shadow-xl hover:border-cyan-400/30 hover:shadow-cyan-400/10",
    
    glass: cn(
      "glass-card",
      "hover:transform hover:translateY(-1px)",
      "hover:shadow-glass-xl"
    ),
    
    glow: "rounded-xl border border-purple-400/30 bg-slate-900/90 backdrop-blur-sm shadow-2xl shadow-purple-500/10 hover:shadow-purple-500/20 transition-all duration-500 hover:border-purple-400/60 relative before:absolute before:inset-0 before:rounded-xl before:bg-gradient-to-r before:from-purple-400/10 before:via-pink-400/10 before:to-cyan-400/10 before:opacity-0 before:hover:opacity-100 before:transition-opacity before:duration-500 before:-z-10 before:blur-sm",
    
    gradient: "rounded-xl border-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 shadow-2xl hover:shadow-3xl transition-all duration-500 relative overflow-hidden before:absolute before:inset-[1px] before:rounded-xl before:bg-gradient-to-br before:from-cyan-400/20 before:via-transparent before:to-purple-400/20 before:p-[1px] after:absolute after:inset-0 after:rounded-xl after:bg-gradient-to-br after:from-slate-900 after:via-slate-800 after:to-slate-900 after:-z-10",
    
    premium: "rounded-2xl border border-gradient-to-r border-cyan-400/40 bg-slate-900/95 backdrop-blur-md shadow-2xl shadow-cyan-400/10 hover:shadow-cyan-400/20 transition-all duration-700 hover:border-emerald-400/40 hover:shadow-emerald-400/20 relative overflow-hidden group before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-400/5 before:via-emerald-400/5 before:to-purple-400/5 before:opacity-0 group-hover:before:opacity-100 before:transition-opacity before:duration-700 before:pointer-events-none",
    
    // Professional Bloomberg Terminal Variants
    terminal: cn(
      "rounded-3xl backdrop-blur-intense border-2 border-cyan-500/30",
      "bg-gradient-to-br from-slate-950/95 via-slate-900/90 to-slate-950/98",
      "shadow-glass-xl hover:shadow-glow-cyan",
      "hover:border-cyan-400/60 hover:bg-gradient-to-br hover:from-slate-900/98 hover:via-slate-800/95 hover:to-slate-950/99",
      "group relative",
      "before:absolute before:top-0 before:left-0 before:right-0 before:h-[2px]",
      "before:bg-gradient-to-r before:from-transparent before:via-cyan-400 before:to-transparent before:rounded-t-3xl",
      "after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[1px]",
      "after:bg-gradient-to-r after:from-transparent after:via-cyan-400/50 after:to-transparent after:rounded-b-3xl"
    ),
    
    professional: cn(
      "glass-surface-elevated rounded-2xl",
      "bg-gradient-to-br from-slate-900/85 via-slate-800/70 to-slate-900/90",
      "border border-cyan-500/25 hover:border-cyan-400/40",
      "shadow-glass-lg hover:shadow-glass-xl hover:shadow-cyan-500/20",
      "backdrop-blur-professional",
      "group relative overflow-hidden",
      "before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-400/8 before:via-transparent before:to-purple-400/8",
      "before:opacity-0 group-hover:before:opacity-100 before:transition-opacity before:duration-500"
    ),
    
    trading: cn(
      "rounded-2xl backdrop-blur-lg",
      "bg-gradient-to-br from-emerald-900/20 via-slate-900/80 to-cyan-900/20",
      "border border-emerald-500/30 hover:border-emerald-400/50",
      "shadow-glass-md hover:shadow-glass-lg hover:shadow-emerald-500/20",
      "relative group",
      "before:absolute before:inset-0 before:bg-gradient-to-br",
      "before:from-emerald-400/5 before:via-transparent before:to-cyan-400/5",
      "before:opacity-0 group-hover:before:opacity-100 before:transition-opacity before:duration-500",
      "after:absolute after:top-0 after:left-4 after:right-4 after:h-[1px]",
      "after:bg-gradient-to-r after:from-transparent after:via-emerald-400/60 after:to-transparent"
    ),
    
    analytics: cn(
      "rounded-2xl backdrop-blur-md",
      "bg-gradient-to-br from-purple-900/20 via-slate-900/85 to-indigo-900/20",
      "border border-purple-500/30 hover:border-purple-400/50",
      "shadow-glass-md hover:shadow-glass-lg hover:shadow-purple-500/20",
      "relative group",
      "before:absolute before:inset-0 before:bg-gradient-to-br",
      "before:from-purple-400/5 before:via-transparent before:to-indigo-400/5",
      "before:opacity-0 group-hover:before:opacity-100 before:transition-opacity before:duration-500"
    ),
    
    status: cn(
      "rounded-xl backdrop-blur-sm",
      "bg-gradient-to-br from-slate-900/70 via-slate-800/60 to-slate-900/80",
      "border border-slate-600/40 hover:border-amber-400/50",
      "shadow-glass-sm hover:shadow-glass-md hover:shadow-amber-500/20",
      "transition-all duration-300"
    ),
  };

  const MotionDiv = motion.div;
  
  const cardVariants = animated ? {
    ...motionLibrary.components.hoverCard,
    ...(loading && {
      animate: {
        ...motionLibrary.components.hoverCard.initial,
        backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
        transition: {
          backgroundPosition: {
            duration: 2,
            repeat: Infinity,
            ease: 'linear'
          }
        }
      }
    })
  } : {};
  
  return (
    <MotionDiv
      ref={ref}
      className={cn(
        baseClasses, 
        variantClasses[variant], 
        loading && "loading-shimmer-glass",
        className
      )}
      variants={cardVariants}
      {...(animated && {
        initial: "initial",
        ...(hover && { whileHover: "hover" }),
        whileTap: "tap"
      })}
      {...props}
    >
      {loading && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
          variants={motionLibrary.loading.shimmer}
          animate="animate"
        />
      )}
      <motion.div
        className="relative z-10"
        variants={animated ? {
          initial: { opacity: 1 },
          hover: { scale: 1.002 }
        } : {}}
      >
        {children}
      </motion.div>
    </MotionDiv>
  );
})
Card.displayName = "Card"

interface CardHeaderProps extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>, MotionProps {
  variant?: 'default' | 'professional' | 'minimal';
  animated?: boolean;
}

const CardHeader = React.forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, variant = 'default', animated = true, children, ...props }, ref) => {
  const variantClasses = {
    default: "flex flex-col space-y-2 p-6 border-b border-slate-700/30",
    professional: cn(
      "flex flex-col space-y-3 p-6 relative",
      "border-b border-gradient-to-r from-transparent via-cyan-400/30 to-transparent",
      "after:absolute after:bottom-0 after:left-6 after:right-6 after:h-[1px]",
      "after:bg-gradient-to-r after:from-transparent after:via-cyan-400/40 after:to-transparent"
    ),
    minimal: "flex flex-col space-y-2 p-4"
  };

  const MotionDiv = motion.div;
  
  return (
    <MotionDiv
      ref={ref}
      className={cn(variantClasses[variant], className)}
      {...(animated && {
        initial: { opacity: 0, y: -10 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.3, ease: motionLibrary.presets.easing.smooth }
      })}
      {...props}
    >
      <motion.div
        variants={animated ? motionLibrary.utils.createSlideAnimation('up', 5) : {}}
        initial={animated ? "initial" : undefined}
        animate={animated ? "animate" : undefined}
        transition={{ delay: 0.1 }}
      >
        {children}
      </motion.div>
    </MotionDiv>
  );
})
CardHeader.displayName = "CardHeader"

interface CardTitleProps extends Omit<React.HTMLAttributes<HTMLHeadingElement>, keyof MotionProps>, MotionProps {
  variant?: 'default' | 'professional' | 'terminal' | 'glow';
  gradient?: boolean;
  animated?: boolean;
}

const CardTitle = React.forwardRef<HTMLHeadingElement, CardTitleProps>(
  ({ className, variant = 'default', gradient = true, animated = true, children, ...props }, ref) => {
  const variantClasses = {
    default: cn(
      "text-xl font-bold leading-none tracking-tight",
      gradient && "text-gradient-primary"
    ),
    professional: cn(
      "text-xl font-bold leading-none tracking-tight text-glow",
      gradient ? "text-gradient-secondary" : "text-cyan-100"
    ),
    terminal: cn(
      "text-xl font-mono font-bold leading-none tracking-wider text-accent-glow",
      "text-cyan-300 hover:text-cyan-100 transition-colors duration-300"
    ),
    glow: cn(
      "text-xl font-bold leading-none tracking-tight text-accent-glow",
      gradient ? "text-gradient-primary" : "text-cyan-400"
    )
  };

  const MotionH3 = motion.h3;
  
  return (
    <MotionH3
      ref={ref}
      className={cn(variantClasses[variant], className)}
      {...(animated && {
        initial: { opacity: 0, x: -20 },
        animate: { opacity: 1, x: 0 },
        transition: { 
          duration: 0.4, 
          ease: motionLibrary.presets.easing.smooth,
          delay: 0.15
        }
      })}
      {...props}
    >
      {children}
    </MotionH3>
  );
})
CardTitle.displayName = "CardTitle"

interface CardDescriptionProps extends Omit<React.HTMLAttributes<HTMLParagraphElement>, keyof MotionProps>, MotionProps {
  animated?: boolean;
}

const CardDescription = React.forwardRef<HTMLParagraphElement, CardDescriptionProps>(
  ({ className, animated = true, children, ...props }, ref) => {
    const MotionP = motion.p;
    
    return (
      <MotionP
        ref={ref}
        className={cn("text-sm text-slate-400 leading-relaxed font-mono", className)}
        {...(animated && {
          initial: { opacity: 0, y: 10 },
          animate: { opacity: 1, y: 0 },
          transition: { 
            duration: 0.3, 
            ease: motionLibrary.presets.easing.smooth,
            delay: 0.25 
          }
        })}
        {...props}
      >
        {children}
      </MotionP>
    );
  }
)
CardDescription.displayName = "CardDescription"

interface CardContentProps extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>, MotionProps {
  animated?: boolean;
  staggerChildren?: boolean;
}

const CardContent = React.forwardRef<HTMLDivElement, CardContentProps>(
  ({ className, animated = true, staggerChildren = false, children, ...props }, ref) => {
    const MotionDiv = motion.div;
    
    const contentVariants = animated ? {
      initial: { opacity: 0, y: 15 },
      animate: { 
        opacity: 1, 
        y: 0,
        transition: {
          duration: 0.4,
          ease: motionLibrary.presets.easing.smooth,
          delay: 0.3,
          ...(staggerChildren && {
            staggerChildren: 0.1,
            delayChildren: 0.4
          })
        }
      }
    } : {};
    
    return (
      <MotionDiv 
        ref={ref} 
        className={cn("p-6", className)} 
        variants={contentVariants}
        initial={animated ? "initial" : undefined}
        animate={animated ? "animate" : undefined}
        {...props}
      >
        {staggerChildren ? (
          <motion.div variants={motionLibrary.lists.container}>
            {React.Children.map(children, (child, index) => (
              <motion.div
                key={index}
                variants={motionLibrary.lists.item}
                className="mb-4 last:mb-0"
              >
                {child}
              </motion.div>
            ))}
          </motion.div>
        ) : (
          children
        )}
      </MotionDiv>
    );
  }
)
CardContent.displayName = "CardContent"

interface CardFooterProps extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>, MotionProps {
  animated?: boolean;
  staggerChildren?: boolean;
}

const CardFooter = React.forwardRef<HTMLDivElement, CardFooterProps>(
  ({ className, animated = true, staggerChildren = false, children, ...props }, ref) => {
    const MotionDiv = motion.div;
    
    return (
      <MotionDiv
        ref={ref}
        className={cn(
          "flex items-center p-6 pt-0 relative",
          "before:absolute before:top-0 before:left-6 before:right-6 before:h-px",
          "before:bg-gradient-to-r before:from-transparent before:via-slate-600/40 before:to-transparent",
          className
        )}
        {...(animated && {
          initial: { opacity: 0, y: 15 },
          animate: { opacity: 1, y: 0 },
          transition: { 
            duration: 0.3, 
            ease: motionLibrary.presets.easing.smooth,
            delay: 0.45,
            ...(staggerChildren && {
              staggerChildren: 0.1,
              delayChildren: 0.5
            })
          }
        })}
        {...props}
      >
        {staggerChildren ? (
          <motion.div 
            variants={motionLibrary.lists.container}
            className="flex items-center gap-4 w-full"
          >
            {React.Children.map(children, (child, index) => (
              <motion.div
                key={index}
                variants={motionLibrary.lists.item}
              >
                {child}
              </motion.div>
            ))}
          </motion.div>
        ) : (
          children
        )}
      </MotionDiv>
    );
  }
)
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }