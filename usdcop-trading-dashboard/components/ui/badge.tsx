import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-slate-900",
  {
    variants: {
      variant: {
        default:
          "border-slate-600/50 bg-slate-800/60 text-slate-100 hover:bg-slate-700/60 hover:border-cyan-400/50 shadow-sm hover:shadow-md hover:shadow-cyan-400/20",
        secondary:
          "border-slate-700/50 bg-slate-900/40 text-slate-200 hover:bg-slate-800/40 hover:text-white backdrop-blur-sm",
        destructive:
          "border-red-500/50 bg-gradient-to-r from-red-600/80 to-red-700/80 text-white hover:from-red-500/80 hover:to-red-600/80 shadow-lg shadow-red-500/20 hover:shadow-red-500/30",
        outline: "border-slate-600/50 bg-transparent text-slate-300 hover:bg-slate-800/30 hover:text-white hover:border-cyan-400/50",
        success: "border-emerald-500/50 bg-gradient-to-r from-emerald-600/80 to-emerald-700/80 text-white hover:from-emerald-500/80 hover:to-emerald-600/80 shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30",
        warning: "border-yellow-500/50 bg-gradient-to-r from-yellow-600/80 to-yellow-700/80 text-white hover:from-yellow-500/80 hover:to-yellow-600/80 shadow-lg shadow-yellow-500/20 hover:shadow-yellow-500/30",
        info: "border-cyan-500/50 bg-gradient-to-r from-cyan-600/80 to-cyan-700/80 text-white hover:from-cyan-500/80 hover:to-cyan-600/80 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30",
        pulse: "border-purple-500/50 bg-gradient-to-r from-purple-600/80 to-pink-600/80 text-white hover:from-purple-500/80 hover:to-pink-500/80 shadow-lg shadow-purple-500/20 hover:shadow-purple-500/30 animate-pulse",
        glow: "border-cyan-400/60 bg-gradient-to-r from-cyan-500/90 to-emerald-500/90 text-white shadow-lg shadow-cyan-400/40 hover:shadow-cyan-400/60 relative overflow-hidden",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props}>
      {/* Glow effect for glow variant */}
      {variant === 'glow' && (
        <>
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-emerald-500/20 to-cyan-500/20 animate-ping rounded-full" />
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] animate-shimmer" />
        </>
      )}
      
      {/* Pulse animation for pulse variant */}
      {variant === 'pulse' && (
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/30 to-pink-600/30 animate-ping rounded-full opacity-60" />
      )}
      
      {/* Success pulse */}
      {variant === 'success' && (
        <div className="absolute top-0 right-0 w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
      )}
      
      {/* Warning pulse */}
      {variant === 'warning' && (
        <div className="absolute top-0 right-0 w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
      )}
      
      {/* Destructive pulse */}
      {variant === 'destructive' && (
        <div className="absolute top-0 right-0 w-2 h-2 bg-red-400 rounded-full animate-ping" />
      )}
      
      {/* Info pulse */}
      {variant === 'info' && (
        <div className="absolute top-0 right-0 w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
      )}
      
      {/* Badge content with relative positioning */}
      <span className="relative z-10">{props.children}</span>
    </div>
  )
}

export { Badge, badgeVariants }