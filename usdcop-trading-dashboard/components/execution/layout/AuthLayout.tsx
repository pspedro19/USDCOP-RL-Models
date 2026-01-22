import { Outlet } from 'react-router-dom'
import { Rocket } from 'lucide-react'
import { ToastContainer } from '@/components/ui/toast'

export function AuthLayout() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-[500px] h-[500px] bg-gradient-to-r from-terminal-accent/20 via-terminal-purple/20 to-terminal-accent/20 rounded-full blur-3xl -top-20 -left-20 animate-pulse" />
        <div className="absolute w-[400px] h-[400px] bg-gradient-to-r from-terminal-emerald/15 to-terminal-accent/15 rounded-full blur-3xl bottom-20 right-10" />
        <div className="absolute w-[300px] h-[300px] bg-gradient-to-r from-terminal-purple/10 to-terminal-blue/10 rounded-full blur-3xl top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>

      {/* Content */}
      <div className="relative w-full max-w-md">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-terminal-accent to-terminal-purple shadow-lg shadow-cyan-500/30 mb-4">
            <Rocket className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gradient-primary">
            SignalBridge
          </h1>
          <p className="text-text-secondary mt-2">
            Professional Trading Signal Platform
          </p>
        </div>

        {/* Auth Form */}
        <div className="glass-card p-8">
          <Outlet />
        </div>

        {/* Footer */}
        <p className="text-center text-text-muted text-sm mt-6">
          &copy; 2026 SignalBridge. All rights reserved.
        </p>
      </div>

      {/* Toast Notifications */}
      <ToastContainer />
    </div>
  )
}
