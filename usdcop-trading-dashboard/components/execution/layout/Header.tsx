import { Bell, User, Menu } from 'lucide-react'
import { useAuthStore } from '@/stores/authStore'
import { useUIStore } from '@/stores/uiStore'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface HeaderProps {
  title?: string
}

export function Header({ title }: HeaderProps) {
  const { user } = useAuthStore()
  const { toggleMobileSidebar } = useUIStore()

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-slate-700/50 bg-terminal-bg/80 backdrop-blur-lg px-6">
      {/* Left side */}
      <div className="flex items-center gap-4">
        <button
          onClick={toggleMobileSidebar}
          className="lg:hidden p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-slate-800"
        >
          <Menu className="h-5 w-5" />
        </button>
        {title && (
          <h1 className="text-xl font-semibold text-text-primary">{title}</h1>
        )}
      </div>

      {/* Right side */}
      <div className="flex items-center gap-4">
        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-terminal-accent animate-pulse" />
        </Button>

        {/* User Menu */}
        <div className="flex items-center gap-3 pl-4 border-l border-slate-700">
          <div className="hidden sm:block text-right">
            <p className="text-sm font-medium text-text-primary">
              {user?.email?.split('@')[0] || 'User'}
            </p>
            <Badge variant="primary" size="sm">
              {user?.subscription_tier?.toUpperCase() || 'FREE'}
            </Badge>
          </div>
          <Button variant="ghost" size="icon" className="rounded-full bg-slate-800">
            <User className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}
