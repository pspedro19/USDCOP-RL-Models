import { NavLink, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Link2,
  Settings2,
  Radio,
  TrendingUp,
  Wallet,
  Settings,
  LogOut,
  ChevronLeft,
  Rocket,
} from 'lucide-react'
import { useAuthStore } from '@/stores/authStore'
import { useUIStore } from '@/stores/uiStore'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Exchanges', href: '/exchanges', icon: Link2 },
  { name: 'Trading Config', href: '/trading', icon: Settings2 },
  { name: 'Signals', href: '/signals', icon: Radio },
  { name: 'Executions', href: '/executions', icon: TrendingUp },
  { name: 'Portfolio', href: '/portfolio', icon: Wallet },
]

const bottomNavigation = [
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const location = useLocation()
  const { logout } = useAuthStore()
  const { sidebarCollapsed, toggleSidebar } = useUIStore()

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen transition-all duration-300 ease-in-out',
        'glass-surface-elevated border-r border-slate-700/50',
        sidebarCollapsed ? 'w-20' : 'w-64'
      )}
    >
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-16 items-center justify-between px-4 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-terminal-accent to-terminal-purple">
              <Rocket className="h-5 w-5 text-white" />
            </div>
            {!sidebarCollapsed && (
              <span className="text-lg font-bold text-gradient-primary">
                SignalBridge
              </span>
            )}
          </div>
          <button
            onClick={toggleSidebar}
            className={cn(
              'p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-slate-800 transition-all',
              sidebarCollapsed && 'rotate-180'
            )}
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-4">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href ||
              (item.href !== '/dashboard' && location.pathname.startsWith(item.href))
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={cn(
                  'sidebar-item',
                  isActive && 'active',
                  sidebarCollapsed && 'justify-center px-2'
                )}
                title={sidebarCollapsed ? item.name : undefined}
              >
                <item.icon className="h-5 w-5 shrink-0" />
                {!sidebarCollapsed && <span>{item.name}</span>}
              </NavLink>
            )
          })}
        </nav>

        {/* Bottom Navigation */}
        <div className="border-t border-slate-700/50 p-4 space-y-1">
          {bottomNavigation.map((item) => {
            const isActive = location.pathname.startsWith(item.href)
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={cn(
                  'sidebar-item',
                  isActive && 'active',
                  sidebarCollapsed && 'justify-center px-2'
                )}
                title={sidebarCollapsed ? item.name : undefined}
              >
                <item.icon className="h-5 w-5 shrink-0" />
                {!sidebarCollapsed && <span>{item.name}</span>}
              </NavLink>
            )
          })}
          <button
            onClick={logout}
            className={cn(
              'sidebar-item w-full text-red-400 hover:text-red-300 hover:bg-red-500/10',
              sidebarCollapsed && 'justify-center px-2'
            )}
            title={sidebarCollapsed ? 'Logout' : undefined}
          >
            <LogOut className="h-5 w-5 shrink-0" />
            {!sidebarCollapsed && <span>Logout</span>}
          </button>
        </div>
      </div>
    </aside>
  )
}
