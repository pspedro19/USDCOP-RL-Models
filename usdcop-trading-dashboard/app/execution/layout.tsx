'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
  LayoutDashboard,
  Link2,
  Settings,
  Activity,
  History,
  Shield,
  LogOut,
  Menu,
  X,
  Zap,
  Home,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { EXECUTION_ROUTES } from '@/lib/config/execution/constants';

const navItems = [
  { href: EXECUTION_ROUTES.DASHBOARD, label: 'Dashboard', icon: LayoutDashboard },
  { href: EXECUTION_ROUTES.EXCHANGES, label: 'Exchanges', icon: Link2 },
  { href: EXECUTION_ROUTES.EXECUTIONS, label: 'Executions', icon: History },
  { href: EXECUTION_ROUTES.SETTINGS, label: 'Settings', icon: Settings },
];

export default function ExecutionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication - supports both main app auth and execution module auth
  useEffect(() => {
    const executionToken = localStorage.getItem('auth-token');
    const mainAppAuth = localStorage.getItem('isAuthenticated') === 'true' ||
                        sessionStorage.getItem('isAuthenticated') === 'true';

    // User is authenticated if they have either token
    if (executionToken || mainAppAuth) {
      setIsAuthenticated(true);
    } else if (pathname !== EXECUTION_ROUTES.LOGIN) {
      // Not authenticated, redirect to main login (not execution login)
      router.push('/login?callbackUrl=' + encodeURIComponent(pathname));
    }
  }, [pathname, router]);

  const handleLogout = () => {
    // Clear execution module auth
    localStorage.removeItem('auth-token');
    localStorage.removeItem('auth-storage');
    // Clear main app auth
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    sessionStorage.removeItem('isAuthenticated');
    sessionStorage.removeItem('username');
    // Redirect to main login
    router.push('/login');
  };

  // Show login page without layout
  if (pathname === EXECUTION_ROUTES.LOGIN || pathname === EXECUTION_ROUTES.REGISTER) {
    return <>{children}</>;
  }

  // Don't render until auth check completes
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black flex">
      {/* Sidebar - Desktop */}
      <aside className="hidden lg:flex lg:flex-col lg:w-64 bg-gray-900/50 border-r border-gray-800/50">
        {/* Logo */}
        <div className="p-6 border-b border-gray-800/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">SignalBridge</h1>
              <p className="text-xs text-gray-500">Execution Module</p>
            </div>
          </div>
        </div>

        {/* Back to Hub */}
        <div className="px-4 pt-4">
          <button
            onClick={() => router.push('/hub')}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10 border border-gray-700/50 hover:border-cyan-500/30 transition-all duration-200"
          >
            <Home className="w-4 h-4" />
            <span className="text-sm font-medium">Volver al Hub</span>
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <button
                key={item.href}
                onClick={() => router.push(item.href)}
                className={`
                  w-full flex items-center gap-3 px-4 py-3 rounded-lg
                  transition-all duration-200
                  ${isActive
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Bottom Section */}
        <div className="p-4 border-t border-gray-800/50 space-y-2">
          {/* Kill Switch Status Indicator */}
          <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 rounded-lg">
            <Shield className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Kill Switch</span>
            <span className="ml-auto text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full">
              OK
            </span>
          </div>

          {/* Logout */}
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
          >
            <LogOut className="w-5 h-5" />
            <span className="font-medium">Logout</span>
          </button>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-white">SignalBridge</span>
          </div>
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 text-gray-400 hover:text-white"
          >
            {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="lg:hidden fixed top-14 left-0 right-0 z-40 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50"
          >
            <nav className="p-4 space-y-1">
              {/* Back to Hub - Mobile */}
              <button
                onClick={() => {
                  router.push('/hub');
                  setIsMobileMenuOpen(false);
                }}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-cyan-400 hover:bg-cyan-500/10 border border-cyan-500/30 mb-2"
              >
                <Home className="w-5 h-5" />
                <span>Volver al Hub</span>
              </button>

              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <button
                    key={item.href}
                    onClick={() => {
                      router.push(item.href);
                      setIsMobileMenuOpen(false);
                    }}
                    className={`
                      w-full flex items-center gap-3 px-4 py-3 rounded-lg
                      ${isActive
                        ? 'bg-cyan-500/20 text-cyan-400'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                      }
                    `}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:text-red-400"
              >
                <LogOut className="w-5 h-5" />
                <span>Logout</span>
              </button>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 lg:ml-0">
        <div className="pt-14 lg:pt-0 min-h-screen">
          {children}
        </div>
      </main>
    </div>
  );
}
