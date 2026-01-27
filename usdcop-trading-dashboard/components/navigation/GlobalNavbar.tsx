'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, BarChart3, Calendar, Home, LogOut, User,
  ChevronDown, Menu, X, Settings, Zap
} from 'lucide-react';

interface GlobalNavbarProps {
  currentPage?: 'hub' | 'dashboard' | 'forecasting' | 'signalbridge';
}

export function GlobalNavbar({ currentPage = 'hub' }: GlobalNavbarProps) {
  const router = useRouter();
  const [username, setUsername] = useState<string>('');
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    // Get username from storage
    const storedUsername = localStorage.getItem('username') || sessionStorage.getItem('username') || 'Usuario';
    setUsername(storedUsername);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    sessionStorage.removeItem('isAuthenticated');
    sessionStorage.removeItem('username');
    router.push('/login');
  };

  const handleNavigate = (path: string) => {
    setIsMobileMenuOpen(false);
    router.push(path);
  };

  const navItems = [
    { id: 'hub', label: 'Inicio', icon: Home, href: '/hub' },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3, href: '/dashboard' },
    { id: 'forecasting', label: 'Forecasting', icon: Calendar, href: '/forecasting' },
    { id: 'signalbridge', label: 'SignalBridge', icon: Zap, href: '/execution/dashboard' },
  ];

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#030712] border-b border-gray-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">

            {/* Logo */}
            <button
              onClick={() => handleNavigate('/hub')}
              className="flex items-center gap-2 group"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg"
              >
                <Activity className="w-5 h-5 text-white" />
              </motion.div>
              <span className="text-lg font-bold text-white hidden sm:block group-hover:text-cyan-400 transition-colors">
                USD/COP
              </span>
            </button>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPage === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigate(item.href)}
                    className={`
                      flex items-center gap-2 px-4 py-2 rounded-lg
                      transition-all duration-200
                      ${isActive
                        ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-white border border-cyan-500/30'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                      }
                    `}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : ''}`} />
                    <span className="text-sm font-medium">{item.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Profile Dropdown */}
            <div className="flex items-center gap-3">
              {/* Desktop Profile */}
              <div className="relative hidden md:block">
                <button
                  onClick={() => setIsProfileOpen(!isProfileOpen)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700/50 transition-all duration-200"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-sm text-gray-300 max-w-24 truncate">
                    {username}
                  </span>
                  <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${isProfileOpen ? 'rotate-180' : ''}`} />
                </button>

                <AnimatePresence>
                  {isProfileOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 mt-2 w-48 rounded-xl bg-gray-900/95 backdrop-blur-xl border border-gray-800 shadow-xl overflow-hidden"
                    >
                      <div className="p-3 border-b border-gray-800">
                        <p className="text-sm font-medium text-white truncate">{username}</p>
                        <p className="text-xs text-gray-500">Trader Account</p>
                      </div>
                      <div className="p-2">
                        <button
                          onClick={handleLogout}
                          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          <span className="text-sm">Cerrar Sesion</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Mobile Menu Button */}
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="md:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800/50 transition-colors"
              >
                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="md:hidden border-t border-gray-800/50 bg-black/95 backdrop-blur-xl"
            >
              <div className="px-4 py-4 space-y-2">
                {/* Mobile Nav Items */}
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentPage === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavigate(item.href)}
                      className={`
                        w-full flex items-center gap-3 px-4 py-3 rounded-lg
                        transition-all duration-200
                        ${isActive
                          ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-white border border-cyan-500/30'
                          : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                        }
                      `}
                    >
                      <Icon className={`w-5 h-5 ${isActive ? 'text-cyan-400' : ''}`} />
                      <span className="font-medium">{item.label}</span>
                    </button>
                  );
                })}

                {/* Mobile Profile Section */}
                <div className="pt-4 mt-4 border-t border-gray-800">
                  <div className="flex items-center gap-3 px-4 py-2 mb-2">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 flex items-center justify-center">
                      <User className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">{username}</p>
                      <p className="text-xs text-gray-500">Trader Account</p>
                    </div>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <LogOut className="w-5 h-5" />
                    <span className="font-medium">Cerrar Sesion</span>
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Backdrop for profile dropdown */}
      {isProfileOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsProfileOpen(false)}
        />
      )}
    </>
  );
}
