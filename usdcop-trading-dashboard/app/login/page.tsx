'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Eye, EyeOff, LogIn, Shield, Lock, User } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Prevent Web3 wallet injections on mount
  useEffect(() => {
    // Safely handle Web3 objects without causing errors
    if (typeof window !== 'undefined') {
      // Block common Web3 wallet injections
      const blockList = ['ethereum', 'web3', 'tronWeb', 'solana', 'phantom'];
      blockList.forEach(prop => {
        try {
          // Check if property exists and is configurable
          const descriptor = Object.getOwnPropertyDescriptor(window, prop);
          if (descriptor && descriptor.configurable) {
            delete (window as any)[prop];
          } else if ((window as any)[prop]) {
            // If we can't delete it, try to neutralize it
            const wallet = (window as any)[prop];
            if (wallet && typeof wallet === 'object') {
              // Disable auto-connect features
              if (wallet.autoRefreshOnNetworkChange !== undefined) {
                wallet.autoRefreshOnNetworkChange = false;
              }
              if (wallet.isMetaMask !== undefined) {
                wallet.isMetaMask = false;
              }
              // Override request method to prevent connections
              if (wallet.request) {
                wallet.request = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
              if (wallet.enable) {
                wallet.enable = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
              if (wallet.connect) {
                wallet.connect = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
            }
          }
        } catch (e) {
          // Silently ignore errors - some properties might be protected
          console.log(`Could not modify ${prop}:`, e.message);
        }
      });
    }
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // Simple authentication check
      if (username === 'admin' && password === 'admin') {
        // Use try-catch for sessionStorage in case it's blocked
        try {
          sessionStorage.setItem('isAuthenticated', 'true');
          sessionStorage.setItem('username', username);
          sessionStorage.setItem('loginTime', new Date().toISOString());
        } catch (storageError) {
          // Fallback to localStorage if sessionStorage fails
          try {
            localStorage.setItem('isAuthenticated', 'true');
            localStorage.setItem('username', username);
            localStorage.setItem('loginTime', new Date().toISOString());
          } catch (localStorageError) {
            console.error('Storage error:', localStorageError);
            // Continue anyway - auth will work for this session
          }
        }
        
        // Small delay to ensure storage is written
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Redirect to dashboard
        router.push('/');
      } else {
        setError('Invalid username or password');
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('An error occurred during login. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}} />
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-md"
      >
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-700/50 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-cyan-500/20 to-purple-500/20 p-6 border-b border-slate-700/50">
            <div className="flex items-center justify-center gap-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl"
              >
                <Shield className="w-8 h-8 text-white" />
              </motion.div>
              <div className="text-center">
                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  USD/COP Trading RL
                </h1>
                <p className="text-slate-400 text-sm mt-1">Secure Access Portal</p>
              </div>
            </div>
          </div>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="p-6 space-y-6">
            {/* Username Field */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <User className="w-4 h-4 text-cyan-400" />
                Username
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-200"
                  placeholder="Enter username"
                  required
                  autoFocus
                />
              </div>
            </div>

            {/* Password Field */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <Lock className="w-4 h-4 text-purple-400" />
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all duration-200"
                  placeholder="Enter password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors duration-200"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg"
              >
                <p className="text-red-400 text-sm">{error}</p>
              </motion.div>
            )}

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`w-full py-3 px-4 rounded-xl font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                isLoading 
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white hover:shadow-lg hover:shadow-cyan-500/25'
              }`}
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Authenticating...
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  Sign In
                </>
              )}
            </motion.button>

            {/* Hint */}
            <div className="text-center">
              <p className="text-xs text-slate-500">
                Use credentials: admin / admin
              </p>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
}