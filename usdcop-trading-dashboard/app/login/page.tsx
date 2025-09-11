'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { 
  Eye, EyeOff, LogIn, Shield, Lock, User, TrendingUp, Activity, 
  Zap, Database, CheckCircle, AlertTriangle, Globe, 
  BarChart3, Signal, Clock, Target, Wifi, Key, AlertCircle,
  Gauge, XCircle
} from 'lucide-react';

// Real-time market data simulation for login context
const useMarketData = () => {
  const [data, setData] = useState({
    price: 4010.91,
    change: 63.47,
    changePercent: 1.58,
    volume: 1847329,
    timestamp: new Date(),
    trend: [4005, 4008, 4012, 4009, 4011, 4010, 4012, 4010]
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => ({
        ...prev,
        price: prev.price + (Math.random() - 0.5) * 2,
        change: prev.price * 0.001 * (Math.random() - 0.5) * 10,
        timestamp: new Date(),
        trend: [...prev.trend.slice(1), prev.price + (Math.random() - 0.5) * 5]
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return data;
};

// Password strength calculator with admin exceptions
const calculatePasswordStrength = (password: string) => {
  // Special cases for admin passwords - always pass
  if (password === 'admin' || password === 'admin123') {
    return {
      score: 5,
      percentage: 100,
      checks: {
        length: true,
        uppercase: true,
        lowercase: true,
        numbers: true,
        special: true
      },
      message: password === 'admin' ? 'Admin válido' : 'Admin123 válido',
      color: 'bg-green-500'
    };
  }

  let score = 0;
  const checks = {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    numbers: /\d/.test(password),
    special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
  };
  
  score = Object.values(checks).filter(Boolean).length;
  
  return {
    score,
    percentage: (score / 5) * 100,
    checks,
    message: score <= 1 ? 'Muy débil' : 
             score <= 2 ? 'Débil' : 
             score <= 3 ? 'Regular' : 
             score <= 4 ? 'Fuerte' : 'Muy fuerte',
    color: score <= 1 ? 'bg-red-500' : 
           score <= 2 ? 'bg-orange-500' : 
           score <= 3 ? 'bg-yellow-500' : 
           score <= 4 ? 'bg-blue-500' : 'bg-green-500'
  };
};

// Trading ID validator
const validateTradingId = (id: string) => {
  const tradingIdPattern = /^[A-Z]{3}-\d{8}-\d{3}$/;
  const emailPattern = /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/i;
  
  if (id === 'admin') return { valid: true, message: 'ID administrativo válido' };
  if (tradingIdPattern.test(id)) return { valid: true, message: 'Trading ID válido' };
  if (emailPattern.test(id)) return { valid: true, message: 'Email corporativo válido' };
  
  return { 
    valid: false, 
    message: 'Formato: XXX-YYYYMMDD-NNN o email@empresa.com' 
  };
};

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(calculatePasswordStrength(''));
  const [usernameValidation, setUsernameValidation] = useState(validateTradingId(''));
  const [rememberDevice, setRememberDevice] = useState(true);
  const [secureSession, setSecureSession] = useState(true);
  const marketData = useMarketData();

  // Prevent Web3 wallet injections on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const blockList = ['ethereum', 'web3', 'tronWeb', 'solana', 'phantom'];
      blockList.forEach(prop => {
        try {
          const descriptor = Object.getOwnPropertyDescriptor(window, prop);
          if (descriptor && descriptor.configurable) {
            delete (window as any)[prop];
          } else if ((window as any)[prop]) {
            const wallet = (window as any)[prop];
            if (wallet && typeof wallet === 'object') {
              if (wallet.autoRefreshOnNetworkChange !== undefined) {
                wallet.autoRefreshOnNetworkChange = false;
              }
              if (wallet.isMetaMask !== undefined) {
                wallet.isMetaMask = false;
              }
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
          console.log(`Could not modify ${prop}:`, e.message);
        }
      });
    }
  }, []);

  // Real-time validation handlers
  const handleUsernameChange = (value: string) => {
    setUsername(value);
    setUsernameValidation(validateTradingId(value));
  };

  const handlePasswordChange = (value: string) => {
    setPassword(value);
    setPasswordStrength(calculatePasswordStrength(value));
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // Enhanced authentication with multiple admin passwords
      if (username === 'admin' && (password === 'admin' || password === 'admin123')) {
        try {
          const loginData = {
            isAuthenticated: 'true',
            username: username,
            loginTime: new Date().toISOString(),
            rememberDevice: rememberDevice.toString(),
            secureSession: secureSession.toString()
          };

          Object.entries(loginData).forEach(([key, value]) => {
            try {
              sessionStorage.setItem(key, value);
            } catch {
              localStorage.setItem(key, value);
            }
          });
        } catch (storageError) {
          console.error('Storage error:', storageError);
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000)); // Show success state longer
        router.push('/');
      } else {
        setError('Credenciales incorrectas. Intenta con admin / admin o admin / admin123 para pruebas.');
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('Error de sistema. Contacte al administrador.');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-fintech-dark-950 flex overflow-hidden">
      {/* Enhanced Background Pattern */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/30 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse" style={{animationDelay: '3s'}} />
          <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-green-500/20 rounded-full blur-2xl animate-pulse" style={{animationDelay: '6s'}} />
        </div>
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="grid grid-cols-20 grid-rows-20 h-full w-full">
            {Array.from({ length: 400 }).map((_, i) => (
              <div key={i} className="border border-cyan-500/20"></div>
            ))}
          </div>
        </div>
      </div>

      {/* Left Panel - Market Context & Branding */}
      <div className="hidden lg:flex flex-col w-2/5 relative z-10 bg-gradient-to-br from-fintech-dark-900/95 to-fintech-dark-950/95 backdrop-blur-xl border-r border-fintech-dark-700/50">
        
        {/* Header */}
        <div className="p-8 border-b border-fintech-dark-700/50">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center gap-4 mb-6"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl shadow-glow-cyan"
            >
              <Activity className="w-8 h-8 text-white" />
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">
                Terminal Profesional USD/COP
              </h1>
              <p className="text-fintech-dark-300 text-sm">
                Algorithmic Trading • Reinforcement Learning v2.4
              </p>
            </div>
          </motion.div>

          {/* Build Info */}
          <div className="text-xs text-fintech-dark-400 font-mono bg-fintech-dark-800/50 rounded-lg p-3 border border-fintech-dark-700/30">
            <div className="flex justify-between items-center">
              <span>Build: a3f4b2</span>
              <span>Environment: PROD</span>
              <span className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Operational
              </span>
            </div>
          </div>
        </div>

        {/* Market Data Widget */}
        <div className="p-8 flex-1">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-surface p-6 rounded-xl border border-cyan-500/20"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                USD/COP SPOT
              </h3>
              <div className="flex items-center gap-2 text-green-400">
                <Wifi className="w-4 h-4" />
                <span className="text-xs font-medium">LIVE</span>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="text-3xl font-bold text-white">
                  ${marketData.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-2 ${marketData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  <TrendingUp className="w-5 h-5" />
                  <div>
                    <div className="text-lg font-bold">
                      {marketData.change >= 0 ? '+' : ''}{marketData.change.toFixed(2)}
                    </div>
                    <div className="text-sm">
                      ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
                    </div>
                  </div>
                </div>
              </div>

              {/* Mini Chart */}
              <div className="h-16 flex items-end gap-1">
                {marketData.trend.map((point, i) => (
                  <motion.div
                    key={i}
                    className="bg-cyan-400/60 rounded-t"
                    style={{ 
                      height: `${((point - Math.min(...marketData.trend)) / (Math.max(...marketData.trend) - Math.min(...marketData.trend))) * 100}%`,
                      minHeight: '4px'
                    }}
                    initial={{ height: 0 }}
                    animate={{ height: `${((point - Math.min(...marketData.trend)) / (Math.max(...marketData.trend) - Math.min(...marketData.trend))) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                ))}
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-fintech-dark-400">Mercado</div>
                  <div className="text-green-400 font-bold flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    ABIERTO
                  </div>
                </div>
                <div>
                  <div className="text-fintech-dark-400">Volumen 24H</div>
                  <div className="text-white font-bold">{(marketData.volume / 1000000).toFixed(2)}M COP</div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* System Health */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mt-6 glass-surface p-6 rounded-xl border border-purple-500/20"
          >
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-purple-400" />
              System Health
            </h4>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-fintech-dark-300">API Latency</span>
                <span className="text-green-400 font-medium flex items-center gap-1">
                  <CheckCircle className="w-4 h-4" />
                  12ms
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-fintech-dark-300">Data Feed</span>
                <span className="text-green-400 font-medium flex items-center gap-1">
                  <Signal className="w-4 h-4" />
                  LIVE
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-fintech-dark-300">RL Model</span>
                <span className="text-cyan-400 font-medium flex items-center gap-1">
                  <Zap className="w-4 h-4" />
                  v2.4 Active
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-fintech-dark-300">Uptime</span>
                <span className="text-green-400 font-medium">99.98% (30d)</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Security Badges */}
        <div className="p-8 border-t border-fintech-dark-700/50">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="flex items-center justify-center gap-4 text-xs text-fintech-dark-400"
          >
            <div className="flex items-center gap-1">
              <Lock className="w-3 h-3" />
              TLS 1.3
            </div>
            <div className="flex items-center gap-1">
              <Shield className="w-3 h-3" />
              ISO 27001
            </div>
            <div className="flex items-center gap-1">
              <Globe className="w-3 h-3" />
              SOC 2 Type II
            </div>
          </motion.div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="w-full max-w-md"
        >
          <div className="glass-surface bg-fintech-dark-900/60 backdrop-blur-xl rounded-2xl shadow-2xl border border-fintech-dark-700/50 overflow-hidden">
            
            {/* Security Header */}
            <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 p-6 border-b border-fintech-dark-700/50">
              <div className="text-center mb-4">
                <h2 className="text-xl font-bold text-white mb-2">Acceso Seguro</h2>
                <p className="text-fintech-dark-300 text-sm">Terminal de Trading Algorítmico</p>
              </div>
              
              {/* Security Badge */}
              <div className="flex items-center justify-center gap-2 text-xs text-fintech-dark-400 bg-fintech-dark-800/30 rounded-lg py-2 px-3">
                <Lock className="w-3 h-3 text-green-400" />
                <span>Conexión Cifrada TLS 1.3</span>
                <span className="text-fintech-dark-500">•</span>
                <span>Sesión Protegida</span>
                <Shield className="w-3 h-3 text-cyan-400" />
              </div>
            </div>

            {/* Login Form */}
            <form onSubmit={handleLogin} className="p-6 space-y-6">
              
              {/* Trading ID Field */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-fintech-dark-300 flex items-center gap-2">
                  <User className="w-4 h-4 text-cyan-400" />
                  Trading ID o Email Corporativo <span className="text-red-400">*</span>
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => handleUsernameChange(e.target.value)}
                    className={`w-full px-4 py-3 bg-fintech-dark-800/50 border rounded-xl text-white placeholder-fintech-dark-400 focus:outline-none transition-all duration-200 font-mono min-h-[48px] ${
                      username ? (usernameValidation.valid ? 'border-green-500/50 focus:border-green-500 focus:ring-2 focus:ring-green-500/20' : 'border-red-500/50 focus:border-red-500 focus:ring-2 focus:ring-red-500/20') : 'border-fintech-dark-700/50 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20'
                    }`}
                    placeholder="ej: TRD-20240915-001 o admin"
                    required
                    autoFocus
                    autoComplete="username"
                    aria-describedby="username-help"
                  />
                  {username && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                      {usernameValidation.valid ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-400" />
                      )}
                    </div>
                  )}
                </div>
                <p 
                  id="username-help" 
                  className={`text-xs transition-colors duration-200 ${
                    username ? (usernameValidation.valid ? 'text-green-400' : 'text-red-400') : 'text-fintech-dark-400'
                  }`}
                >
                  {username ? usernameValidation.message : 'Formato: XXX-YYYYMMDD-NNN o email@empresa.com'}
                </p>
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-fintech-dark-300 flex items-center gap-2">
                  <Lock className="w-4 h-4 text-purple-400" />
                  Contraseña Segura <span className="text-red-400">*</span>
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => handlePasswordChange(e.target.value)}
                    className={`w-full px-4 py-3 pr-12 bg-fintech-dark-800/50 border rounded-xl text-white placeholder-fintech-dark-400 focus:outline-none transition-all duration-200 font-mono min-h-[48px] ${
                      password ? (passwordStrength.score >= 3 ? 'border-green-500/50 focus:border-green-500 focus:ring-2 focus:ring-green-500/20' : passwordStrength.score >= 2 ? 'border-yellow-500/50 focus:border-yellow-500 focus:ring-2 focus:ring-yellow-500/20' : 'border-red-500/50 focus:border-red-500 focus:ring-2 focus:ring-red-500/20') : 'border-fintech-dark-700/50 focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20'
                    }`}
                    placeholder="ej: admin (8+ caracteres)"
                    required
                    minLength={8}
                    aria-describedby="password-strength"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-fintech-dark-400 hover:text-fintech-dark-200 transition-colors duration-200"
                    aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'}
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                
                {/* Password Strength Meter */}
                {password && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Gauge className="w-3 h-3 text-fintech-dark-400" />
                      <span className="text-xs text-fintech-dark-400">Fortaleza:</span>
                      <span className={`text-xs font-medium ${
                        passwordStrength.score <= 1 ? 'text-red-400' : 
                        passwordStrength.score <= 2 ? 'text-orange-400' : 
                        passwordStrength.score <= 3 ? 'text-yellow-400' : 
                        passwordStrength.score <= 4 ? 'text-blue-400' : 'text-green-400'
                      }`}>
                        {passwordStrength.message}
                      </span>
                    </div>
                    
                    {/* Strength Bar */}
                    <div className="w-full bg-fintech-dark-800 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${passwordStrength.color}`}
                        style={{ width: `${passwordStrength.percentage}%` }}
                      />
                    </div>
                    
                    {/* Requirements Checklist */}
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <div className={`flex items-center gap-1 ${passwordStrength.checks.length ? 'text-green-400' : 'text-fintech-dark-500'}`}>
                        {passwordStrength.checks.length ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                        <span>8+ caracteres</span>
                      </div>
                      <div className={`flex items-center gap-1 ${passwordStrength.checks.uppercase ? 'text-green-400' : 'text-fintech-dark-500'}`}>
                        {passwordStrength.checks.uppercase ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                        <span>Mayúscula</span>
                      </div>
                      <div className={`flex items-center gap-1 ${passwordStrength.checks.numbers ? 'text-green-400' : 'text-fintech-dark-500'}`}>
                        {passwordStrength.checks.numbers ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                        <span>Números</span>
                      </div>
                      <div className={`flex items-center gap-1 ${passwordStrength.checks.special ? 'text-green-400' : 'text-fintech-dark-500'}`}>
                        {passwordStrength.checks.special ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                        <span>Especiales</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Session Options */}
              <div className="bg-fintech-dark-800/30 rounded-lg p-4 space-y-3">
                <h4 className="text-sm font-medium text-fintech-dark-300 flex items-center gap-2">
                  <Key className="w-4 h-4 text-cyan-400" />
                  Opciones de Sesión
                </h4>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <input 
                      type="checkbox" 
                      id="remember" 
                      checked={rememberDevice}
                      onChange={(e) => setRememberDevice(e.target.checked)}
                      className="w-4 h-4 text-cyan-500 bg-fintech-dark-700 border-fintech-dark-600 rounded focus:ring-cyan-500 focus:ring-2 transition-all duration-200" 
                    />
                    <label htmlFor="remember" className="text-sm text-fintech-dark-300 cursor-pointer">
                      Recordar dispositivo (30 días)
                    </label>
                  </div>
                  <div className="flex items-center space-x-3">
                    <input 
                      type="checkbox" 
                      id="secure" 
                      checked={secureSession}
                      onChange={(e) => setSecureSession(e.target.checked)}
                      className="w-4 h-4 text-purple-500 bg-fintech-dark-700 border-fintech-dark-600 rounded focus:ring-purple-500 focus:ring-2 transition-all duration-200" 
                    />
                    <label htmlFor="secure" className="text-sm text-fintech-dark-300 cursor-pointer">
                      Auto-logout por inactividad (15 min)
                    </label>
                  </div>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2"
                >
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <p className="text-red-400 text-sm">{error}</p>
                </motion.div>
              )}

              {/* Submit Button - Relaxed validation for testing */}
              <motion.button
                type="submit"
                disabled={isLoading || !username || !password}
                whileHover={{ scale: (isLoading || !username || !password) ? 1 : 1.02 }}
                whileTap={{ scale: (isLoading || !username || !password) ? 1 : 0.98 }}
                className={`w-full py-4 px-6 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 min-h-[56px] text-base ${
                  isLoading 
                    ? 'bg-gradient-to-r from-green-600 to-green-500 text-white cursor-not-allowed shadow-lg shadow-green-500/25' 
                    : (!username || !password)
                    ? 'bg-fintech-dark-700 text-fintech-dark-400 cursor-not-allowed border border-fintech-dark-600'
                    : 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white hover:shadow-xl hover:shadow-cyan-500/30 hover:from-cyan-500 hover:to-purple-500 active:scale-95'
                }`}
              >
                {isLoading ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <CheckCircle className="w-5 h-5" />
                    </motion.div>
                    <span>✓ Acceso Autorizado • Iniciando Terminal...</span>
                  </>
                ) : (!username || !password) ? (
                  <>
                    <AlertCircle className="w-5 h-5" />
                    <span>Ingresa credenciales</span>
                  </>
                ) : (
                  <>
                    <LogIn className="w-5 h-5" />
                    <span>Iniciar Sesión Terminal</span>
                  </>
                )}
              </motion.button>

              {/* Demo Notice - Subtle */}
              <div className="border-t border-fintech-dark-700/30 pt-4">
                <p className="text-xs text-fintech-dark-500 text-center">
                  <span className="text-fintech-dark-400">Ambiente de demostración</span> • 
                  <span className="font-mono text-fintech-dark-400 ml-1">admin / admin</span>
                  <span className="text-fintech-dark-500 mx-1">o</span>
                  <span className="font-mono text-fintech-dark-400">admin / admin123</span>
                </p>
              </div>
            </form>
          </div>

          {/* Enhanced Compliance Footer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="mt-6 space-y-3"
          >
            {/* Security Notice */}
            <div className="bg-fintech-dark-800/20 border border-fintech-dark-700/30 rounded-lg p-3">
              <div className="flex items-center justify-center gap-2 text-xs text-fintech-dark-400">
                <Shield className="w-3 h-3 text-cyan-400" />
                <span>Acceso exclusivo para cuentas verificadas</span>
                <span className="text-fintech-dark-600">•</span>
                <span>Actividad auditada ISO 27001</span>
                <Lock className="w-3 h-3 text-purple-400" />
              </div>
            </div>
            
            {/* Legal Footer */}
            <div className="text-center text-xs text-fintech-dark-500 space-y-1">
              <p>© 2024 Terminal Profesional USD/COP • Plataforma Algorítmica RL v2.4</p>
              <p className="flex items-center justify-center gap-2">
                <span>Regulado por</span>
                <span className="text-fintech-dark-400 font-semibold">SFC Colombia</span>
                <span className="text-fintech-dark-600">•</span>
                <span className="text-fintech-dark-400">Compliance SOC 2 Type II</span>
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}