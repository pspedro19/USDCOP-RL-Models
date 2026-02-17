'use client';

import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import {
  BarChart3, TrendingUp, Activity, ChevronRight,
  Calendar, LineChart, Zap, Target, ArrowRight, Cpu, FlaskConical
} from 'lucide-react';
import { GlobalNavbar } from '@/components/navigation/GlobalNavbar';

export default function HubPage() {
  const router = useRouter();

  const menuOptions = [
    {
      id: 'dashboard',
      title: 'Trading Dashboard',
      subtitle: 'Backtest y analisis',
      description: 'Visualiza precios, senales de trading, metricas de rendimiento y el historial de operaciones del modelo RL.',
      icon: BarChart3,
      gradient: 'from-cyan-500 to-blue-600',
      glowColor: 'cyan',
      href: '/dashboard',
      features: ['Backtest Interactivo', 'Senales RL', 'Equity Curve', 'Historial de Trades']
    },
    {
      id: 'production',
      title: 'Monitor de Produccion',
      subtitle: 'Modelo en tiempo real',
      description: 'Visualiza el modelo activo en produccion durante horario de mercado. Equity curve, posicion actual y P&L en vivo.',
      icon: Cpu,
      gradient: 'from-green-500 to-teal-600',
      glowColor: 'green',
      href: '/production',
      features: ['Modelo Activo', 'Equity NRT', 'Posicion Actual', 'P&L en Vivo']
    },
    {
      id: 'experiments',
      title: 'Experimentos',
      subtitle: 'Aprobacion de modelos',
      description: 'Revisa y aprueba experimentos propuestos por L4. Sistema de dos votos para promocion a produccion.',
      icon: FlaskConical,
      gradient: 'from-purple-500 to-pink-600',
      glowColor: 'purple',
      href: '/dashboard', // Experiments approval is integrated in Dashboard via FloatingExperimentPanel
      features: ['Propuestas L4', 'Metricas Backtest', 'Comparacion Baseline', 'Segundo Voto']
    },
    {
      id: 'forecasting',
      title: 'Forecasting Semanal',
      subtitle: 'Predicciones a mediano plazo',
      description: 'Analiza proyecciones semanales del USD/COP basadas en modelos de series de tiempo y machine learning.',
      icon: Calendar,
      gradient: 'from-amber-500 to-orange-600',
      glowColor: 'amber',
      href: '/forecasting',
      features: ['Proyeccion 7 dias', 'Intervalos de confianza', 'Tendencias macro', 'Analisis tecnico']
    },
    {
      id: 'execution',
      title: 'SignalBridge',
      subtitle: 'Ejecucion automatizada',
      description: 'Conecta tus exchanges y ejecuta trades automaticamente basados en las senales del modelo RL.',
      icon: Zap,
      gradient: 'from-rose-500 to-red-600',
      glowColor: 'rose',
      href: '/execution/dashboard',
      features: ['Conexion Exchanges', 'Ejecucion Real', 'Gestion de Riesgo', 'Kill Switch']
    }
  ];

  const handleNavigate = (href: string) => {
    console.log('[HUB] handleNavigate called with:', href);
    router.push(href);
  };

  return (
    <div className="min-h-screen bg-black">
      <GlobalNavbar currentPage="hub" />

      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/30 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        </div>
      </div>

      {/* Main Content */}
      <main className="relative z-10 pt-20 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">

          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl"
              >
                <Activity className="w-8 h-8 text-white" />
              </motion.div>
              <h1 className="text-3xl sm:text-4xl font-bold text-white">
                Terminal USD/COP
              </h1>
            </div>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Selecciona el modulo al que deseas acceder
            </p>
          </motion.div>

          {/* Menu Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {menuOptions.map((option, index) => {
              const Icon = option.icon;
              return (
                <motion.div
                  key={option.id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.15 }}
                >
                  <button
                    data-testid={`hub-card-${option.id}`}
                    onClick={() => handleNavigate(option.href)}
                    className="w-full text-left group"
                  >
                    <div className={`
                      relative overflow-hidden rounded-2xl
                      bg-gray-900/80 backdrop-blur-xl
                      border border-gray-800/50
                      p-6 sm:p-8
                      transition-all duration-300
                      hover:border-${option.glowColor}-500/50
                      hover:shadow-lg hover:shadow-${option.glowColor}-500/20
                      hover:scale-[1.02]
                      active:scale-[0.98]
                    `}>
                      {/* Gradient Overlay on Hover */}
                      <div className={`
                        absolute inset-0 opacity-0 group-hover:opacity-10
                        bg-gradient-to-br ${option.gradient}
                        transition-opacity duration-300
                      `} />

                      {/* Icon & Title */}
                      <div className="relative flex items-start gap-4 mb-4">
                        <div className={`
                          p-3 rounded-xl bg-gradient-to-br ${option.gradient}
                          shadow-lg
                        `}>
                          <Icon className="w-7 h-7 text-white" />
                        </div>
                        <div className="flex-1">
                          <h2 className="text-xl sm:text-2xl font-bold text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:${option.gradient} transition-all duration-300">
                            {option.title}
                          </h2>
                          <p className="text-sm text-gray-400 mt-1">
                            {option.subtitle}
                          </p>
                        </div>
                        <ChevronRight className="w-6 h-6 text-gray-500 group-hover:text-white group-hover:translate-x-1 transition-all duration-300" />
                      </div>

                      {/* Description */}
                      <p className="relative text-gray-300 mb-6 leading-relaxed">
                        {option.description}
                      </p>

                      {/* Features */}
                      <div className="relative grid grid-cols-2 gap-2">
                        {option.features.map((feature, i) => (
                          <div
                            key={i}
                            className="flex items-center gap-2 text-sm text-gray-400"
                          >
                            <div className={`w-1.5 h-1.5 rounded-full bg-gradient-to-r ${option.gradient}`} />
                            {feature}
                          </div>
                        ))}
                      </div>

                      {/* CTA */}
                      <div className="relative mt-6 pt-4 border-t border-gray-800/50">
                        <div className={`
                          flex items-center gap-2 text-sm font-medium
                          text-gray-400 group-hover:text-white
                          transition-colors duration-300
                        `}>
                          <span>Acceder al modulo</span>
                          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
                        </div>
                      </div>
                    </div>
                  </button>
                </motion.div>
              );
            })}
          </div>

          {/* Quick Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="mt-12 grid grid-cols-2 sm:grid-cols-4 gap-4"
          >
            {[
              { label: 'Modelo Activo', value: 'PPO v2.4', icon: Zap, color: 'text-cyan-400' },
              { label: 'Precision', value: '67.3%', icon: Target, color: 'text-green-400' },
              { label: 'Sharpe Ratio', value: '1.84', icon: TrendingUp, color: 'text-purple-400' },
              { label: 'Trades Hoy', value: '12', icon: LineChart, color: 'text-amber-400' },
            ].map((stat, i) => {
              const Icon = stat.icon;
              return (
                <div
                  key={i}
                  className="bg-gray-900/50 backdrop-blur rounded-xl p-4 border border-gray-800/30"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Icon className={`w-4 h-4 ${stat.color}`} />
                    <span className="text-xs text-gray-500">{stat.label}</span>
                  </div>
                  <span className="text-lg font-bold text-white">{stat.value}</span>
                </div>
              );
            })}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
