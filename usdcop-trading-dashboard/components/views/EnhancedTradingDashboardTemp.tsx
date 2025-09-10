'use client';

import { motion } from 'framer-motion';

export default function EnhancedTradingDashboardTemp() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-6"
    >
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-cyan-500/20 rounded-2xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent mb-6">
          Enhanced Trading Dashboard
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50"
          >
            <h3 className="text-cyan-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse" />
              Price Chart
            </h3>
            <div className="h-64 bg-slate-800/50 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center">
                  <div className="w-6 h-6 bg-cyan-400 rounded-full animate-pulse" />
                </div>
                <p className="text-gray-400">Interactive Chart</p>
              </div>
            </div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50"
          >
            <h3 className="text-purple-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse" />
              Trading Signals
            </h3>
            <div className="h-64 bg-slate-800/50 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-600/20 flex items-center justify-center">
                  <div className="w-6 h-6 bg-purple-400 rounded-full animate-pulse" />
                </div>
                <p className="text-gray-400">AI Signals</p>
              </div>
            </div>
          </motion.div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: 'Current Price', value: '4,127.50', change: '+0.85%', color: 'emerald' },
            { label: 'Volume', value: '2.4M', change: '+12.3%', color: 'cyan' },
            { label: 'High 24h', value: '4,135.20', change: '--', color: 'purple' },
            { label: 'Low 24h', value: '4,089.10', change: '--', color: 'orange' },
          ].map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className={`bg-slate-800/50 backdrop-blur-sm border border-${metric.color}-500/30 rounded-lg p-4 hover:border-${metric.color}-400/50 transition-colors`}
            >
              <h4 className={`text-${metric.color}-400 font-semibold text-sm mb-2`}>{metric.label}</h4>
              <p className="text-white text-xl font-bold">{metric.value}</p>
              <p className={`text-${metric.color}-400 text-sm`}>{metric.change}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}