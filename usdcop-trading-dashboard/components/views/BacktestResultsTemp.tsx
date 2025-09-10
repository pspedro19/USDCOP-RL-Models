'use client';

import { motion } from 'framer-motion';

export default function BacktestResultsTemp() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-6"
    >
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-cyan-500/20 rounded-2xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-6">
          Backtest Results
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50"
          >
            <h3 className="text-emerald-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
              Performance Summary
            </h3>
            <div className="space-y-4">
              {[
                { label: 'Total Return', value: '+127.8%', color: 'emerald' },
                { label: 'Annual Return', value: '+42.6%', color: 'green' },
                { label: 'Max Drawdown', value: '-8.3%', color: 'red' },
                { label: 'Volatility', value: '12.4%', color: 'yellow' },
              ].map((metric, index) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex justify-between items-center"
                >
                  <span className="text-gray-400">{metric.label}</span>
                  <span className={`text-${metric.color}-400 font-mono font-semibold`}>{metric.value}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50"
          >
            <h3 className="text-cyan-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse" />
              Trading Stats
            </h3>
            <div className="space-y-4">
              {[
                { label: 'Total Trades', value: '1,847', color: 'cyan' },
                { label: 'Win Rate', value: '67.2%', color: 'emerald' },
                { label: 'Avg Win', value: '+2.8%', color: 'green' },
                { label: 'Avg Loss', value: '-1.4%', color: 'red' },
              ].map((metric, index) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex justify-between items-center"
                >
                  <span className="text-gray-400">{metric.label}</span>
                  <span className={`text-${metric.color}-400 font-mono font-semibold`}>{metric.value}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-purple-400 font-semibold mb-4 flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse" />
            Equity Curve
          </h3>
          <div className="h-64 bg-slate-800/50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <motion.div
                animate={{ 
                  boxShadow: [
                    '0 0 20px rgba(147, 51, 234, 0.5)',
                    '0 0 40px rgba(147, 51, 234, 0.8)',
                    '0 0 20px rgba(147, 51, 234, 0.5)',
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
                className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-600/20 flex items-center justify-center"
              >
                <div className="w-8 h-8 bg-purple-400 rounded-full animate-pulse" />
              </motion.div>
              <p className="text-gray-400 text-lg">Interactive Equity Curve</p>
              <p className="text-gray-500 text-sm mt-2">Historical performance visualization</p>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {[
            { label: 'Best Month', value: '+24.7%', icon: '📈' },
            { label: 'Worst Month', value: '-5.2%', icon: '📉' },
            { label: 'Profitable Months', value: '28/36', icon: '📊' },
          ].map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/30 rounded-lg p-4 text-center"
            >
              <div className="text-2xl mb-2">{metric.icon}</div>
              <h4 className="text-gray-400 font-medium mb-1">{metric.label}</h4>
              <p className="text-white text-lg font-bold">{metric.value}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}