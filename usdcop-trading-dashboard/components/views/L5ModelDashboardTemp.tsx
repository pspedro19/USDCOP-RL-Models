'use client';

import { motion } from 'framer-motion';

export default function L5ModelDashboardTemp() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-6"
    >
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-cyan-500/20 rounded-2xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent mb-6">
          L5 Model Dashboard
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-emerald-500/30"
          >
            <h3 className="text-emerald-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
              Model Status
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Status</span>
                <span className="text-emerald-400 font-semibold">Active</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Accuracy</span>
                <span className="text-white font-mono">97.8%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Update</span>
                <span className="text-gray-300 text-sm">2 mins ago</span>
              </div>
            </div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-cyan-500/30"
          >
            <h3 className="text-cyan-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse" />
              Predictions
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Next Hour</span>
                <span className="text-cyan-400 font-mono">↗ +0.23%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Confidence</span>
                <span className="text-white font-mono">85.2%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Signal</span>
                <span className="text-emerald-400 font-semibold">BUY</span>
              </div>
            </div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-900/50 rounded-xl p-6 border border-purple-500/30"
          >
            <h3 className="text-purple-400 font-semibold mb-4 flex items-center gap-2">
              <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse" />
              Performance
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate</span>
                <span className="text-purple-400 font-mono">73.5%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Profit Factor</span>
                <span className="text-white font-mono">2.14</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Sharpe Ratio</span>
                <span className="text-gray-300 font-mono">1.83</span>
              </div>
            </div>
          </motion.div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-orange-400 font-semibold mb-4 flex items-center gap-2">
            <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse" />
            Neural Network Visualization
          </h3>
          <div className="h-64 bg-slate-800/50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <motion.div
                animate={{ 
                  boxShadow: [
                    '0 0 20px rgba(249, 115, 22, 0.5)',
                    '0 0 40px rgba(249, 115, 22, 0.8)',
                    '0 0 20px rgba(249, 115, 22, 0.5)',
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
                className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-orange-500/20 to-red-600/20 flex items-center justify-center"
              >
                <div className="w-8 h-8 bg-orange-400 rounded-full animate-pulse" />
              </motion.div>
              <p className="text-gray-400 text-lg">Neural Network Visualization</p>
              <p className="text-gray-500 text-sm mt-2">Advanced ML model insights coming soon</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}