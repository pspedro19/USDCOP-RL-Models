'use client';

import { motion } from 'framer-motion';

export default function RealTimeChartTemp() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-6"
    >
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl border border-cyan-500/20 rounded-2xl p-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-6">
          Real-Time Chart
        </h2>
        
        <div className="h-96 bg-slate-900/50 rounded-xl flex items-center justify-center border border-slate-700/50">
          <div className="text-center">
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 20px rgba(6, 182, 212, 0.5)',
                  '0 0 40px rgba(6, 182, 212, 0.8)',
                  '0 0 20px rgba(6, 182, 212, 0.5)',
                ]
              }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center"
            >
              <div className="w-8 h-8 bg-cyan-400 rounded-full animate-pulse" />
            </motion.div>
            <p className="text-gray-400 text-lg">Real-Time Chart Coming Soon</p>
            <p className="text-gray-500 text-sm mt-2">Advanced chart functionality will be available here</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/30 rounded-lg p-4"
          >
            <h3 className="text-cyan-400 font-semibold mb-2">Live Data</h3>
            <p className="text-gray-300 text-sm">Real-time USD/COP streaming</p>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/30 rounded-lg p-4"
          >
            <h3 className="text-purple-400 font-semibold mb-2">Indicators</h3>
            <p className="text-gray-300 text-sm">Technical analysis tools</p>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/30 rounded-lg p-4"
          >
            <h3 className="text-emerald-400 font-semibold mb-2">Alerts</h3>
            <p className="text-gray-300 text-sm">Price movement notifications</p>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}