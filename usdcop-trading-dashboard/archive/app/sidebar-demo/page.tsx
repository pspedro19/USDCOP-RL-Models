'use client';

import { SidebarSystemExample } from '@/components/examples/SidebarSystemExample';
import { motion } from 'framer-motion';

export default function SidebarDemoPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-purple-500/3 to-emerald-500/5" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/8 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/6 rounded-full blur-3xl" />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent mb-4">
            Sidebar System Demo
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Test the intelligent and dynamic sidebar system with real-time state monitoring, 
            responsive behavior, and persistent preferences.
          </p>
        </motion.div>

        <SidebarSystemExample />
      </div>
    </div>
  );
}