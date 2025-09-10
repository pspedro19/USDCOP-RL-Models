'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, BarChart3, DollarSign, Activity, Zap } from 'lucide-react';

interface SplashScreenProps {
  onComplete: () => void;
  duration?: number;
}

const loadingSteps = [
  { label: "Initializing trading engine", icon: TrendingUp, color: "from-cyan-400 to-blue-500" },
  { label: "Loading market data", icon: BarChart3, color: "from-blue-500 to-purple-500" },
  { label: "Connecting to data feeds", icon: Activity, color: "from-purple-500 to-pink-500" },
  { label: "Calibrating ML models", icon: Zap, color: "from-pink-500 to-emerald-500" },
  { label: "Ready for trading", icon: DollarSign, color: "from-emerald-500 to-cyan-400" }
];

export const SplashScreen: React.FC<SplashScreenProps> = ({ 
  onComplete, 
  duration = 3000 
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const stepDuration = duration / loadingSteps.length;
    const progressInterval = 10;
    const progressPerStep = 100 / loadingSteps.length;
    
    const timer = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + (progressPerStep / (stepDuration / progressInterval));
        
        // Update current step based on progress
        const newStep = Math.min(Math.floor(newProgress / progressPerStep), loadingSteps.length - 1);
        if (newStep !== currentStep) {
          setCurrentStep(newStep);
        }
        
        if (newProgress >= 100) {
          clearInterval(timer);
          setIsComplete(true);
          setTimeout(onComplete, 500);
          return 100;
        }
        
        return newProgress;
      });
    }, progressInterval);

    return () => clearInterval(timer);
  }, [duration, currentStep, onComplete]);

  const CurrentIcon = loadingSteps[currentStep]?.icon || TrendingUp;
  const currentColor = loadingSteps[currentStep]?.color || "from-cyan-400 to-blue-500";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[10000] flex items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950"
      >
        {/* Animated background */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            animate={{
              rotate: [0, 360],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-r from-cyan-500/10 via-transparent to-purple-500/10 rounded-full"
          />
          <motion.div
            animate={{
              rotate: [360, 0],
              scale: [1, 0.9, 1],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-l from-emerald-500/10 via-transparent to-blue-500/10 rounded-full"
          />
        </div>

        {/* Main content */}
        <div className="relative z-10 flex flex-col items-center space-y-8 p-8 max-w-md mx-auto text-center">
          {/* Logo/Icon */}
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ 
              type: "spring", 
              stiffness: 200, 
              damping: 20,
              delay: 0.2 
            }}
            className="relative"
          >
            <div className={`w-24 h-24 rounded-full bg-gradient-to-r ${currentColor} p-6 shadow-2xl relative overflow-hidden`}>
              <motion.div
                key={currentStep}
                initial={{ scale: 0, rotate: -90 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              >
                <CurrentIcon className="w-full h-full text-white drop-shadow-lg" />
              </motion.div>
              
              {/* Pulse effect */}
              <motion.div
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className={`absolute inset-0 rounded-full bg-gradient-to-r ${currentColor} opacity-20`}
              />
            </div>
          </motion.div>

          {/* Title */}
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          >
            <h1 className="text-4xl font-bold bg-gradient-to-r from-white via-cyan-200 to-purple-200 bg-clip-text text-transparent mb-2">
              USDCOP Pro
            </h1>
            <p className="text-slate-400 text-lg font-medium">
              Advanced Trading Dashboard
            </p>
          </motion.div>

          {/* Loading steps */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.6 }}
            className="space-y-4 w-full"
          >
            {/* Current step display */}
            <motion.div
              key={currentStep}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="flex items-center space-x-3 bg-slate-800/50 backdrop-blur-sm rounded-lg p-3 border border-slate-700/50"
            >
              <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${currentColor} animate-pulse`} />
              <span className="text-slate-300 text-sm font-medium">
                {loadingSteps[currentStep]?.label}
              </span>
            </motion.div>

            {/* Progress bar */}
            <div className="w-full bg-slate-800/50 rounded-full h-2 overflow-hidden border border-slate-700/50">
              <motion.div
                className={`h-full bg-gradient-to-r ${currentColor} rounded-full shadow-lg`}
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ ease: "easeOut" }}
              />
              
              {/* Progress glow */}
              <motion.div
                className={`absolute inset-y-0 right-0 w-4 bg-gradient-to-r ${currentColor} opacity-60 blur-sm`}
                animate={{ x: [-16, 0] }}
                transition={{ duration: 0.8, repeat: Infinity }}
                style={{ width: "16px" }}
              />
            </div>

            {/* Progress percentage */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="text-slate-400 text-sm font-mono"
            >
              {Math.round(progress)}%
            </motion.div>
          </motion.div>

          {/* Floating particles */}
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-cyan-400/60 rounded-full"
              animate={{
                y: [-20, -100, -20],
                x: [0, Math.sin(i) * 50, 0],
                opacity: [0, 1, 0],
                scale: [0, 1, 0]
              }}
              transition={{
                duration: 3 + i * 0.5,
                repeat: Infinity,
                delay: i * 0.3
              }}
              style={{
                left: `${20 + i * 10}%`,
                top: `${80 + Math.sin(i) * 10}%`
              }}
            />
          ))}
        </div>

        {/* Loading completion animation */}
        <AnimatePresence>
          {isComplete && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 50 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: "easeInOut" }}
              className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-purple-500/20 rounded-full"
            />
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
};

export default SplashScreen;