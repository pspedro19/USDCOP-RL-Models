"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: "info" | "success" | "warning" | "system";
}

const BOOT_SEQUENCE: string[] = [
  "[BOOT] Initializing EXECUTION_ENGINE_V19...",
  "[BOOT] Loading neural network weights... OK",
  "[BOOT] Connecting to market data feed... OK",
  "[BOOT] Validating API credentials... OK",
  "[BOOT] Starting risk management module... OK",
  "[BOOT] Engine ready. Monitoring USD/COP pair.",
  "─".repeat(50),
];

const ROTATING_LOGS: Array<{ message: string; type: LogEntry["type"] }> = [
  { message: "Analyzing tick data [BID: {bid} | ASK: {ask}]", type: "info" },
  { message: "Alpha signal detected: CONFIDENCE {confidence}", type: "success" },
  { message: "Risk check passed. Exposure: {exposure}%", type: "success" },
  { message: "Processing order book depth... {depth} levels", type: "info" },
  { message: "Volatility spike detected: +{vol}% | Adjusting position size", type: "warning" },
  { message: "Model inference complete: {latency}ms latency", type: "info" },
  { message: "Feature extraction: {features} indicators computed", type: "info" },
  { message: "Portfolio rebalance signal: {action}", type: "success" },
  { message: "Market regime: {regime} | Adapting strategy", type: "info" },
  { message: "Slippage estimate: {slippage} pips | Within tolerance", type: "success" },
  { message: "Heartbeat: System healthy | Uptime: {uptime}h", type: "system" },
  { message: "Order executed: {side} {units} @ {price}", type: "success" },
  { message: "Stop-loss updated: {sl} | Take-profit: {tp}", type: "info" },
  { message: "Correlation matrix updated: {pairs} pairs analyzed", type: "info" },
  { message: "Drawdown monitor: Current {dd}% | Max allowed 15%", type: "warning" },
];

function generateRandomValue(type: string): string {
  switch (type) {
    case "bid":
      return (4100 + Math.random() * 50).toFixed(2);
    case "ask":
      return (4102 + Math.random() * 50).toFixed(2);
    case "confidence":
      return (0.75 + Math.random() * 0.24).toFixed(2);
    case "exposure":
      return (5 + Math.random() * 15).toFixed(1);
    case "depth":
      return String(Math.floor(10 + Math.random() * 40));
    case "vol":
      return (0.5 + Math.random() * 2).toFixed(2);
    case "latency":
      return String(Math.floor(2 + Math.random() * 15));
    case "features":
      return String(Math.floor(45 + Math.random() * 30));
    case "action":
      return ["INCREASE_LONG", "REDUCE_EXPOSURE", "HOLD", "HEDGE_PARTIAL"][
        Math.floor(Math.random() * 4)
      ];
    case "regime":
      return ["TRENDING", "MEAN_REVERTING", "HIGH_VOLATILITY", "RANGE_BOUND"][
        Math.floor(Math.random() * 4)
      ];
    case "slippage":
      return (0.1 + Math.random() * 0.8).toFixed(2);
    case "uptime":
      return (24 + Math.random() * 200).toFixed(1);
    case "side":
      return Math.random() > 0.5 ? "BUY" : "SELL";
    case "units":
      return String(Math.floor(1000 + Math.random() * 9000));
    case "price":
      return (4100 + Math.random() * 50).toFixed(2);
    case "sl":
      return (4080 + Math.random() * 20).toFixed(2);
    case "tp":
      return (4130 + Math.random() * 30).toFixed(2);
    case "pairs":
      return String(Math.floor(5 + Math.random() * 10));
    case "dd":
      return (2 + Math.random() * 8).toFixed(1);
    default:
      return "";
  }
}

function interpolateMessage(template: string): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => generateRandomValue(key));
}

function getTimestamp(): string {
  const now = new Date();
  return now.toTimeString().slice(0, 8);
}

export default function Terminal() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [bootComplete, setBootComplete] = useState(false);
  const [showCursor, setShowCursor] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const logIdRef = useRef(0);

  // Boot sequence effect
  useEffect(() => {
    let bootIndex = 0;
    const bootInterval = setInterval(() => {
      if (bootIndex < BOOT_SEQUENCE.length) {
        const newLog: LogEntry = {
          id: logIdRef.current++,
          timestamp: getTimestamp(),
          message: BOOT_SEQUENCE[bootIndex],
          type: "system",
        };
        setLogs((prev) => [...prev, newLog]);
        bootIndex++;
      } else {
        clearInterval(bootInterval);
        setBootComplete(true);
      }
    }, 400);

    return () => clearInterval(bootInterval);
  }, []);

  // Rotating logs effect
  useEffect(() => {
    if (!bootComplete) return;

    const logInterval = setInterval(() => {
      const randomLog =
        ROTATING_LOGS[Math.floor(Math.random() * ROTATING_LOGS.length)];
      const newLog: LogEntry = {
        id: logIdRef.current++,
        timestamp: getTimestamp(),
        message: interpolateMessage(randomLog.message),
        type: randomLog.type,
      };

      setLogs((prev) => {
        const updated = [...prev, newLog];
        // Keep only last 50 logs for performance
        if (updated.length > 50) {
          return updated.slice(-50);
        }
        return updated;
      });
    }, 1500 + Math.random() * 1000);

    return () => clearInterval(logInterval);
  }, [bootComplete]);

  // Auto-scroll effect
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  // Cursor blink effect
  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setShowCursor((prev) => !prev);
    }, 530);

    return () => clearInterval(cursorInterval);
  }, []);

  const getTextColor = (type: LogEntry["type"]): string => {
    switch (type) {
      case "success":
        return "text-green-500";
      case "warning":
        return "text-amber-400";
      case "system":
        return "text-slate-400";
      case "info":
      default:
        return "text-green-400";
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl">
      {/* Terminal Window */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="overflow-hidden rounded-xl border border-slate-700/50 bg-black/90 shadow-2xl shadow-black/50 backdrop-blur-sm"
      >
        {/* Title Bar */}
        <div className="flex items-center gap-2 border-b border-slate-700/50 bg-slate-900/80 px-4 py-3">
          {/* Window Controls */}
          <div className="flex gap-2">
            <div className="h-3 w-3 rounded-full bg-red-500 transition-colors hover:bg-red-400" />
            <div className="h-3 w-3 rounded-full bg-yellow-500 transition-colors hover:bg-yellow-400" />
            <div className="h-3 w-3 rounded-full bg-green-500 transition-colors hover:bg-green-400" />
          </div>
          {/* Title */}
          <div className="flex-1 text-center">
            <span className="font-mono text-xs text-slate-400 sm:text-sm">
              EXECUTION_ENGINE_V19.exe
            </span>
          </div>
          {/* Spacer for centering */}
          <div className="w-14" />
        </div>

        {/* Terminal Content */}
        <div
          ref={scrollRef}
          className="h-64 overflow-y-auto p-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-700 sm:h-80 md:h-96"
        >
          <AnimatePresence mode="popLayout">
            {logs.map((log) => (
              <motion.div
                key={log.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="mb-1"
              >
                <code
                  className={`font-mono text-xs leading-relaxed sm:text-sm ${getTextColor(
                    log.type
                  )}`}
                >
                  <span className="text-slate-500">[{log.timestamp}]</span>{" "}
                  {log.message}
                </code>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Blinking Cursor */}
          <div className="mt-1 flex items-center">
            <span className="font-mono text-xs text-green-400 sm:text-sm">
              {">"}{" "}
            </span>
            <motion.span
              animate={{ opacity: showCursor ? 1 : 0 }}
              transition={{ duration: 0.1 }}
              className="inline-block h-4 w-2 bg-green-400 sm:h-5"
            />
          </div>
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between border-t border-slate-700/50 bg-slate-900/60 px-4 py-2">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
            <span className="font-mono text-xs text-slate-400">LIVE</span>
          </div>
          <span className="font-mono text-xs text-slate-500">
            USD/COP | M5 Timeframe
          </span>
          <span className="font-mono text-xs text-slate-500">v19.2.1</span>
        </div>
      </motion.div>
    </div>
  );
}
