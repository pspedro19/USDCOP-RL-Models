"use client";

import * as React from "react";
import { ProfessionalTradingInterface } from "@/components/ui/professional-trading-interface";
import { toast } from "react-hot-toast";

// Mock chart data for demonstration
const generateMockData = () => {
  const data = [];
  const now = Date.now();
  let price = 4200;

  for (let i = 0; i < 100; i++) {
    const time = now - (100 - i) * 5 * 60 * 1000; // 5-minute intervals
    const change = (Math.random() - 0.5) * 20;
    price += change;

    const open = price;
    const high = price + Math.random() * 15;
    const low = price - Math.random() * 15;
    const close = low + Math.random() * (high - low);

    data.push({
      time,
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 1000000),
    });

    price = close;
  }

  return data;
};

interface TradingTerminalExampleProps {
  className?: string;
}

export const TradingTerminalExample: React.FC<TradingTerminalExampleProps> = ({
  className
}) => {
  const [chartData, setChartData] = React.useState(() => generateMockData());
  const [currentTimeframe, setCurrentTimeframe] = React.useState("5m");
  const [activeTool, setActiveTool] = React.useState<string | null>(null);

  // Simulate real-time data updates
  React.useEffect(() => {
    const interval = setInterval(() => {
      setChartData(prev => {
        const newData = [...prev];
        const lastCandle = newData[newData.length - 1];
        const change = (Math.random() - 0.5) * 10;

        // Update last candle or add new one
        const updatedCandle = {
          ...lastCandle,
          close: lastCandle.close + change,
          high: Math.max(lastCandle.high, lastCandle.close + change),
          low: Math.min(lastCandle.low, lastCandle.close + change),
          volume: lastCandle.volume + Math.floor(Math.random() * 10000),
        };

        newData[newData.length - 1] = updatedCandle;
        return newData;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Handlers for various actions
  const handleTimeframeChange = React.useCallback((timeframe: string) => {
    setCurrentTimeframe(timeframe);
    toast.success(`Switched to ${timeframe} timeframe`, {
      duration: 1500,
      position: "bottom-right",
    });

    // Here you would typically fetch new data for the timeframe
    console.log(`Fetching data for ${timeframe} timeframe`);
  }, []);

  const handleDrawingTool = React.useCallback((tool: string) => {
    setActiveTool(tool);
    toast.success(`${tool} tool activated`, {
      duration: 1500,
      position: "bottom-right",
    });

    // Here you would activate the drawing tool in your chart library
    console.log(`Activating ${tool} drawing tool`);
  }, []);

  const handleTradingAction = React.useCallback((action: string) => {
    switch (action) {
      case "buy":
        toast.success("Buy order dialog opened", {
          duration: 2000,
          position: "top-center",
        });
        console.log("Opening buy order dialog");
        break;
      case "sell":
        toast.success("Sell order dialog opened", {
          duration: 2000,
          position: "top-center",
        });
        console.log("Opening sell order dialog");
        break;
      case "close":
        toast.success("Closing all positions", {
          duration: 2000,
          position: "top-center",
        });
        console.log("Closing all positions");
        break;
      case "cancel":
        toast.success("Cancelling all orders", {
          duration: 2000,
          position: "top-center",
        });
        console.log("Cancelling all orders");
        break;
      default:
        console.log(`Trading action: ${action}`);
    }
  }, []);

  const handleChartAction = React.useCallback((action: string) => {
    if (action.startsWith("zoom:")) {
      const [, scale, center] = action.split(":");
      console.log(`Zooming to ${scale} at center ${center}`);
      return;
    }

    if (action.startsWith("pan:")) {
      const [, delta] = action.split(":");
      console.log(`Panning by ${delta}`);
      return;
    }

    switch (action) {
      case "save":
        toast.success("Chart saved", {
          duration: 1500,
          position: "bottom-right",
        });
        console.log("Saving chart configuration");
        break;
      case "undo":
        toast.success("Undid last action", {
          duration: 1500,
          position: "bottom-right",
        });
        console.log("Undoing last action");
        break;
      case "redo":
        toast.success("Redid last action", {
          duration: 1500,
          position: "bottom-right",
        });
        console.log("Redoing last action");
        break;
      case "resetZoom":
        toast.success("Reset zoom", {
          duration: 1500,
          position: "bottom-right",
        });
        console.log("Resetting zoom level");
        break;
      case "pan":
        setActiveTool("pan");
        toast.success("Pan mode activated", {
          duration: 1500,
          position: "bottom-right",
        });
        break;
      default:
        console.log(`Chart action: ${action}`);
    }
  }, []);

  // Mock chart component (replace with your actual chart)
  const MockChart = () => (
    <div className="w-full h-full bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center relative">
      {/* Chart background grid */}
      <svg className="absolute inset-0 w-full h-full opacity-10">
        <defs>
          <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#374151" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

      {/* Mock candlestick chart */}
      <div className="relative w-full h-full p-8">
        <div className="text-center mb-4">
          <h2 className="text-2xl font-bold text-white mb-2">USD/COP</h2>
          <div className="flex items-center justify-center space-x-4 text-sm text-gray-400">
            <span>Timeframe: {currentTimeframe}</span>
            <span>•</span>
            <span>Last: {chartData[chartData.length - 1]?.close.toFixed(2)}</span>
            <span>•</span>
            <span className="text-green-400">+12.45 (0.29%)</span>
          </div>
        </div>

        {/* Mock price chart visualization */}
        <div className="relative h-full flex items-end justify-center space-x-1">
          {chartData.slice(-20).map((candle, index) => {
            const height = Math.max(10, (candle.high - candle.low) * 2);
            const isGreen = candle.close > candle.open;

            return (
              <div
                key={index}
                className="relative flex flex-col items-center"
                style={{ height: `${height}px` }}
              >
                {/* Wick */}
                <div
                  className={`w-0.5 ${isGreen ? 'bg-green-400' : 'bg-red-400'}`}
                  style={{
                    height: `${(candle.high - Math.max(candle.open, candle.close)) * 2}px`
                  }}
                />

                {/* Body */}
                <div
                  className={`w-3 ${isGreen ? 'bg-green-400' : 'bg-red-400'} ${
                    isGreen ? 'bg-opacity-80' : 'bg-opacity-80'
                  }`}
                  style={{
                    height: `${Math.abs(candle.close - candle.open) * 2}px`
                  }}
                />

                {/* Lower wick */}
                <div
                  className={`w-0.5 ${isGreen ? 'bg-green-400' : 'bg-red-400'}`}
                  style={{
                    height: `${(Math.min(candle.open, candle.close) - candle.low) * 2}px`
                  }}
                />
              </div>
            );
          })}
        </div>

        {/* Active tool indicator */}
        {activeTool && (
          <div className="absolute top-4 left-4 px-3 py-1 bg-blue-500/20 border border-blue-500/30 rounded-lg text-blue-300 text-sm">
            Active: {activeTool}
          </div>
        )}

        {/* Chart info overlay */}
        <div className="absolute bottom-4 right-4 text-xs text-gray-500 space-y-1">
          <div>Right-click for context menu</div>
          <div>Cmd+K for command palette</div>
          <div>Scroll to zoom, drag to pan</div>
          <div>Use keyboard shortcuts</div>
        </div>
      </div>
    </div>
  );

  return (
    <div className={`w-full h-screen ${className}`}>
      <ProfessionalTradingInterface
        chartData={chartData}
        onTimeframeChange={handleTimeframeChange}
        onDrawingTool={handleDrawingTool}
        onTradingAction={handleTradingAction}
        onChartAction={handleChartAction}
      >
        <MockChart />
      </ProfessionalTradingInterface>
    </div>
  );
};

export default TradingTerminalExample;