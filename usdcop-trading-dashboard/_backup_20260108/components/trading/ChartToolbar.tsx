import React from 'react';
import { MousePointer, Move, Target, Ruler, TrendingUp } from 'lucide-react';

interface ChartToolbarProps {
  activeTools: string[];
  onToolToggle: (tool: string) => void;
  onTimeframeChange: (timeframe: string) => void;
  activeTimeframe: string;
}

export const ChartToolbar: React.FC<ChartToolbarProps> = ({
  activeTools,
  onToolToggle,
  onTimeframeChange,
  activeTimeframe
}) => {
  const timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];

  const drawingTools = [
    { id: 'line', icon: MousePointer, label: 'Line' },
    { id: 'rect', icon: Move, label: 'Rectangle' },
    { id: 'circle', icon: Target, label: 'Circle' },
    { id: 'fib', icon: Ruler, label: 'Fibonacci' },
    { id: 'trend', icon: TrendingUp, label: 'Trendline' },
  ];

  const indicators = [
    { id: 'bollinger', label: 'Bollinger Bands' },
    { id: 'ema', label: 'EMA 20/50/200' },
    { id: 'rsi', label: 'RSI' },
    { id: 'macd', label: 'MACD' },
    { id: 'volume', label: 'Volume Profile' },
  ];

  return (
    <div className="flex items-center justify-between p-4 glass-surface rounded-xl border border-fintech-dark-700">
      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Timeframe:</span>
        <div className="flex items-center gap-1 bg-fintech-dark-800 rounded-lg p-1">
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange(tf)}
              className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                activeTimeframe === tf
                  ? 'bg-fintech-cyan-500 text-white shadow-glow-cyan'
                  : 'text-fintech-dark-400 hover:text-white'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Tools:</span>
        <div className="flex items-center gap-1">
          {drawingTools.map((tool) => (
            <button
              key={tool.id}
              onClick={() => onToolToggle(tool.id)}
              className={`p-2 rounded-lg transition-all ${
                activeTools.includes(tool.id)
                  ? 'bg-fintech-cyan-500/20 text-fintech-cyan-400 border border-fintech-cyan-500/30'
                  : 'bg-fintech-dark-800/50 text-fintech-dark-400 hover:text-white'
              }`}
              title={tool.label}
            >
              <tool.icon className="w-4 h-4" />
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Indicators:</span>
        <div className="flex items-center gap-1">
          {indicators.map((indicator) => (
            <button
              key={indicator.id}
              onClick={() => onToolToggle(indicator.id)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                activeTools.includes(indicator.id)
                  ? 'bg-fintech-purple-400/20 text-fintech-purple-400 border border-fintech-purple-400/30'
                  : 'bg-fintech-dark-800/50 text-fintech-dark-400 hover:text-white'
              }`}
            >
              {indicator.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
