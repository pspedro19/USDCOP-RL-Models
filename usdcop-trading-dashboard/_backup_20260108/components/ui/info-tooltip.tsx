'use client';

import { Info } from 'lucide-react';
import { useState } from 'react';

interface InfoTooltipProps {
  title: string;
  calculation?: string;
  meaning: string;
  example?: string;
  className?: string;
}

export function InfoTooltip({ title, calculation, meaning, example, className = '' }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        className={`inline-flex items-center justify-center rounded-full hover:bg-slate-700/50 transition-colors ${className}`}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onClick={(e) => {
          e.preventDefault();
          setIsVisible(!isVisible);
        }}
        type="button"
      >
        <Info className="h-3.5 w-3.5 text-slate-400 hover:text-cyan-400" />
      </button>

      {isVisible && (
        <div className="absolute z-50 left-1/2 -translate-x-1/2 bottom-full mb-2 w-80 bg-slate-800 border border-cyan-500/30 rounded-lg shadow-xl p-4 text-left">
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
            <div className="border-8 border-transparent border-t-slate-800"></div>
          </div>

          {/* Content */}
          <div className="space-y-3">
            <h4 className="text-sm font-bold text-cyan-400 font-mono">{title}</h4>

            {calculation && (
              <div className="space-y-1">
                <p className="text-xs text-slate-400 font-mono">Calculation:</p>
                <code className="block text-xs bg-slate-900 text-green-400 p-2 rounded font-mono">
                  {calculation}
                </code>
              </div>
            )}

            <div className="space-y-1">
              <p className="text-xs text-slate-400 font-mono">What it means:</p>
              <p className="text-xs text-slate-200 leading-relaxed">{meaning}</p>
            </div>

            {example && (
              <div className="space-y-1">
                <p className="text-xs text-slate-400 font-mono">Example:</p>
                <p className="text-xs text-slate-300 leading-relaxed bg-slate-900 p-2 rounded">
                  {example}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
