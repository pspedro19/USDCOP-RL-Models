'use client'

/**
 * Volume Profile Component
 * ========================
 *
 * Professional volume profile implementation with POC and value area visualization.
 * Integrates with TradingView-style charts for real-time trading analysis.
 */

import React, { useRef, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { VolumeProfile as VolumeProfileData, calculateVolumeProfile, CandleData } from '@/lib/technical-indicators';

interface VolumeProfileProps {
  data: CandleData[];
  width: number;
  height: number;
  priceRange: { min: number; max: number };
  onPriceHover?: (price: number, volume: number) => void;
  showLabels?: boolean;
  className?: string;
}

const VolumeProfile: React.FC<VolumeProfileProps> = ({
  data,
  width,
  height,
  priceRange,
  onPriceHover,
  showLabels = true,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Calculate volume profile data
  const volumeProfile: VolumeProfileData = useMemo(() => {
    if (!data.length) {
      return {
        levels: [],
        poc: 0,
        valueAreaHigh: 0,
        valueAreaLow: 0,
        valueAreaVolume: 0,
        totalVolume: 0
      };
    }
    return calculateVolumeProfile(data, 50);
  }, [data]);

  // Draw volume profile
  const drawVolumeProfile = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size for high DPI
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!volumeProfile.levels.length) return;

    const { levels, poc, valueAreaHigh, valueAreaLow } = volumeProfile;
    const maxVolume = Math.max(...levels.map(l => l.volume));
    const barMaxWidth = width * 0.8; // 80% of canvas width

    // Helper functions
    const priceToY = (price: number) => {
      const range = priceRange.max - priceRange.min;
      return ((priceRange.max - price) / range) * height;
    };

    const volumeToWidth = (volume: number) => {
      return (volume / maxVolume) * barMaxWidth;
    };

    // Draw value area background
    const valueAreaY1 = priceToY(valueAreaHigh);
    const valueAreaY2 = priceToY(valueAreaLow);

    ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
    ctx.fillRect(0, valueAreaY1, width, valueAreaY2 - valueAreaY1);

    // Draw volume profile bars
    levels.forEach((level, index) => {
      const y = priceToY(level.price);
      const barWidth = volumeToWidth(level.volume);
      const barHeight = Math.max(1, height / levels.length * 0.8);

      // Determine bar color
      let barColor = 'rgba(59, 130, 246, 0.6)'; // Default blue

      if (level.price === poc) {
        barColor = 'rgba(239, 68, 68, 0.8)'; // Red for POC
      } else if (level.price >= valueAreaLow && level.price <= valueAreaHigh) {
        barColor = 'rgba(16, 185, 129, 0.7)'; // Green for value area
      }

      // Draw bar with gradient
      const gradient = ctx.createLinearGradient(0, 0, barWidth, 0);
      gradient.addColorStop(0, barColor);
      gradient.addColorStop(1, barColor.replace(/[\d.]+\)$/, '0.3)'));

      ctx.fillStyle = gradient;
      ctx.fillRect(0, y - barHeight / 2, barWidth, barHeight);

      // Draw border
      ctx.strokeStyle = barColor.replace(/0\.[0-9]+\)$/, '1)');
      ctx.lineWidth = 0.5;
      ctx.strokeRect(0, y - barHeight / 2, barWidth, barHeight);
    });

    // Draw POC line
    const pocY = priceToY(poc);
    ctx.strokeStyle = 'rgba(239, 68, 68, 1)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, pocY);
    ctx.lineTo(width, pocY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw value area boundaries
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    // Value area high
    ctx.beginPath();
    ctx.moveTo(0, valueAreaY1);
    ctx.lineTo(width, valueAreaY1);
    ctx.stroke();

    // Value area low
    ctx.beginPath();
    ctx.moveTo(0, valueAreaY2);
    ctx.lineTo(width, valueAreaY2);
    ctx.stroke();

    ctx.setLineDash([]);

    // Draw labels if enabled
    if (showLabels) {
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'left';

      // POC label
      ctx.fillStyle = 'rgba(239, 68, 68, 1)';
      ctx.fillText(`POC: ${poc.toFixed(2)}`, 5, pocY - 5);

      // Value area labels
      ctx.fillStyle = 'rgba(6, 182, 212, 1)';
      ctx.fillText(`VAH: ${valueAreaHigh.toFixed(2)}`, 5, valueAreaY1 - 5);
      ctx.fillText(`VAL: ${valueAreaLow.toFixed(2)}`, 5, valueAreaY2 + 15);
    }
  };

  // Handle mouse events
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPriceHover || !volumeProfile.levels.length) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const y = event.clientY - rect.top;

    // Convert Y coordinate to price
    const priceRange_ = priceRange.max - priceRange.min;
    const price = priceRange.max - ((y / height) * priceRange_);

    // Find closest volume level
    const closestLevel = volumeProfile.levels.reduce((closest, level) => {
      return Math.abs(level.price - price) < Math.abs(closest.price - price) ? level : closest;
    }, volumeProfile.levels[0]);

    onPriceHover(closestLevel.price, closestLevel.volume);
  };

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawVolumeProfile();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [volumeProfile, width, height, priceRange, showLabels]);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className={`relative ${className}`}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        className="cursor-crosshair"
        style={{ width: `${width}px`, height: `${height}px` }}
      />

      {/* Volume Profile Stats */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="absolute bottom-2 left-2 bg-slate-900/80 backdrop-blur-xl border border-slate-700/50 rounded-lg p-2 text-xs"
      >
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-red-500 rounded"></div>
            <span className="text-red-400">POC: ${poc.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-green-500 rounded"></div>
            <span className="text-green-400">VA: ${valueAreaLow.toFixed(2)} - ${valueAreaHigh.toFixed(2)}</span>
          </div>
          <div className="text-slate-400">
            Total Vol: {(volumeProfile.totalVolume / 1000).toFixed(1)}K
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default VolumeProfile;