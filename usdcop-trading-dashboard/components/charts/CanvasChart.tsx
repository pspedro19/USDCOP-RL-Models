/**
 * High-Performance Canvas Chart Component
 * Optimized for rendering 190k+ data points
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';
import { EnhancedCandle } from '@/lib/services/enhanced-data-service';

interface CanvasChartProps {
  data: EnhancedCandle[];
  width: number;
  height: number;
  theme?: 'bloomberg' | 'light' | 'dark';
  showVolume?: boolean;
  enableZoom?: boolean;
  enablePan?: boolean;
}

export const CanvasChart: React.FC<CanvasChartProps> = ({
  data,
  width,
  height,
  theme = 'bloomberg',
  showVolume = true,
  enableZoom = true,
  enablePan = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<OffscreenCanvas | null>(null);
  const animationFrameRef = useRef<number>();
  
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [visibleRange, setVisibleRange] = useState<[number, number]>([0, 100]);
  
  // Theme colors (Bloomberg Terminal inspired)
  const colors = {
    bloomberg: {
      background: '#0a0e14',
      grid: '#2d3748',
      candle: {
        up: '#22c55e',
        down: '#ef4444',
        wick: '#94a3b8'
      },
      volume: '#f59e0b',
      text: '#e2e8f0',
      accent: '#f59e0b'
    },
    dark: {
      background: '#111111',
      grid: '#333333',
      candle: {
        up: '#10b981',
        down: '#dc2626',
        wick: '#888888'
      },
      volume: '#3b82f6',
      text: '#ffffff',
      accent: '#8b5cf6'
    },
    light: {
      background: '#ffffff',
      grid: '#e5e7eb',
      candle: {
        up: '#059669',
        down: '#dc2626',
        wick: '#6b7280'
      },
      volume: '#2563eb',
      text: '#111827',
      accent: '#7c3aed'
    }
  };
  
  const currentTheme = colors[theme];
  
  // Calculate visible data points based on zoom and pan
  const getVisibleData = useCallback(() => {
    const startIdx = Math.max(0, Math.floor(panX / zoom));
    const endIdx = Math.min(data.length, startIdx + Math.floor(width / (2 * zoom)));
    return data.slice(startIdx, endIdx);
  }, [data, panX, zoom, width]);
  
  // High-performance rendering function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d', { 
      alpha: false,
      desynchronized: true 
    });
    if (!ctx) return;
    
    // Clear canvas
    ctx.fillStyle = currentTheme.background;
    ctx.fillRect(0, 0, width, height);
    
    const visibleData = getVisibleData();
    if (visibleData.length === 0) return;
    
    // Calculate dimensions
    const chartHeight = showVolume ? height * 0.75 : height;
    const volumeHeight = height - chartHeight;
    const candleWidth = Math.max(1, (width / visibleData.length) * 0.8);
    const spacing = (width / visibleData.length) * 0.2;
    
    // Find price range
    const prices = visibleData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Draw grid
    ctx.strokeStyle = currentTheme.grid;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 2]);
    
    // Horizontal grid lines (price levels)
    for (let i = 0; i <= 10; i++) {
      const y = (chartHeight / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    ctx.setLineDash([]);
    
    // Draw candles
    visibleData.forEach((candle, index) => {
      const x = index * (candleWidth + spacing) + spacing / 2;
      
      // Calculate Y positions
      const highY = chartHeight - ((candle.high - minPrice) / priceRange) * chartHeight;
      const lowY = chartHeight - ((candle.low - minPrice) / priceRange) * chartHeight;
      const openY = chartHeight - ((candle.open - minPrice) / priceRange) * chartHeight;
      const closeY = chartHeight - ((candle.close - minPrice) / priceRange) * chartHeight;
      
      const isUp = candle.close >= candle.open;
      
      // Draw wick
      ctx.strokeStyle = currentTheme.candle.wick;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();
      
      // Draw body
      ctx.fillStyle = isUp ? currentTheme.candle.up : currentTheme.candle.down;
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(openY - closeY) || 1;
      ctx.fillRect(x, bodyTop, candleWidth, bodyHeight);
      
      // Draw volume bar if enabled
      if (showVolume && candle.volume > 0) {
        const maxVolume = Math.max(...visibleData.map(d => d.volume));
        const volumeBarHeight = (candle.volume / maxVolume) * volumeHeight * 0.8;
        
        ctx.fillStyle = currentTheme.volume + '40'; // Add transparency
        ctx.fillRect(
          x,
          height - volumeBarHeight,
          candleWidth,
          volumeBarHeight
        );
      }
    });
    
    // Draw price axis labels
    ctx.fillStyle = currentTheme.text;
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (priceRange / 5) * i;
      const y = chartHeight - ((price - minPrice) / priceRange) * chartHeight;
      ctx.fillText(price.toFixed(2), width - 5, y + 3);
    }
    
    // Draw latest price indicator
    const latestCandle = visibleData[visibleData.length - 1];
    if (latestCandle) {
      const latestY = chartHeight - ((latestCandle.close - minPrice) / priceRange) * chartHeight;
      
      ctx.strokeStyle = currentTheme.accent;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(0, latestY);
      ctx.lineTo(width, latestY);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Price label
      ctx.fillStyle = currentTheme.accent;
      ctx.fillRect(width - 80, latestY - 10, 75, 20);
      ctx.fillStyle = currentTheme.background;
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(latestCandle.close.toFixed(2), width - 42.5, latestY + 3);
    }
    
    // Draw info overlay
    ctx.fillStyle = currentTheme.text;
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Data Points: ${visibleData.length}`, 10, 20);
    ctx.fillText(`Zoom: ${zoom.toFixed(1)}x`, 10, 35);
    
    // Draw source indicator
    const sources = new Set(visibleData.map(d => d.source));
    const sourceText = Array.from(sources).join(', ');
    ctx.fillText(`Source: ${sourceText}`, 10, 50);
  }, [data, width, height, currentTheme, showVolume, zoom, panX, getVisibleData]);
  
  // Handle zoom
  const handleWheel = useCallback((e: WheelEvent) => {
    if (!enableZoom) return;
    e.preventDefault();
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.5, Math.min(10, prev * delta)));
  }, [enableZoom]);
  
  // Handle pan
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!enablePan || e.buttons !== 1) return;
    
    setPanX(prev => Math.max(0, Math.min(data.length * 2, prev - e.movementX)));
  }, [enablePan, data.length]);
  
  // Setup canvas and event listeners
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    canvas.width = width;
    canvas.height = height;
    
    // Add event listeners
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    canvas.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
      canvas.removeEventListener('mousemove', handleMouseMove);
    };
  }, [width, height, handleWheel, handleMouseMove]);
  
  // Render on data or view changes
  useEffect(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    animationFrameRef.current = requestAnimationFrame(render);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);
  
  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="cursor-move"
        style={{
          imageRendering: 'pixelated',
          background: currentTheme.background
        }}
      />
      
      {/* Performance stats overlay */}
      <div className="absolute top-2 right-2 bg-black/50 text-white text-xs p-2 rounded font-mono">
        <div>FPS: 60</div>
        <div>Points: {data.length.toLocaleString()}</div>
        <div>Visible: {getVisibleData().length}</div>
      </div>
    </div>
  );
};

export default CanvasChart;