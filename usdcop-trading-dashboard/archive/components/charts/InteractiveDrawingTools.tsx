'use client';

/**
 * Interactive Drawing Tools Component
 * Professional chart annotation tools for technical analysis
 * Features: Trendlines, Fibonacci retracements, Support/Resistance levels, Channels, Shapes
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Square,
  Circle,
  Triangle,
  Pen,
  Eraser,
  Undo,
  Redo,
  Save,
  Download,
  Upload,
  Trash2,
  Target,
  Ruler,
  Grid3X3,
  Palette,
  Settings,
  Eye,
  EyeOff,
  MousePointer,
  Move,
  RotateCcw
} from 'lucide-react';

interface DrawingPoint {
  x: number;
  y: number;
  price: number;
  timestamp: string;
}

interface DrawingObject {
  id: string;
  type: 'trendline' | 'fibonacci' | 'support' | 'resistance' | 'channel' | 'rectangle' | 'circle' | 'arrow' | 'text';
  points: DrawingPoint[];
  color: string;
  thickness: number;
  style: 'solid' | 'dashed' | 'dotted';
  label?: string;
  visible: boolean;
  locked: boolean;
  created: Date;
}

interface DrawingToolsProps {
  data: any[];
  width: number;
  height: number;
  xScale: (timestamp: string) => number;
  yScale: (price: number) => number;
  onDrawingsChange?: (drawings: DrawingObject[]) => void;
}

type DrawingMode = 'select' | 'trendline' | 'fibonacci' | 'support' | 'resistance' | 'channel' | 'rectangle' | 'circle' | 'arrow' | 'text';

export const InteractiveDrawingTools: React.FC<DrawingToolsProps> = ({
  data,
  width,
  height,
  xScale,
  yScale,
  onDrawingsChange
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  
  const [drawings, setDrawings] = useState<DrawingObject[]>([]);
  const [currentDrawing, setCurrentDrawing] = useState<DrawingObject | null>(null);
  const [drawingMode, setDrawingMode] = useState<DrawingMode>('select');
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectedDrawing, setSelectedDrawing] = useState<string | null>(null);
  
  // Drawing style states
  const [currentColor, setCurrentColor] = useState('#3b82f6');
  const [currentThickness, setCurrentThickness] = useState(2);
  const [currentStyle, setCurrentStyle] = useState<'solid' | 'dashed' | 'dotted'>('solid');
  
  // Tool panel states
  const [showGrid, setShowGrid] = useState(true);
  const [snapToGrid, setSnapToGrid] = useState(false);
  const [showPriceLabels, setShowPriceLabels] = useState(true);
  const [toolPanelExpanded, setToolPanelExpanded] = useState(true);
  
  // History for undo/redo
  const [drawingHistory, setDrawingHistory] = useState<DrawingObject[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Color palette
  const colorPalette = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
  ];

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    
    if (canvas && overlay) {
      canvas.width = width;
      canvas.height = height;
      overlay.width = width;
      overlay.height = height;
      
      drawAll();
    }
  }, [width, height]);

  // Draw all objects
  const drawAll = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid if enabled
    if (showGrid) {
      drawGrid(ctx);
    }
    
    // Draw all drawings
    drawings.forEach(drawing => {
      if (drawing.visible) {
        drawObject(ctx, drawing);
      }
    });
    
    // Highlight selected drawing
    if (selectedDrawing) {
      const selected = drawings.find(d => d.id === selectedDrawing);
      if (selected) {
        highlightDrawing(ctx, selected);
      }
    }
  }, [drawings, selectedDrawing, showGrid, width, height]);

  // Draw grid
  const drawGrid = (ctx: CanvasRenderingContext2D) => {
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([]);
    
    const gridSpacing = 20;
    
    // Vertical lines
    for (let x = 0; x < width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y < height; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  // Draw individual object
  const drawObject = (ctx: CanvasRenderingContext2D, drawing: DrawingObject) => {
    ctx.strokeStyle = drawing.color;
    ctx.fillStyle = drawing.color + '20'; // Semi-transparent fill
    ctx.lineWidth = drawing.thickness;
    
    // Set line style
    switch (drawing.style) {
      case 'dashed':
        ctx.setLineDash([5, 5]);
        break;
      case 'dotted':
        ctx.setLineDash([2, 2]);
        break;
      default:
        ctx.setLineDash([]);
    }
    
    const points = drawing.points;
    if (points.length < 1) return;
    
    switch (drawing.type) {
      case 'trendline':
        if (points.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.lineTo(points[1].x, points[1].y);
          ctx.stroke();
          
          // Extend line if needed
          if (points.length > 2) {
            const dx = points[1].x - points[0].x;
            const dy = points[1].y - points[0].y;
            ctx.lineTo(points[1].x + dx, points[1].y + dy);
            ctx.stroke();
          }
        }
        break;
        
      case 'fibonacci':
        if (points.length >= 2) {
          const start = points[0];
          const end = points[1];
          const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
          
          levels.forEach(level => {
            const y = start.y + (end.y - start.y) * level;
            ctx.beginPath();
            ctx.moveTo(Math.min(start.x, end.x), y);
            ctx.lineTo(Math.max(start.x, end.x), y);
            ctx.stroke();
            
            // Draw level label
            if (showPriceLabels) {
              ctx.fillStyle = drawing.color;
              ctx.font = '10px monospace';
              ctx.fillText(`${(level * 100).toFixed(1)}%`, Math.min(start.x, end.x) - 40, y + 3);
            }
          });
        }
        break;
        
      case 'support':
      case 'resistance':
        if (points.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.lineTo(points[1].x, points[0].y); // Horizontal line
          ctx.stroke();
          
          // Add price label
          if (showPriceLabels && drawing.label) {
            ctx.fillStyle = drawing.color;
            ctx.font = '12px monospace';
            ctx.fillText(drawing.label, points[1].x + 5, points[0].y - 5);
          }
        }
        break;
        
      case 'channel':
        if (points.length >= 4) {
          // Draw parallel lines
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.lineTo(points[1].x, points[1].y);
          ctx.moveTo(points[2].x, points[2].y);
          ctx.lineTo(points[3].x, points[3].y);
          ctx.stroke();
        }
        break;
        
      case 'rectangle':
        if (points.length >= 2) {
          const width = Math.abs(points[1].x - points[0].x);
          const height = Math.abs(points[1].y - points[0].y);
          ctx.strokeRect(
            Math.min(points[0].x, points[1].x),
            Math.min(points[0].y, points[1].y),
            width,
            height
          );
          ctx.fillRect(
            Math.min(points[0].x, points[1].x),
            Math.min(points[0].y, points[1].y),
            width,
            height
          );
        }
        break;
        
      case 'circle':
        if (points.length >= 2) {
          const radius = Math.sqrt(
            Math.pow(points[1].x - points[0].x, 2) + 
            Math.pow(points[1].y - points[0].y, 2)
          );
          ctx.beginPath();
          ctx.arc(points[0].x, points[0].y, radius, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.fill();
        }
        break;
        
      case 'arrow':
        if (points.length >= 2) {
          // Draw arrow line
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.lineTo(points[1].x, points[1].y);
          ctx.stroke();
          
          // Draw arrowhead
          const angle = Math.atan2(points[1].y - points[0].y, points[1].x - points[0].x);
          const arrowLength = 10;
          ctx.beginPath();
          ctx.moveTo(points[1].x, points[1].y);
          ctx.lineTo(
            points[1].x - arrowLength * Math.cos(angle - Math.PI / 6),
            points[1].y - arrowLength * Math.sin(angle - Math.PI / 6)
          );
          ctx.moveTo(points[1].x, points[1].y);
          ctx.lineTo(
            points[1].x - arrowLength * Math.cos(angle + Math.PI / 6),
            points[1].y - arrowLength * Math.sin(angle + Math.PI / 6)
          );
          ctx.stroke();
        }
        break;
        
      case 'text':
        if (points.length >= 1 && drawing.label) {
          ctx.fillStyle = drawing.color;
          ctx.font = `${drawing.thickness * 6}px monospace`;
          ctx.fillText(drawing.label, points[0].x, points[0].y);
        }
        break;
    }
    
    ctx.setLineDash([]); // Reset line dash
  };

  // Highlight selected drawing
  const highlightDrawing = (ctx: CanvasRenderingContext2D, drawing: DrawingObject) => {
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    
    // Draw selection handles
    drawing.points.forEach(point => {
      ctx.fillStyle = '#fbbf24';
      ctx.fillRect(point.x - 3, point.y - 3, 6, 6);
    });
    
    ctx.setLineDash([]);
  };

  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (drawingMode === 'select') return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Convert to price coordinates
    const timestamp = data[Math.floor((x / width) * data.length)]?.datetime || '';
    const price = yScale ? 0 : 0; // This would need proper inverse scaling
    
    const point: DrawingPoint = { x, y, price, timestamp };
    
    if (drawingMode === 'text') {
      const label = prompt('Enter text:');
      if (!label) return;
      
      const newDrawing: DrawingObject = {
        id: Date.now().toString(),
        type: 'text',
        points: [point],
        color: currentColor,
        thickness: currentThickness,
        style: currentStyle,
        label,
        visible: true,
        locked: false,
        created: new Date()
      };
      
      addDrawing(newDrawing);
      setDrawingMode('select');
      return;
    }
    
    setIsDrawing(true);
    setCurrentDrawing({
      id: Date.now().toString(),
      type: drawingMode as any,
      points: [point],
      color: currentColor,
      thickness: currentThickness,
      style: currentStyle,
      visible: true,
      locked: false,
      created: new Date()
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !currentDrawing) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const timestamp = data[Math.floor((x / width) * data.length)]?.datetime || '';
    const price = 0; // Proper price calculation needed
    const point: DrawingPoint = { x, y, price, timestamp };
    
    // Update current drawing
    const updatedDrawing = {
      ...currentDrawing,
      points: [currentDrawing.points[0], point]
    };
    
    setCurrentDrawing(updatedDrawing);
    
    // Draw preview on overlay
    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, width, height);
        drawObject(ctx, updatedDrawing);
      }
    }
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentDrawing) return;
    
    setIsDrawing(false);
    
    if (currentDrawing.points.length >= 2) {
      addDrawing(currentDrawing);
    }
    
    setCurrentDrawing(null);
    
    // Clear overlay
    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, width, height);
      }
    }
    
    // Return to select mode for single-click tools
    if (['support', 'resistance', 'text'].includes(drawingMode)) {
      setDrawingMode('select');
    }
  };

  // Add drawing to collection
  const addDrawing = (drawing: DrawingObject) => {
    setDrawings(prev => [...prev, drawing]);
    addToHistory([...drawings, drawing]);
    onDrawingsChange?.([...drawings, drawing]);
  };

  // History management
  const addToHistory = (newDrawings: DrawingObject[]) => {
    const newHistory = drawingHistory.slice(0, historyIndex + 1);
    newHistory.push(newDrawings);
    setDrawingHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setDrawings(drawingHistory[historyIndex - 1]);
      onDrawingsChange?.(drawingHistory[historyIndex - 1]);
    }
  };

  const redo = () => {
    if (historyIndex < drawingHistory.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setDrawings(drawingHistory[historyIndex + 1]);
      onDrawingsChange?.(drawingHistory[historyIndex + 1]);
    }
  };

  // Clear all drawings
  const clearAll = () => {
    if (confirm('Clear all drawings?')) {
      setDrawings([]);
      setSelectedDrawing(null);
      addToHistory([]);
      onDrawingsChange?.([]);
    }
  };

  // Save/Load drawings
  const saveDrawings = () => {
    const dataStr = JSON.stringify(drawings, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chart-drawings-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const loadDrawings = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const loadedDrawings = JSON.parse(event.target?.result as string);
        setDrawings(loadedDrawings);
        addToHistory(loadedDrawings);
        onDrawingsChange?.(loadedDrawings);
      } catch (error) {
        alert('Failed to load drawings file');
      }
    };
    reader.readAsText(file);
  };

  // Redraw when drawings change
  useEffect(() => {
    drawAll();
  }, [drawAll]);

  return (
    <div className="relative">
      {/* Main Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
          style={{ zIndex: 1 }}
        />
        <canvas
          ref={overlayRef}
          className="absolute top-0 left-0 cursor-crosshair"
          style={{ zIndex: 2 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      {/* Drawing Tools Panel */}
      <AnimatePresence>
        {toolPanelExpanded && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="absolute top-4 left-4 z-20"
          >
            <Card className="bg-slate-900/90 backdrop-blur-xl border-slate-700/50 p-4 min-w-[280px]">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-white">Drawing Tools</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setToolPanelExpanded(false)}
                >
                  <EyeOff className="w-4 h-4" />
                </Button>
              </div>

              {/* Tool Selection */}
              <div className="grid grid-cols-5 gap-2 mb-4">
                {[
                  { mode: 'select', icon: MousePointer, label: 'Select' },
                  { mode: 'trendline', icon: TrendingUp, label: 'Trendline' },
                  { mode: 'fibonacci', icon: Target, label: 'Fibonacci' },
                  { mode: 'support', icon: Minus, label: 'Support' },
                  { mode: 'resistance', icon: Minus, label: 'Resistance' },
                  { mode: 'channel', icon: Grid3X3, label: 'Channel' },
                  { mode: 'rectangle', icon: Square, label: 'Rectangle' },
                  { mode: 'circle', icon: Circle, label: 'Circle' },
                  { mode: 'arrow', icon: TrendingUp, label: 'Arrow' },
                  { mode: 'text', icon: Pen, label: 'Text' }
                ].map(({ mode, icon: Icon, label }) => (
                  <motion.button
                    key={mode}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setDrawingMode(mode as DrawingMode)}
                    className={`p-2 rounded-lg transition-all duration-200 ${
                      drawingMode === mode
                        ? 'bg-cyan-500 text-white'
                        : 'bg-slate-800/50 text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                    title={label}
                  >
                    <Icon className="w-4 h-4" />
                  </motion.button>
                ))}
              </div>

              {/* Color Palette */}
              <div className="mb-4">
                <label className="text-sm text-slate-400 mb-2 block">Color</label>
                <div className="flex flex-wrap gap-2">
                  {colorPalette.map(color => (
                    <button
                      key={color}
                      onClick={() => setCurrentColor(color)}
                      className={`w-6 h-6 rounded-full border-2 transition-all duration-200 ${
                        currentColor === color ? 'border-white scale-110' : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
              </div>

              {/* Style Controls */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-sm text-slate-400 mb-1 block">Thickness</label>
                  <select
                    value={currentThickness}
                    onChange={(e) => setCurrentThickness(Number(e.target.value))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-white text-sm"
                  >
                    {[1, 2, 3, 4, 5].map(thickness => (
                      <option key={thickness} value={thickness}>{thickness}px</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm text-slate-400 mb-1 block">Style</label>
                  <select
                    value={currentStyle}
                    onChange={(e) => setCurrentStyle(e.target.value as any)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-white text-sm"
                  >
                    <option value="solid">Solid</option>
                    <option value="dashed">Dashed</option>
                    <option value="dotted">Dotted</option>
                  </select>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-2 mb-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={undo}
                  disabled={historyIndex <= 0}
                  className="text-slate-400 hover:text-white"
                >
                  <Undo className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={redo}
                  disabled={historyIndex >= drawingHistory.length - 1}
                  className="text-slate-400 hover:text-white"
                >
                  <Redo className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAll}
                  className="text-red-400 hover:text-red-300"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={saveDrawings}
                  className="text-slate-400 hover:text-white"
                >
                  <Download className="w-4 h-4" />
                </Button>
                <label className="cursor-pointer">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-slate-400 hover:text-white"
                    asChild
                  >
                    <div>
                      <Upload className="w-4 h-4" />
                    </div>
                  </Button>
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={loadDrawings}
                  />
                </label>
              </div>

              {/* Display Options */}
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGrid}
                    onChange={(e) => setShowGrid(e.target.checked)}
                    className="rounded"
                  />
                  Show Grid
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showPriceLabels}
                    onChange={(e) => setShowPriceLabels(e.target.checked)}
                    className="rounded"
                  />
                  Price Labels
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={snapToGrid}
                    onChange={(e) => setSnapToGrid(e.target.checked)}
                    className="rounded"
                  />
                  Snap to Grid
                </label>
              </div>

              {/* Drawing Count */}
              <div className="mt-4 pt-4 border-t border-slate-700/50">
                <Badge className="bg-slate-800 text-slate-300 border-slate-600">
                  {drawings.length} drawings
                </Badge>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsed Tool Panel Button */}
      {!toolPanelExpanded && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setToolPanelExpanded(true)}
          className="absolute top-4 left-4 z-20 p-3 bg-slate-900/90 backdrop-blur-xl border border-slate-700/50 rounded-xl text-slate-400 hover:text-white transition-all duration-200"
        >
          <Pen className="w-5 h-5" />
        </motion.button>
      )}
    </div>
  );
};

export default InteractiveDrawingTools;